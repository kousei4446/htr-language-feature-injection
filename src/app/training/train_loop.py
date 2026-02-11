import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, VisionEncoderDecoderModel

from ..dataset import IAMLineDataset, build_char_vocab, collate_fn, read_labels, resolve_split_dir
from ..models import TrOCREncoderCTC, parse_layers
from ..utils import build_id2char, cer_from_pairs, clm_loss_with_prefix, ctc_greedy_decode, encode_texts_for_llm


def _prepare_run_paths(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_base = str(args.run_name).strip() if getattr(args, "run_name", None) else "run"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_base}_{ts}"
    run_dir = out_dir / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_id = f"{run_base}_{ts}_{suffix}"
        run_dir = out_dir / run_id

    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, run_id, run_dir, ckpt_dir, tb_dir


@torch.no_grad()
def evaluate(model, loader, device, blank_id, id2char, fp16: bool):
    model.eval()
    preds_all, gts_all = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        pixel_values = batch.pixel_values.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(fp16 and device.type == "cuda")):
            logits = model(pixel_values)
        preds = ctc_greedy_decode(logits, blank_id, id2char)
        preds_all.extend(preds)
        gts_all.extend(batch.target_texts)
    return cer_from_pairs(preds_all, gts_all)


def run_train(args, device, fp16: bool, processor) -> None:
    out_dir, run_id, run_dir, ckpt_dir, tb_dir = _prepare_run_paths(args)
    writer = SummaryWriter(log_dir=str(tb_dir))

    latest_run_file = out_dir / "latest_run.txt"
    latest_run_file.write_text(str(run_dir.resolve()), encoding="utf-8")
    if OmegaConf.is_config(args):
        OmegaConf.save(config=args, f=str(run_dir / "config.resolved.yaml"))

    print(f"[info] run_id: {run_id}")
    print(f"[info] run_dir: {run_dir}")
    print(f"[info] tensorboard: {tb_dir}")
    print(f"[info] checkpoints: {ckpt_dir}")

    llm = None
    llm_tok = None
    llm_hidden = None
    lail_layers = parse_layers(args.lail_layers) if args.use_lail else []

    if args.use_lail:
        llm_tok = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).to(device)

        llm.eval()
        llm.config.use_cache = False
        for p in llm.parameters():
            p.requires_grad = False

        if args.llm_grad_ckpt and hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable()

        llm_hidden = int(llm.config.hidden_size)

    if args.data_root is None:
        raise ValueError("data_root is required for train")
    data_root = Path(args.data_root)

    train_dir = resolve_split_dir(data_root, ["train", "tarain"])
    val_dir = resolve_split_dir(data_root, ["VAL", "val"])

    train_labels = train_dir / "labels.txt"
    val_labels = val_dir / "labels.txt"

    train_items = read_labels(train_labels)
    vocab = build_char_vocab(train_items)
    id2char = build_id2char(vocab)
    blank_id = vocab["<blank>"]

    tmp = VisionEncoderDecoderModel.from_pretrained(args.model_name).encoder
    patch_size = getattr(tmp.config, "patch_size", 16)

    run_vocab_path = run_dir / "vocab.json"
    with run_vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    shutil.copy2(run_vocab_path, out_dir / "vocab.json")

    train_ds = IAMLineDataset(
        images_dir=train_dir / "images",
        labels_path=train_labels,
        processor=processor,
        vocab=vocab,
        img_h=args.img_h,
        max_w=args.max_w,
        patch_size=patch_size,
        upsample=args.upsample,
        drop_too_long=True,
    )
    val_ds = IAMLineDataset(
        images_dir=val_dir / "images",
        labels_path=val_labels,
        processor=processor,
        vocab=vocab,
        img_h=args.img_h,
        max_w=args.max_w,
        patch_size=patch_size,
        upsample=args.upsample,
        drop_too_long=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda s: collate_fn(s, pad_value=0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda s: collate_fn(s, pad_value=0),
    )

    model = TrOCREncoderCTC(
        model_name=args.model_name,
        num_classes=len(vocab),
        upsample=args.upsample,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
        lail_layers=lail_layers,
        llm_hidden_size=llm_hidden,
        lail_use_upsample=args.lail_use_upsample,
    ).to(device)

    print(f"[info] image: H={args.img_h}, W={args.max_w}  patch={model.patch_size}")
    print(
        f"[info] CTC time steps T = (W/patch)*upsample = {(args.max_w // model.patch_size) * max(1, args.upsample)}"
    )
    print(f"[info] train samples: {len(train_ds)} (dropped too-long automatically)")
    print(f"[info] val samples:   {len(val_ds)}")

    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    best = 1e9
    global_step = 0
    metrics_path = run_dir / "metrics.jsonl"

    def build_ckpt(epoch_idx: int, val_cer_value: float, best_cer: float):
        return {
            "model_state": model.state_dict(),
            "model_name": args.model_name,
            "vocab": vocab,
            "img_h": args.img_h,
            "max_w": args.max_w,
            "upsample": args.upsample,
            "patch_size": model.patch_size,
            "use_lail": args.use_lail,
            "lail_layers": model.lail_layers,
            "lail_use_upsample": args.lail_use_upsample,
            "lail_alpha": args.lail_alpha,
            "llm_name": args.llm_name if args.use_lail else None,
            "llm_hidden_size": llm_hidden if args.use_lail else None,
            "llm_max_len": args.llm_max_len if args.use_lail else None,
            "epoch": int(epoch_idx),
            "val_cer": float(val_cer_value),
            "best_cer": float(best_cer),
            "global_step": int(global_step),
            "run_id": run_id,
            "run_dir": str(run_dir),
        }

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")

        ema = None
        for batch in pbar:
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            targets = batch.targets.to(device, non_blocking=True)
            target_lengths = batch.target_lengths.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=fp16):
                if args.use_lail:
                    logits, prefixes = model(pixel_values, return_lail=True)
                else:
                    logits = model(pixel_values)

                log_probs = F.log_softmax(logits, dim=-1)
                B, T, _ = log_probs.shape
                input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
                loss_ctc = ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)

                writer.add_scalar("Loss/ctc_loss", loss_ctc, global_step)
                loss = loss_ctc

                if args.use_lail:
                    llm_input_ids, llm_attn = encode_texts_for_llm(batch.target_texts, llm_tok, args.llm_max_len)
                    llm_input_ids = llm_input_ids.to(device, non_blocking=True)
                    llm_attn = llm_attn.to(device, non_blocking=True)

                    lail_losses = []
                    for l in model.lail_layers:
                        pref = prefixes[l]
                        lail_loss = clm_loss_with_prefix(llm, llm_input_ids, llm_attn, pref)
                        lail_losses.append(lail_loss)
                        writer.add_scalar(f"Loss/lail_{l}", lail_loss.item(), global_step)

                    loss_lail = torch.stack(lail_losses).mean()
                    writer.add_scalar("Loss/lail_total", loss_lail.item(), global_step)
                    loss = loss + args.lail_alpha * loss_lail
                    writer.add_scalar("Loss/total_loss", loss, global_step)

            global_step += 1

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optim)
            scaler.update()

            v = float(loss.item())
            ema = v if ema is None else (0.95 * ema + 0.05 * v)
            pbar.set_postfix(loss=ema)

        val_cer = evaluate(model, val_loader, device, blank_id, id2char, fp16)
        print(f"[epoch {epoch}] val CER = {val_cer:.4f}")
        writer.add_scalar("CER/val", val_cer, epoch)
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_cer": float(val_cer),
                        "best_cer_before_update": float(best),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        last_ckpt = build_ckpt(epoch, val_cer, min(best, val_cer))
        torch.save(last_ckpt, ckpt_dir / "last.pt")
        if getattr(args, "save_every", 0) and args.save_every > 0 and epoch % int(args.save_every) == 0:
            torch.save(last_ckpt, ckpt_dir / f"epoch_{epoch:04d}.pt")

        if val_cer < best:
            best = val_cer
            best_ckpt = build_ckpt(epoch, val_cer, best)
            best_path = ckpt_dir / "best.pt"
            torch.save(best_ckpt, best_path)
            # Backward-compatible path for existing infer config.
            torch.save(best_ckpt, out_dir / "best.pt")
            print(f"  -> saved: {best_path} (best CER={best:.4f})")

    writer.close()
