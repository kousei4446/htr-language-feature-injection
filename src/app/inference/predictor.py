from pathlib import Path
import json
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor

from ..dataset import read_labels, resize_keep_aspect_and_pad_right
from ..models import TrOCREncoderCTC
from ..utils import build_id2char, cer_from_pairs, ctc_greedy_decode


@torch.no_grad()
def infer_one(model, processor, vocab, device, image_path: Path, img_h: int, max_w: int) -> str:
    model.eval()
    id2char = build_id2char(vocab)
    blank_id = vocab["<blank>"]

    img = resize_keep_aspect_and_pad_right(
        Image.open(image_path),
        target_h=img_h,
        max_w=max_w,
    )
    pixel_values = processor.image_processor(
        images=img,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).pixel_values.to(device)

    logits = model(pixel_values)
    pred = ctc_greedy_decode(logits, blank_id, id2char)[0]
    return pred


def _collect_infer_images(infer_dir: Path) -> list[Path]:
    if not infer_dir.exists():
        raise FileNotFoundError(f"infer_dir not found: {infer_dir}")
    if not infer_dir.is_dir():
        raise NotADirectoryError(f"infer_dir is not a directory: {infer_dir}")

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = [
        path
        for path in sorted(infer_dir.iterdir())
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]
    if not image_paths:
        raise ValueError(f"no image files found in infer_dir: {infer_dir}")
    return image_paths


def _resolve_output_dir_from_ckpt(ckpt_path: Path) -> Path:
    parent = ckpt_path.parent
    parent_of_parent = parent.parent
    if parent_of_parent == parent:
        return parent
    return parent_of_parent


def _build_output_path(ckpt_path: Path) -> Path:
    out_dir = _resolve_output_dir_from_ckpt(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"infer_results_{ts}.jsonl"


def _maybe_load_gt_map(infer_image: Path | None, infer_dir: Path | None) -> dict[str, str]:
    candidates = []
    if infer_dir is not None:
        candidates.append(infer_dir.parent / "labels.txt")
    if infer_image is not None and infer_image.parent.parent != infer_image.parent:
        candidates.append(infer_image.parent.parent / "labels.txt")

    for labels_path in candidates:
        if labels_path.exists():
            return dict(read_labels(labels_path))
    return {}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_infer(args, device) -> None:
    infer_image = getattr(args, "infer_image", None)
    infer_dir = getattr(args, "infer_dir", None)

    if args.ckpt is None:
        raise ValueError("ckpt is required for infer")
    if infer_image is None and infer_dir is None:
        raise ValueError("either infer_image or infer_dir is required for infer")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = ckpt["vocab"]
    model_name = ckpt.get("model_name", args.model_name)
    img_h = int(ckpt.get("img_h", args.img_h))
    max_w = int(ckpt.get("max_w", args.max_w))
    upsample = int(ckpt.get("upsample", args.upsample))

    processor = TrOCRProcessor.from_pretrained(model_name)
    lail_layers = ckpt.get("lail_layers", [])
    llm_hidden_size = ckpt.get("llm_hidden_size", None)
    lail_use_upsample = bool(ckpt.get("lail_use_upsample", False))

    model = TrOCREncoderCTC(
        model_name=model_name,
        num_classes=len(vocab),
        upsample=upsample,
        lail_layers=lail_layers,
        llm_hidden_size=llm_hidden_size,
        lail_use_upsample=lail_use_upsample,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    infer_image_path = Path(infer_image) if infer_image is not None else None
    infer_dir_path = Path(infer_dir) if infer_dir is not None else None

    if infer_image_path is not None:
        image_paths = [infer_image_path]
    else:
        image_paths = _collect_infer_images(infer_dir_path)

    gt_map = _maybe_load_gt_map(infer_image_path, infer_dir_path)
    preds_eval, gts_eval = [], []
    result_rows = []

    for image_path in tqdm(image_paths, desc="infer", unit="img"):
        pred = infer_one(model, processor, vocab, device, image_path, img_h=img_h, max_w=max_w)
        img_id = image_path.stem
        record = {
            "img_id": img_id,
            "pred": pred,
        }
        gt = gt_map.get(img_id)
        if gt is not None:
            record["gt"] = gt
            preds_eval.append(pred)
            gts_eval.append(gt)
        result_rows.append(record)

    summary = {"score": len(result_rows)}
    if gts_eval:
        summary["CER"] = f"{cer_from_pairs(preds_eval, gts_eval):.4f}"

    output_path = _build_output_path(Path(args.ckpt))
    _write_jsonl(output_path, [summary, *result_rows])

    print(json.dumps(summary, ensure_ascii=False))
    print(f"[info] saved: {output_path}")
