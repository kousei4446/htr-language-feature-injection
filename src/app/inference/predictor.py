from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor

from ..dataset import resize_keep_aspect_and_pad_right
from ..models import TrOCREncoderCTC
from ..utils import build_id2char, ctc_greedy_decode


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


def run_infer(args, device) -> None:
    if args.ckpt is None or args.infer_image is None:
        raise ValueError("ckpt and infer_image are required for infer")

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

    pred = infer_one(model, processor, vocab, device, Path(args.infer_image), img_h=img_h, max_w=max_w)
    print(pred)
