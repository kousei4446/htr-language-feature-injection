from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def read_labels(labels_path: Path) -> List[Tuple[str, str]]:
    items = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                img_id, text = parts[0], ""
            else:
                img_id, text = parts[0], parts[1]
            items.append((img_id, text))
    return items


def build_char_vocab(train_items: List[Tuple[str, str]]) -> Dict[str, int]:
    chars = set()
    for _, text in train_items:
        for ch in text:
            chars.add(ch)

    vocab = {"<blank>": 0, "<unk>": 1}
    for i, ch in enumerate(sorted(chars), start=2):
        vocab[ch] = i
    return vocab


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text]


def find_image_file(images_dir: Path, img_id: str) -> Path:
    p = images_dir / img_id
    if p.exists():
        return p

    for ext in [".jpg"]:
        p = images_dir / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found: id={img_id} under {images_dir}")


def resize_keep_aspect_and_pad_right(
    img: Image.Image,
    target_h: int,
    max_w: int,
    pad_value: int = 255,
) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size")

    scale = target_h / h
    new_w = int(round(w * scale))
    new_w = max(1, new_w)

    if new_w > max_w:
        scale2 = max_w / new_w
        new_w = max_w
        new_h = int(round(target_h * scale2))
        new_h = max(1, new_h)
        img_rs = img.resize((new_w, new_h), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (max_w, target_h), (pad_value, pad_value, pad_value))
        canvas.paste(img_rs, (0, 0))
        return canvas

    img_rs = img.resize((new_w, target_h), resample=Image.BICUBIC)

    if new_w < max_w:
        canvas = Image.new("RGB", (max_w, target_h), (pad_value, pad_value, pad_value))
        canvas.paste(img_rs, (0, 0))
        return canvas

    return img_rs


class IAMLineDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_path: Path,
        processor,
        vocab: Dict[str, int],
        img_h: int,
        max_w: int,
        patch_size: int,
        upsample: int,
        drop_too_long: bool = True,
    ):
        self.images_dir = images_dir
        self.items_all = read_labels(labels_path)
        self.processor = processor
        self.vocab = vocab
        self.img_h = img_h
        self.max_w = max_w
        self.patch_size = patch_size
        self.upsample = max(1, upsample)

        self.max_T = (max_w // patch_size) * self.upsample

        if drop_too_long:
            kept = []
            for img_id, text in self.items_all:
                if len(text) <= self.max_T:
                    kept.append((img_id, text))
            self.items = kept
        else:
            self.items = self.items_all

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_id, text = self.items[idx]
        img_path = find_image_file(self.images_dir, img_id)
        img = Image.open(img_path)

        img = resize_keep_aspect_and_pad_right(img, target_h=self.img_h, max_w=self.max_w)

        pixel_values = self.processor.image_processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).pixel_values[0]

        target_ids = encode_text(text, self.vocab)
        return {
            "pixel_values": pixel_values,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_text": text,
            "img_id": img_id,
        }


@dataclass
class Batch:
    pixel_values: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    target_texts: List[str]


def collate_fn(samples: List[dict], pad_value: int = 0) -> Batch:
    pixel_values = torch.stack([s["pixel_values"] for s in samples], dim=0)

    lengths = torch.tensor([len(s["target_ids"]) for s in samples], dtype=torch.long)
    smax = int(lengths.max().item()) if len(samples) else 0
    targets = torch.full((len(samples), smax), fill_value=pad_value, dtype=torch.long)
    for i, s in enumerate(samples):
        t = s["target_ids"]
        if len(t) > 0:
            targets[i, : len(t)] = t

    target_texts = [s["target_text"] for s in samples]
    return Batch(pixel_values=pixel_values, targets=targets, target_lengths=lengths, target_texts=target_texts)


def resolve_split_dir(root: Path, name_candidates: List[str]) -> Path:
    for n in name_candidates:
        p = root / n
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of these under {root}: {name_candidates}")
