from __future__ import annotations
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg"}

def convert_one(src: Path, in_dir: Path, out_dir: Path, quality: int, method: int, gray: bool) -> tuple[Path, int]:
    rel = src.relative_to(in_dir)
    dst = (out_dir / rel).with_suffix(".webp")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        # HTR用途なら gray=True が効きやすい（必要なければOFFで）
        if gray:
            im = im.convert("L")
        else:
            # 変換安定化（RGBAなどでも落ちないように）
            if im.mode not in ("RGB", "RGBA", "L"):
                im = im.convert("RGB")

        # WebP保存（メタデータは基本落ちます）
        im.save(dst, format="WEBP", quality=quality, method=method, optimize=True)

    return dst, dst.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="IAM_Aachen のパス（train/val/test がある場所）")
    ap.add_argument("--out", type=Path, default=Path("IAM_Aachen_webp"), help="出力先")
    ap.add_argument("--quality", type=int, default=45, help="WebP quality (0-100). 小さいほど高圧縮")
    ap.add_argument("--method", type=int, default=6, help="WebP method (0-6). 大きいほど高圧縮だが遅い")
    ap.add_argument("--workers", type=int, default=0, help="並列数（0ならCPU数-1）")
    ap.add_argument("--gray", action="store_true", help="グレースケール化（サイズがかなり落ちることが多い）")
    args = ap.parse_args()

    root = args.root.resolve()
    out_root = args.out.resolve()

    sets = ["train", "val", "test"]
    tasks = []

    for s in sets:
        in_dir = root / s / "images"
        if not in_dir.exists():
            continue
        for p in in_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                tasks.append((p, in_dir, out_root / s / "images"))

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 1)

    total_bytes = 0
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(convert_one, src, in_dir, out_dir, args.quality, args.method, args.gray)
            for (src, in_dir, out_dir) in tasks
        ]
        for f in as_completed(futs):
            dst, size = f.result()
            total_bytes += size
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(tasks)} done...")

    print(f"Done: {len(tasks)} files")
    print(f"Output: {out_root}")
    print(f"Total: {total_bytes/1024/1024:.2f} MB")

if __name__ == "__main__":
    import os
    main()
