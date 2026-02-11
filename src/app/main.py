"""Application entry point used by train.py."""

import torch
from transformers import TrOCRProcessor

from .config import default_config_path, load_config
from .inference import run_infer
from .training import run_train
from .utils import set_seed


def main():
    args = load_config(default_config_path())
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = (not args.no_fp16) and (device.type == "cuda")

    if args.mode == "train":
        processor = TrOCRProcessor.from_pretrained(args.model_name)
        run_train(args, device, fp16, processor)
        return

    if args.mode == "infer":
        run_infer(args, device)
        return

    raise ValueError(f"Unsupported mode: {args.mode}. Use 'train' or 'infer'.")


if __name__ == "__main__":
    main()
