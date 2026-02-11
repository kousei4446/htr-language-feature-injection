from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    defaults = {
        "mode": "train",
        "data_root": None,
        "model_name": "microsoft/trocr-small-handwritten",
        "out_dir": "out_ctc_best",
        "epochs": 20,
        "batch_size": 6,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "freeze_encoder": False,
        "dropout": 0.0,
        "img_h": 384,
        "max_w": 2048,
        "upsample": 1,
        "num_workers": 0,
        "no_fp16": False,
        "seed": 42,
        "infer_image": None,
        "ckpt": None,
        "use_lail": False,
        "llm_name": "meta-llama/Meta-Llama-3-8B",
        "lail_layers": "12",
        "lail_alpha": 0.1,
        "llm_max_len": 128,
        "llm_grad_ckpt": False,
        "lail_use_upsample": False,
        "lail_ds_blocks": 1,
        "run_name": "lail_liner_2layer",
        "save_every": 0,
    }
    base = OmegaConf.create(defaults)
    user_cfg = OmegaConf.load(path)
    return OmegaConf.merge(base, user_cfg)


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config.yaml"
