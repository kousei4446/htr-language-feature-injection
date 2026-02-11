from typing import List, Optional

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel


class LAILConnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(self.norm(x))


def parse_layers(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, int):
        return [int(value)]
    s = str(value).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class TrOCREncoderCTC(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        upsample: int = 1,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        lail_layers: Optional[List[int]] = None,
        llm_hidden_size: Optional[int] = None,
        lail_use_upsample: bool = False,
    ):
        super().__init__()
        base = VisionEncoderDecoderModel.from_pretrained(model_name)
        base.config.eos_token_id = 1
        base.config.pad_token_id = 2
        base.config.decoder_start_token_id = 2

        self.encoder = base.encoder
        self.hidden = self.encoder.config.hidden_size
        self.patch_size = getattr(self.encoder.config, "patch_size", 16)

        self.upsample = max(1, upsample)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.hidden, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.lail_layers = sorted(set(lail_layers or []))
        self.lail_use_upsample = bool(lail_use_upsample)
        self.lail_connectors = None
        self.llm_hidden_size = llm_hidden_size

        if self.lail_layers:
            assert llm_hidden_size is not None, "llm_hidden_size is required when lail_layers is not empty"
            self.lail_connectors = nn.ModuleDict(
                {str(l): LAILConnector(self.hidden, llm_hidden_size) for l in self.lail_layers}
            )

    def _tokens_to_1d(self, hs: torch.Tensor, pixel_values: torch.Tensor, apply_upsample: bool) -> torch.Tensor:
        B, N, Hh = hs.shape
        H_img = pixel_values.shape[-2]
        W_img = pixel_values.shape[-1]
        gh = H_img // self.patch_size
        gw = W_img // self.patch_size

        expected_patch = gh * gw
        num_special = N - expected_patch
        if num_special in (1, 2):
            hs = hs[:, num_special:, :]
        elif num_special == 0:
            pass
        else:
            raise RuntimeError(
                f"Unexpected token length N={N}, expected {expected_patch} (+1/+2 special). "
                f"(img={H_img}x{W_img}, patch={self.patch_size})"
            )

        hs2d = hs.view(B, gh, gw, Hh)
        hs1d = hs2d.mean(dim=1)

        if apply_upsample and self.upsample > 1:
            hs1d = hs1d.repeat_interleave(self.upsample, dim=1)
        return hs1d

    def forward(self, pixel_values: torch.Tensor, return_lail: bool = False):
        need_hs = return_lail and (self.lail_connectors is not None)

        try:
            enc = self.encoder(
                pixel_values=pixel_values,
                interpolate_pos_encoding=True,
                output_hidden_states=need_hs,
                return_dict=True,
            )
        except TypeError:
            enc = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=need_hs,
                return_dict=True,
            )

        hs_last = enc.last_hidden_state
        seq_last = self._tokens_to_1d(hs_last, pixel_values, apply_upsample=True)
        seq_last = self.drop(seq_last)
        logits = self.classifier(seq_last)

        if not need_hs:
            return logits

        prefixes = {}
        hidden_states = enc.hidden_states
        for l in self.lail_layers:
            idx = l
            hs_l = hidden_states[idx]
            seq_l = self._tokens_to_1d(hs_l, pixel_values, apply_upsample=self.lail_use_upsample)
            prefixes[l] = self.lail_connectors[str(l)](seq_l)

        return logits, prefixes
