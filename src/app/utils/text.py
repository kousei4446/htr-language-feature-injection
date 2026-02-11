import random
from typing import Dict, List

import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer_from_pairs(preds: List[str], gts: List[str]) -> float:
    total_edits = 0
    total_chars = 0
    for p, g in zip(preds, gts):
        total_edits += levenshtein(p, g)
        total_chars += max(1, len(g))
    return total_edits / total_chars


def build_id2char(vocab: Dict[str, int]) -> List[str]:
    id2char = [""] * (max(vocab.values()) + 1)
    for ch, i in vocab.items():
        id2char[i] = ch
    return id2char


def ctc_greedy_decode(logits: torch.Tensor, blank_id: int, id2char: List[str]) -> List[str]:
    pred_ids = logits.argmax(dim=-1)
    out = []
    for seq in pred_ids.tolist():
        s = []
        prev = None
        for t in seq:
            if t == prev:
                continue
            prev = t
            if t == blank_id:
                continue
            ch = id2char[t] if 0 <= t < len(id2char) else ""
            if ch in ("<blank>",):
                continue
            if ch == "<unk>":
                ch = "?"
            if ch.startswith("<") and ch.endswith(">"):
                continue
            s.append(ch)
        out.append("".join(s))
    return out


def encode_texts_for_llm(texts, tokenizer, max_len: int):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    assert bos is not None and eos is not None, "LLM tokenizer must have BOS/EOS"

    ids_list = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        ids = [bos] + ids + [eos]
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        ids_list.append(ids)

    B = len(ids_list)
    L = max(len(x) for x in ids_list) if B else 0
    input_ids = torch.full((B, L), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)

    for i, ids in enumerate(ids_list):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    return input_ids, attn
