import torch


def clm_loss_with_prefix(llm, input_ids, attention_mask, prefix_embeds):
    device = prefix_embeds.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    tok_emb = llm.get_input_embeddings()(input_ids)
    tok_emb = tok_emb.to(dtype=prefix_embeds.dtype)

    inputs_embeds = torch.cat([prefix_embeds, tok_emb], dim=1)

    B, P, _ = prefix_embeds.shape
    prefix_mask = torch.ones((B, P), device=device, dtype=attention_mask.dtype)
    attn = torch.cat([prefix_mask, attention_mask], dim=1)

    labels = torch.full((B, P + input_ids.size(1)), -100, device=device, dtype=torch.long)
    labels[:, P:] = input_ids
    labels[:, P:][attention_mask == 0] = -100
    labels[:, P] = -100

    out = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        labels=labels,
        use_cache=False,
    )
    return out.loss
