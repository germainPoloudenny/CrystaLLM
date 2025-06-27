import argparse
import os
import pickle
import torch


def expand_checkpoint(ckpt_path: str, meta_path: str, out_path: str) -> None:
    """Expand embedding weights of a checkpoint to match a new vocabulary.

    Parameters
    ----------
    ckpt_path : str
        Path to the existing checkpoint file.
    meta_path : str
        Path to the meta.pkl describing the new vocabulary.
    out_path : str
        Where to write the expanded checkpoint.
    """
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    new_vocab_size = meta["vocab_size"]

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["model"]
    model_args = checkpoint.get("model_args", {})

    # In older or slightly different checkpoints the embedding weights may be
    # stored under a variety of keys. Try to locate the token embedding weight
    # tensor by checking a few common suffixes.
    possible_wte_keys = [
        "transformer.wte.weight",
        "tok_embeddings.weight",
        "word_embeddings.weight",
    ]
    old_weight_key = None
    for key in state.keys():
        for suffix in possible_wte_keys:
            if key.endswith(suffix):
                old_weight_key = key
                break
        if old_weight_key:
            break

    if old_weight_key is None:
        raise KeyError(
            "Token embedding weights not found in checkpoint state. Tried keys: "
            + ", ".join(possible_wte_keys)
        )

    old_weight = state[old_weight_key]
    old_vocab_size, emb_dim = old_weight.shape
    print(old_vocab_size)
    if new_vocab_size < old_vocab_size:
        raise ValueError("new vocab size must be >= old vocab size")
    if new_vocab_size == old_vocab_size:
        print("Checkpoint already matches requested vocabulary size")
        torch.save(checkpoint, out_path)
        return

    pad = torch.zeros(new_vocab_size - old_vocab_size, emb_dim, dtype=old_weight.dtype)
    # Try to find and expand lm_head.weight
    possible_lm_head_keys = [
        "lm_head.weight",
        "output.weight",
        "head.weight"
    ]

    lm_head_key = None
    for key in state.keys():
        for suffix in possible_lm_head_keys:
            if key.endswith(suffix):
                lm_head_key = key
                break
        if lm_head_key:
            break

    if lm_head_key:
        lm_weight = state[lm_head_key]
        if lm_weight.shape[0] == old_vocab_size:
            state[lm_head_key] = torch.cat([lm_weight, pad.clone()], dim=0)
        else:
            print(f"Warning: Expected lm_head vocab size {old_vocab_size}, but got {lm_weight.shape[0]}. Skipping lm_head.weight expansion.")
    else:
        print("Warning: lm_head.weight not found. Skipping its expansion.")
    # Try to find and expand lm_head.weight
    possible_lm_head_keys = [
        "lm_head.weight",
        "output.weight",
        "head.weight"
    ]

    lm_head_key = None
    for key in state:
        for suffix in possible_lm_head_keys:
            if key.endswith(suffix):
                lm_head_key = key
                break
        if lm_head_key:
            break

    if lm_head_key:
        lm_weight = state[lm_head_key]
        if lm_weight.shape[0] == old_vocab_size:
            state[lm_head_key] = torch.cat([lm_weight, pad.clone()], dim=0)
            print(f"Expanded {lm_head_key} to match new vocab size.")
        else:
            print(f"Warning: {lm_head_key} has unexpected vocab size {lm_weight.shape[0]}. Skipping expansion.")
    else:
        print("Warning: No lm_head.weight found. Skipping its expansion.")

    if "cond_embedding.weight" in state:
        cond_weight = state["cond_embedding.weight"]
        cond_pad = torch.zeros(new_vocab_size - old_vocab_size, cond_weight.shape[1], dtype=cond_weight.dtype)
        state["cond_embedding.weight"] = torch.cat([cond_weight, cond_pad], dim=0)

    model_args["vocab_size"] = new_vocab_size
    checkpoint["model"] = state
    checkpoint["model_args"] = model_args

    torch.save(checkpoint, out_path)
    print(f"Saved expanded checkpoint with vocab_size={new_vocab_size} to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Expand checkpoint to a larger vocabulary")
    parser.add_argument("ckpt", help="Path to ckpt.pt")
    parser.add_argument("meta", help="meta.pkl with new vocabulary")
    parser.add_argument("out", help="Output checkpoint path")
    args = parser.parse_args()
    expand_checkpoint(args.ckpt, args.meta, args.out)


if __name__ == "__main__":
    main()