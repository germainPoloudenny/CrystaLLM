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

    old_weight = state["transformer.wte.weight"]
    old_vocab_size, emb_dim = old_weight.shape
    if new_vocab_size < old_vocab_size:
        raise ValueError("new vocab size must be >= old vocab size")
    if new_vocab_size == old_vocab_size:
        print("Checkpoint already matches requested vocabulary size")
        torch.save(checkpoint, out_path)
        return

    pad = torch.zeros(new_vocab_size - old_vocab_size, emb_dim, dtype=old_weight.dtype)
    state["transformer.wte.weight"] = torch.cat([old_weight, pad], dim=0)
    state["lm_head.weight"] = torch.cat([state["lm_head.weight"], pad.clone()], dim=0)

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