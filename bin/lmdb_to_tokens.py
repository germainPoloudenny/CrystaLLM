import os
import argparse
import numpy as np

from crystallm import sequences_from_lmdb


def main(args):
    """Convert LMDB sequences to train.bin/val.bin.

    The script shuffles the sequences, splits them according to
    ``--train-fraction`` and writes them as raw ``np.uint16`` arrays.  All
    sequences must have the same length so that the resulting binaries can be
    used as a ``condition_dataset`` when training the language model.
    """

    sequences = sequences_from_lmdb(args.lmdb_path, sub_db=args.sub_db)

    if not sequences:
        raise ValueError("LMDB contains no sequences")

    seq_len = len(sequences[0])
    if any(len(s) != seq_len for s in sequences):
        raise ValueError("Sequences do not all have the same length")

    print(f"Detected sequence length = {seq_len}")
    print("Use this value as 'condition_length' when training.")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(sequences)

    split_idx = int(len(sequences) * args.train_fraction)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    os.makedirs(args.output_dir, exist_ok=True)

    meta = {
        "condition_length": seq_len,
        "num_sequences": len(sequences),
    }

    if train_seqs:
        train_arr = np.concatenate([seq.astype(np.uint16).ravel() for seq in train_seqs])
        train_arr.tofile(os.path.join(args.output_dir, "train.bin"))
        print(
            f"Wrote {len(train_seqs)} sequences to {os.path.join(args.output_dir, 'train.bin')}"
        )

    if val_seqs:
        val_arr = np.concatenate([seq.astype(np.uint16).ravel() for seq in val_seqs])
        val_arr.tofile(os.path.join(args.output_dir, "val.bin"))
        print(
            f"Wrote {len(val_seqs)} sequences to {os.path.join(args.output_dir, 'val.bin')}"
        )

    # Store some metadata for easier use by bin/train.py
    import pickle

    with open(os.path.join(args.output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VQVAE LMDB to token dataset for conditioning.")
    parser.add_argument("lmdb_path", help="Path to the .lmdb directory")
    parser.add_argument("output_dir", help="Output directory for train.bin/val.bin")
    parser.add_argument("--train-fraction", type=float, default=0.9, dest="train_fraction", help="Fraction of sequences for training")
    parser.add_argument("--sub-db", type=int, default=0, help="Sub-database index to read from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling before split")
    args = parser.parse_args()
    main(args)