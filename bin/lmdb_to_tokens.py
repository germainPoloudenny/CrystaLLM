import argparse
import os
import numpy as np
from crystallm import sequences_from_lmdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LMDB sequences to a prefix-token dataset.")
    parser.add_argument("train_lmdb", type=str, help="Path to the training LMDB file.")
    parser.add_argument("--val_lmdb", type=str, default=None, help="Optional validation LMDB file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .bin files.")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Data type of raw bytes if the LMDB stores bytes instead of pickled arrays.")
    parser.add_argument("--length", type=int, default=None,
                        help="Length of each sequence when reading raw bytes.")
    parser.add_argument("--db_index", type=int, default=None,
                        help="Index of the LMDB sub-database to read from.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = np.dtype(args.dtype) if args.dtype else None

    train_seqs = sequences_from_lmdb(
        args.train_lmdb, dtype=dtype, length=args.length, sub_db=args.db_index
    )
    train_tokens = np.concatenate(train_seqs).astype(np.uint16)
    train_tokens.tofile(os.path.join(args.out_dir, "train.bin"))

    if args.val_lmdb:
        val_seqs = sequences_from_lmdb(
            args.val_lmdb, dtype=dtype, length=args.length, sub_db=args.db_index
        )
        val_tokens = np.concatenate(val_seqs).astype(np.uint16)
        val_tokens.tofile(os.path.join(args.out_dir, "val.bin"))