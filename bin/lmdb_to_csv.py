import argparse
import numpy as np
import pandas as pd
from crystallm import embeddings_from_lmdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LMDB embeddings to CSV.")
    parser.add_argument("lmdb", type=str, help="Path to the .lmdb database file.")
    parser.add_argument("--out", type=str, required=True, help="Path to the output .csv file.")
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type of raw bytes (optional if LMDB stores pickled objects).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Length of each vector when reading raw bytes.",
    )
    parser.add_argument(
        "--db-index",
        type=int,
        default=None,
        help="Read vectors from a specific LMDB sub-database (default: auto)",
    )
    args = parser.parse_args()

    dtype = None
    shape = None
    if args.dtype:
        dtype = np.dtype(args.dtype)
        shape = (args.length,) if args.length is not None else None

    embedding_data = embeddings_from_lmdb(
        args.lmdb, dtype=dtype, shape=shape, sub_db=args.db_index
    )
    df = pd.DataFrame.from_dict(embedding_data, orient="index")
    df.index.name = "element"
    df.to_csv(args.out)