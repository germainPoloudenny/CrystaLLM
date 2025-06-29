import os
import argparse
import pickle
import numpy as np


def count_cifs(dataset_dir: str) -> int:
    """Return the number of CIF structures encoded in train.bin."""
    meta_path = os.path.join(dataset_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta.get("stoi")
    if stoi is None or "data_" not in stoi:
        raise KeyError("token 'data_' not found in meta.pkl")
    data_token_id = stoi["data_"]

    train_path = os.path.join(dataset_dir, "train.bin")
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    return int(np.count_nonzero(data == data_token_id))


def read_num_cond_sequences(cond_dir: str) -> int:
    """Read the number of conditioning sequences from meta.pkl."""
    meta_path = os.path.join(cond_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if "num_sequences" not in meta:
        raise KeyError("'num_sequences' not found in conditioning meta.pkl")
    return int(meta["num_sequences"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that the main dataset and the conditioning dataset have the same number of sequences.")
    parser.add_argument("dataset", help="Path to the directory with train.bin and meta.pkl for the CIF dataset")
    parser.add_argument("condition_dataset", help="Path to the conditioning dataset directory")
    args = parser.parse_args()

    num_cifs = count_cifs(args.dataset)
    num_cond_seqs = read_num_cond_sequences(args.condition_dataset)

    print(f"CIF structures: {num_cifs}")
    print(f"Condition sequences: {num_cond_seqs}")

    if num_cifs != num_cond_seqs:
        raise AssertionError(f"taille différente : {num_cifs} vs {num_cond_seqs}")
    print("Les deux jeux ont la même taille.")


if __name__ == "__main__":
    main()