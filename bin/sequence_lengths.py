import os
import argparse
import pickle
import numpy as np


def load_start_indices(dataset_dir: str, data_token_id: int) -> np.ndarray:
    """Return CIF start indices for the given dataset."""
    start_path = os.path.join(dataset_dir, "starts.pkl")
    if os.path.exists(start_path):
        with open(start_path, "rb") as f:
            starts = np.array(pickle.load(f), dtype=np.int64)
        return starts

    train_path = os.path.join(dataset_dir, "train.bin")
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    starts = np.nonzero(data == data_token_id)[0]
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print token sequence lengths for a dataset",
    )
    parser.add_argument(
        "dataset",
        help="Path to directory containing train.bin and meta.pkl",
    )
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta.get("stoi") or {}
    has_data_tok = "data_" in stoi
    data_token_id = stoi.get("data_")

    train_path = os.path.join(args.dataset, "train.bin")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")

    lengths = None
    if has_data_tok:
        starts = load_start_indices(args.dataset, data_token_id)
        starts = np.append(starts, len(train_data))
        lengths = np.diff(starts)
    elif os.path.exists(os.path.join(args.dataset, "starts.pkl")):
        starts = load_start_indices(args.dataset, 0)
        starts = np.append(starts, len(train_data))
        lengths = np.diff(starts)

    if lengths is not None and len(lengths) > 0:
        print(f"number of sequences: {len(lengths):,}")
        print(f"min length: {np.min(lengths):,}")
        print(f"max length: {np.max(lengths):,}")
        print(f"mean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
        return

    num_sequences = meta.get("num_sequences")
    if num_sequences:
        mean_len = len(train_data) / num_sequences
        max_len = meta.get("condition_length", "?")
        print(f"number of sequences: {num_sequences:,}")
        print("min length: n/a")
        print(f"max length: {max_len}")
        print(f"mean length: {mean_len:.2f}")
    else:
        print("number of sequences: 0")


if __name__ == "__main__":
    main()