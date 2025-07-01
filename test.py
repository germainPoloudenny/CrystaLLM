import argparse
import os
import pickle
import numpy as np


def verify_split(path: str, condition_length: int) -> int:
    if not os.path.exists(path):
        return 0
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) % condition_length != 0:
        raise AssertionError(
            f"{os.path.basename(path)} length {len(data)} is not a multiple of {condition_length}"
        )
    return len(data) // condition_length


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that all conditioning sequences have the same length",
    )
    parser.add_argument(
        "dataset",
        help="Path to the conditioning dataset directory containing train.bin and meta.pkl",
    )
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    num_sequences = int(meta.get("num_sequences", 0))
    condition_length = int(meta.get("condition_length", 0))
    if num_sequences <= 0 or condition_length <= 0:
        raise ValueError("meta.pkl must contain num_sequences and condition_length > 0")

    total = 0
    for split in ["train", "val", "test"]:
        split_path = os.path.join(args.dataset, f"{split}.bin")
        count = verify_split(split_path, condition_length)
        if count:
            print(f"{split} sequences: {count}")
        total += count

    if total != num_sequences:
        raise AssertionError(
            f"meta.pkl reports {num_sequences} sequences but {total} found in .bin files"
        )
    print(f"All conditioning sequences verified with length {condition_length} tokens")


if __name__ == "__main__":
    main()