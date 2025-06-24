import numpy as np
import pickle

def check_dataset(dirname):
    with open(f"{dirname}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]

    train = np.fromfile(f"{dirname}/train.bin", dtype=np.uint16)
    val   = np.fromfile(f"{dirname}/val.bin",   dtype=np.uint16)

    print(
        f"{dirname}: vocab_size={vocab_size}, "
        f"train max={train.max()}, val max={val.max()}"
    )

check_dataset("data/tokens_mp20_all")
check_dataset("data/tokens_mp20_amp")