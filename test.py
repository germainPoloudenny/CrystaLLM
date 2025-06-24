import numpy as np
for path in ["data/tokens_mp20_train_val/train.bin",
             "data/tokens_mp20_train_val/val.bin",
             "data/tokens_mp20_amp/train.bin",
             "data/tokens_mp20_amp/val.bin"]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(path, "max token =", data.max())