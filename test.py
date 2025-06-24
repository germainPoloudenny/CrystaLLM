import lmdb
import pickle

env = lmdb.open("data/version_0_last.lmdb", readonly=True, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    for i, (key, value) in enumerate(cursor):
        print(key.decode("utf-8"))
        if i > 50:
            break