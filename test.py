import pickle
import lmdb
import numpy as np
from pathlib import Path

def inspect_lmdb(lmdb_path: str):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, max_dbs=10000)
    
    with env.begin() as txn:
        num_dbs = int(txn.get(b"num_dbs"))
        length = int(txn.get(b"length"))
        num_embeddings = pickle.loads(txn.get(b"num_embeddings"))

        print(f"Nombre de bottleneck blocks (num_dbs) : {num_dbs}")
        print(f"Nombre total d'échantillons (length)  : {length}")
        print(f"Nombre d'éléments par dictionnaire (num_embeddings) : {num_embeddings}")
    
    # Ouvre chaque sous-database pour lire des exemples
    sub_dbs = [env.open_db(str(i).encode()) for i in range(num_dbs)]
    with env.begin(buffers=True) as txn:
        for db_idx, sub_db in enumerate(sub_dbs):
            print(f"\n== Bloc {db_idx} ==")
            with env.begin(db=sub_db) as sub_txn:
                cursor = sub_txn.cursor()
                for k, v in cursor:
                    encoding = pickle.loads(bytes(v))
                    print(f" - Sample ID: {k.decode()}, Encodage shape: {encoding.shape}")
                    break  # Lire un seul exemple par db

    env.close()


if __name__ == "__main__":
    lmdb_path = "data/version_0_last.lmdb"  # Remplace par le bon chemin
    inspect_lmdb(lmdb_path)