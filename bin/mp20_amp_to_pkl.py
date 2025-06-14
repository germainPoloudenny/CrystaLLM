import os
import warnings
import torch
from torch_geometric.data import Data, InMemoryDataset
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import argparse
import gzip
import pickle
import lmdb
import h5py
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure
from tqdm import tqdm  # ✅ Barre de progression

class MP20(InMemoryDataset):
    def __init__(self, root: str, cache: bool = True):
        self.root = root
        self.raw_csv = os.path.join(root, "raw/all.csv")
        self.cache_path = os.path.join(root, "raw/all.pt")

        print(f"Chargement du dataset MP20 depuis {self.cache_path}...")
        if cache and os.path.exists(self.cache_path):
            self._data = torch.load(self.cache_path, weights_only=False)
            print(f"→ {len(self._data)} structures chargées.")
        else:
            raise FileNotFoundError(f"Fichier non trouvé: {self.cache_path}")

    def __getitem__(self, idx: int) -> Structure:
        return Structure.from_str(self._data[idx]["cif"], fmt="cif")

    def __len__(self) -> int:
        return len(self._data)

def load_amp_sequences(lmdb_path):
    print(f"Ouverture de la base LMDB depuis {lmdb_path}...")
    env = lmdb.open(lmdb_path, readonly=True, lock=False, max_dbs=100000)

    with env.begin() as txn:
        length = int(txn.get(b"length").decode())
        num_dbs = int(txn.get(b"num_dbs").decode())
        print(f"→ {length} entrées, {num_dbs} sous-bases à lire.")

    sub_dbs = [env.open_db(str(i).encode(), create=False) for i in range(num_dbs)]
    seqs = []

    print("Chargement des séquences AMP...")
    with env.begin() as txn:
        for idx in tqdm(range(length)):
            tokens = []
            for sub_db in sub_dbs:
                raw = txn.get(str(idx).encode(), db=sub_db)
                codes = pickle.loads(raw)
                tokens.extend([f"<AMP{c}>" for c in codes])
            seqs.append(tokens)

    env.close()
    print("→ Séquences AMP chargées.")
    return seqs

def load_structure_indices(dataset_path):
    idx_path = os.path.join(dataset_path, "rotations", "idx.h5")
    print(f"Chargement des indices de structure depuis {idx_path}...")
    with h5py.File(idx_path, "r") as f:
        struct_indices = f["idx"][:]
    print(f"→ {len(struct_indices)} indices chargés.")
    return struct_indices

def main(args):
    print("Initialisation des données...")
    ds = MP20(args.mp20_root)
    amp_seqs = load_amp_sequences(args.lmdb_path)
    struct_indices = load_structure_indices(args.amp_root)

    assert len(struct_indices) == len(amp_seqs), "Length mismatch between amplitudes and structure indices"
    print("Fusion des données (AMP + CIF)...")

    combined = []
    for i, (amps, idx) in enumerate(tqdm(zip(amp_seqs, struct_indices), total=len(amp_seqs))):
        cif = ds[idx].to(fmt="cif")
        text = " ".join(amps) + "\n" + cif
        combined.append((f"id_{i}", text))

    print(f"→ {len(combined)} exemples combinés.")

    print("Découpage en ensemble d'entraînement et de validation...")
    train_data, val_data = train_test_split(
        combined,
        test_size=args.val_fraction,
        shuffle=True,
        random_state=args.random_state,
    )

    print("Sauvegarde des fichiers...")
    with gzip.open(args.train_out, "wb") as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(args.val_out, "wb") as f:
        pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ {len(train_data)} exemples écrits dans {args.train_out}")
    print(f"✅ {len(val_data)} exemples écrits dans {args.val_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build amplitude+CIF dataset")
    parser.add_argument("--mp20_root", type=str, default="/home/gpoloudenny/Projects/all-atom-diffusion-transformer/data/mp_20")
    parser.add_argument("--lmdb_path", type=str, default="/home/gpoloudenny/Projects/3D-VQ-VAE-2/version_0_last.lmdb")
    parser.add_argument("--amp_root", type=str, default="/home/gpoloudenny/Projects/crystallography/data/mp20")
    parser.add_argument("--train_out", type=str, default="mp20_amp_train.pkl.gz")
    parser.add_argument("--val_out", type=str, default="mp20_amp_val.pkl.gz")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=20230610)
    args = parser.parse_args()
    main(args)
