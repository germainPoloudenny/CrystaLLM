import os
import pickle

cif_dir = "data/tokens_mp20_train_val"          # contient train.bin, starts.pkl, etc.
cond_dir = "data/tokens_mp20_amp"  # contient train.bin, meta.pkl, etc.

# nombre de structures dans le jeu CIF
with open(os.path.join(cif_dir, "starts.pkl"), "rb") as f:
    cif_starts = pickle.load(f)
num_cif_structures = len(cif_starts)

# nombre de séquences dans le jeu de conditionnement
with open(os.path.join(cond_dir, "meta.pkl"), "rb") as f:
    cond_meta = pickle.load(f)
num_cond_sequences = cond_meta["num_sequences"]

assert num_cif_structures == num_cond_sequences, \
    f"taille différente : {num_cif_structures} vs {num_cond_sequences}"
print("Les deux jeux ont la même taille.")