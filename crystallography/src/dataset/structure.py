import h5py
from pymatgen.core import Structure
from src.config import *

class StructureDataset:
    def __init__(self):
        h5_path =  f"{structure_type_path}/cif.h5"
        self.h5_file = h5py.File(h5_path, "r")
        self.dataset = self.h5_file["cif"]
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index hors limites.")
        
        cif_content = self.dataset[idx].decode("utf-8")  # s'assurer que c'est bien une chaîne
        try:
            structure = Structure.from_str(cif_content, fmt="cif")
            return structure
        except Exception as e:
            print(f"⚠️ Erreur de parsing CIF à l'index {idx} : {e}")
            return None

    def close(self):
        self.h5_file.close()

    def __del__(self):
        self.close()
