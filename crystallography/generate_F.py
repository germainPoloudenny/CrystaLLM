import h5py
from src.config import *
from src.dataset.structure import *
from src.crystallography.utils import *
import numpy as np
import time
from src.dataset.components.mp20_dataset import MP20 

amplitudes_file = f"{structure_type_path}/amplitudes/raw.h5"
idx_file = f"{structure_type_path}/idx.h5"

dataset = MP20(root="/home/gpoloudenny/Projects/all-atom-diffusion-transformer/data/mp_20")

execution_times = []
i = 0

nb_rotations = len(dataset)
with h5py.File(amplitudes_file, "w") as amp_file,  h5py.File(idx_file, "w") as idx_file:
    amp_dset = amp_file.create_dataset("amplitudes", (nb_rotations, len(hkl_list)), dtype=np.float32)
    #phase_dset = phase_file.create_dataset("phases", (nb_rotations, len(hkl_list)), dtype=np.float32)
    idx_dset = idx_file.create_dataset("idx", (nb_rotations,), dtype=np.int32)
    
    for idx, data in enumerate(dataset):
        #start_time = time.time()
        F = calculate_structure_factors(data)
        #end_time = time.time()
        #execution_times.append(end_time - start_time)
        #avg_time = np.mean(execution_times)
        #print(f"\nTemps moyen d'exécution par structure : {avg_time:.4f} secondes")
        amp_dset[idx] = np.abs(F)
        #print(amp_dset[i])
        #phase_dset[idx] = np.angle(F)
        idx_dset[idx] = idx
        
        print(idx)

print(f"Amplitude dataset saved to {amplitudes_file}")

