import numpy as np
import xraydb
from multiprocessing import Pool
from functools import lru_cache
from src.config import hkl_list

# Récupération de la base de données des rayons X
xdb = xraydb.get_xraydb() 

# Mise en cache des facteurs de diffusion pour éviter les recalculs inutiles
@lru_cache(maxsize=None) 
def precompute_factors(element, q_magnitude):
    """Pré-calculer le facteur de diffusion pour un élément donné et une norme q."""
    return float(xdb.f0(element, q_magnitude))

# Fonction qui sera exécutée en parallèle pour calculer l'amplitude structurelle
def calculate_structure_factors_worker(args):
    """Travailleur parallèle pour calculer une amplitude structurelle."""
    # Décompacte les arguments : indices de Miller, coordonnées cartésiennes et données des atomes 
    hkl, cartesian_hkl, site_data = args 
    
    # Calcul de la norme du vecteur réciproque q
    q_magnitude = np.linalg.norm(cartesian_hkl) 
    
    # Initialisation du facteur de structure complexe
    F_hkl = 0.0 + 0.0j 

    # Boucle sur tous les atomes de la structure pour calculer leur contribution
    for atom_type, frac_coords in site_data: 
        # Récupération du facteur de diffusion de l'atome
        f_atom = precompute_factors(atom_type, q_magnitude)  
        # Calcul de la phase en fonction des coordonnées fractionnaires
        phase = 2 * np.pi * np.dot(hkl, frac_coords)
        # Contribution de chaque atome au facteur de structure
        F_hkl += f_atom * np.exp(-1j * phase)

    # Retourne l'amplitude du facteur de structure
    return F_hkl

# Fonction principale pour calculer les amplitudes structurelles pour plusieurs hkl
def calculate_structure_factors(structure):
    """Calculer les amplitudes structurelles pour une liste de hkl."""
    
    # Récupération du réseau cristallin
    lattice = structure.lattice
    
    # Obtention du réseau réciproque
    reciprocal_lattice = lattice.reciprocal_lattice_crystallographic  
    # Conversion des indices de Miller (hkl) en coordonnées cartésiennes dans l'espace réciproque
    cartesian_hkl = np.array([reciprocal_lattice.get_cartesian_coords(hkl) for hkl in hkl_list])
    
    # Extraction des informations sur les atomes : type et coordonnées fractionnaires
    site_data = [(site.specie.symbol, site.frac_coords) for site in structure.sites]

    # Préparation des arguments pour les travailleurs parallèles
    worker_args = [(hkl, cartesian_hkl[i], site_data) for i, hkl in enumerate(hkl_list)]
    
    # Exécution en parallèle des calculs avec 16 processus
    with Pool(processes=16) as pool:
        structure_factors = pool.map(calculate_structure_factors_worker, worker_args)
        
    #plot_amplitudes_3D(hkl_list, amplitudes)

    # Retourne les amplitudes calculées
    return structure_factors