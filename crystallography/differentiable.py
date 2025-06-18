import jax
import jax.numpy as jnp
import xraydb

def get_f0_table(symbol, q_min=0.01, q_max=50.0, num_points=1000):
    q_vals = np.linspace(q_min, q_max, num_points)
    f_vals = np.array([xraydb.f0(symbol, q) for q in q_vals])
    return q_vals, f_vals

def generate_f0_tables_for_structure(structure):
    """
    Génère les tables f₀(q) pour tous les éléments présents dans une structure donnée.
    
    Returns:
        dict: { "Si": {"q": array([...]), "f": array([...])}, ... }
    """
    unique_elements = set(site.specie.symbol for site in structure.sites)
    q_table_dict = {}

    for symbol in unique_elements:
        q_vals, f_vals = get_f0_table(symbol)
        q_table_dict[symbol] = {
            "q": q_vals,
            "f": f_vals
        }
    
    return q_table_dict

def interpolate_f0_jax(q_input, q_table, f_table):
    q_input = jnp.atleast_1d(q_input)
    
    def interp_scalar(q):
        q = jnp.clip(q, q_table[0], q_table[-1])  # ⬅️ CLAMP important
        idx = jnp.clip(jnp.searchsorted(q_table, q) - 1, 0, len(q_table) - 2)
        q0, q1 = q_table[idx], q_table[idx + 1]
        f0, f1 = f_table[idx], f_table[idx + 1]
        slope = (f1 - f0) / (q1 - q0)
        return f0 + slope * (q - q0)

    result = jax.vmap(interp_scalar)(q_input)
    return result if result.shape[0] > 1 else result[0]

def calculate_structure_factors_jax_pure(frac_coords, symbols, recip_matrix, q_table_dict):
    cartesian_hkl = jnp.dot(jax_hkl_list, recip_matrix)

    def compute_F(hkl_cart, hkl_index):
        q_mag = jnp.linalg.norm(hkl_cart)
        F = 0.0 + 0.0j
        for i in range(len(frac_coords)):
            f = interpolate_f0_jax(q_mag, q_table_dict[symbols[i]]["q"], q_table_dict[symbols[i]]["f"])
            phase = 2 * jnp.pi * jnp.dot(hkl_index, frac_coords[i])
            F += f * jnp.exp(-1j * phase)
        return F

    F_hkls = jax.vmap(compute_F, in_axes=(0, 0))(cartesian_hkl, jax_hkl_list)
    A = jnp.abs(F_hkls)
    A = jnp.squeeze(A)
    return A


import h5py
from src.config import *
import numpy as np
import time
from src.config import hkl_list
from src.dataset.components.mp20_dataset import MP20 

structure_dataset = MP20(root="/home/gpoloudenny/Projects/all-atom-diffusion-transformer/data/mp_20")
structure = structure_dataset[0]

q_table_dict = generate_f0_tables_for_structure(structure)
q_table_dict = {
    k: {
        "q": jnp.array(v["q"]),
        "f": jnp.array(v["f"])
    } for k, v in q_table_dict.items()
}


symbols = [site.specie.symbol for site in structure.sites]
frac_coords = jnp.array([site.frac_coords for site in structure.sites])
recip_matrix = jnp.array(structure.lattice.reciprocal_lattice_crystallographic.matrix)
jax_hkl_list = jnp.array(hkl_list)

A = calculate_structure_factors_jax_pure(frac_coords, symbols, recip_matrix, q_table_dict)

from src.crystallography.utils import *

real_A = np.abs(calculate_structure_factors(structure))

print(A, real_A)

from jax import grad

def test_fn(frac_coords):
     # symbols, recip_matrix, hkl_list et q_table_dict doivent être figés
    model_amplitudes = calculate_structure_factors_jax_pure(frac_coords, symbols, recip_matrix, q_table_dict)
    return jnp.mean((model_amplitudes - real_A) ** 2)

dL_dcoords = grad(test_fn)(frac_coords)
print(dL_dcoords)
