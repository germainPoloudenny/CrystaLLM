import numpy as np
from src.config import *
from project.crystallography.utils import *
from pymatgen.core import Structure
from scipy.spatial import cKDTree

def compute_electron_density(structure: Structure, grid_size=64):
    """
    Calcule la densité électronique d'une structure à partir d'une grille 3D.
    
    Paramètres :
    - structure : Structure pymatgen contenant les atomes et leur position.
    - grid_size : Tuple (nx, ny, nz) définissant la résolution de la grille.
    
    Retourne :
    - grid_density : Tableau 3D de densité électronique.
    """
    # Définition de la cellule unitaire
    lattice = structure.lattice
    frac_coords = np.array([site.frac_coords for site in structure.sites])
    atomic_numbers = np.array([site.specie.Z for site in structure.sites])  # Numéros atomiques

    # Génération de la grille 3D
    x_grid, y_grid, z_grid = np.meshgrid(
        np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), indexing='ij'
    )
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    # Construction d'un arbre spatial pour accélérer la recherche des atomes proches
    tree = cKDTree(frac_coords)

    # Calcul de la densité électronique
    grid_density = np.zeros((grid_size, grid_size, grid_size))

    # Paramètre de lissage (pour éviter une singularité en 1/r)
    sigma = 0.1  

    for i, point in enumerate(grid_points):
        # Trouver les atomes proches
        dists, indices = tree.query(point, k=len(frac_coords))
        dists[dists < 1e-6] = 1e-6  # Éviter la division par zéro
        
        # Contribution de chaque atome à la densité électronique
        density = np.sum(atomic_numbers[indices] * np.exp(-dists**2 / (2 * sigma**2)))

        # Stocker la densité à la position correspondante
        ix, iy, iz = np.unravel_index(i, (grid_size, grid_size, grid_size))
        grid_density[ix, iy, iz] = density

    
    return grid_density

def plot_electron_density(density, slice_index=0):
    """Affiche une coupe 2D de la densité électronique."""
    
    plt.figure(figsize=(6, 6))
    plt.imshow(density[:, :, slice_index], cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label="Densité électronique (e/Å³)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Densité électronique dans le plan z=0")
    plt.show()
import plotly.graph_objects as go
import numpy as np

def plot_electron_density_3d_plotly(density):
    x, y, z = np.where(density > np.max(density) * 0.1)  # Seulement les zones denses
    values = density[x, y, z]

    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=2, color=values, colorscale='Viridis', opacity=0.5)
    ))

    fig.update_layout(title="Densité électronique 3D", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    fig.show()

def compare_electron_densities(density1, density2):
    """Compare deux densités électroniques avec plusieurs métriques."""
    
    # Vérification que les dimensions sont identiques
    assert density1.shape == density2.shape, "Les densités doivent avoir la même forme"
    
    # Différence point par point
    diff = density1 - density2
    
    # Erreur quadratique moyenne (MSE)
    mse = np.mean(diff ** 2)
    
    # Erreur absolue moyenne (MAE)
    mae = np.mean(np.abs(diff))
    
    # Norme L2 (distance euclidienne entre les deux densités)
    l2_norm = np.sqrt(np.sum(diff ** 2))
    
    # Corrélation de Pearson
    correlation = np.corrcoef(density1.flatten(), density2.flatten())[0, 1]

    return {
        "MSE": mse,
        "MAE": mae,
        "L2 Norm": l2_norm,
        "Correlation": correlation
    }