import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from periodictable import cromermann
import xraydb

symbol = 'Si'
q_vals = np.linspace(0, 10, 100)

# Cromer-Mann analytique
def f0_cromer(symbol, q):
    formula = cromermann.getCMformula(symbol)
    a, b, c = formula.a, formula.b, formula.c
    s = q / (4 * np.pi)
    return sum(ai * np.exp(-bi * s**2) for ai, bi in zip(a, b)) + c

f0_jax = np.array([f0_cromer(symbol, q) for q in q_vals])
f0_ref = np.array([xraydb.f0(symbol, q) for q in q_vals])

plt.plot(q_vals, f0_ref, label="xraydb.f0 (tabulé)")
plt.xlabel("q (1/Å)")
plt.ylabel("f₀(q)")
plt.legend()
plt.grid(True)
plt.title("Comparaison des modèles f₀(q)")
plt.savefig("test")
exit()
