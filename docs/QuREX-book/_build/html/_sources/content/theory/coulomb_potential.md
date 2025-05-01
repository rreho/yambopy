# Model Coulomb potential

**Goal**: Compute the Coulomb potential in the Wannier basis with a cutoff

We already have the TB hamiltonian $H_{mn}(R)$ in real space and the Fourier Transform $H_{mn}(k)$ `wannier_model.py`

We have diagonalized Bloch states:

$$
H_{n m}(k) u_m^n(k)=E_n(k) u_n(k)
$$(eq:secular-eqHTB)

Coulomb interaction in real-space

$$
V(\mathbf{r})=\frac{e^2}{\epsilon|\mathbf{r}|}
$$

The Coulomb interaction in momentum space is expressed as:

$$
V(G)=\frac{4 \pi e^2}{\epsilon G^2}
$$

For 2D materials a valid option is 

$$
V(G)=\frac{2 \pi e^2}{\epsilon G} \tanh (G d)
$$
with d the effective thickness of the material.

You also have to introduce an energy cutoff
$$
|G| \leq G_{\mathrm{cut}}=\sqrt{\frac{2 m E_{\mathrm{cut}}}{\hbar^2}}
$$

However, this is in the plane-wave basis and we need it in the Wannier basis.

The Coulomb potential in real space can be defined as:

TB Coulomb matrix elements:

$$
V_{n m}=\sum_G V(G) S_{n m}(G)
$$
S is the structure factor

$$
S_{n m}(G)=\sum_R e^{i G \cdot R} \phi_n^*(R) \phi_m(R)
$$

# Code examples

## Coulomb potential reciprocal space

```{python}
import numpy as np

# Constants
e2 = 1.44  # eV·nm (Coulomb constant in natural units)
epsilon = 2.5  # Dielectric constant
E_cut = 5.0  # Energy cutoff in eV
m_eff = 1.0  # Effective mass (relative to electron mass)
hbar = 0.658  # eV·fs

# Define reciprocal lattice vectors (G-grid)
G_max = np.sqrt(2 * m_eff * E_cut / hbar**2)  # Cutoff in G-space
Gx, Gy = np.meshgrid(np.linspace(-G_max, G_max, 100), 
                     np.linspace(-G_max, G_max, 100))
G = np.sqrt(Gx**2 + Gy**2)

# Compute Coulomb potential in reciprocal space
V_G = np.where(G > 1e-6, 4 * np.pi * e2 / (epsilon * G**2), 0)  # Avoid G=0 divergence

import matplotlib.pyplot as plt
plt.imshow(V_G, extent=[-G_max, G_max, -G_max, G_max], origin='lower', cmap='hot')
plt.colorbar(label="Coulomb Potential V(G)")
plt.title("Coulomb Potential in Reciprocal Space")
plt.show()
```

For the BSE we need to transform the Coulomb potential in the Bloch basis

$$
W_{v c, v^{\prime} c^{\prime}}\left(k, k^{\prime}\right)=\sum_{n m n^{\prime} m^{\prime}} u_n^v(k)^* u_m^c(k) V_{n m, n^{\prime} m^{\prime}}\left(k-k^{\prime}\right) u_{n^{\prime}}^v\left(k^{\prime}\right) u_{m^{\prime}}^c\left(k^{\prime}\right)^*
$$

## Structure factor and Coulomb potential TB basis

```{python}
# Define a simple tight-binding basis (Wannier functions centered at R)
lattice_vectors = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Example 2D lattice
num_sites = len(lattice_vectors)

# Define example Wannier function coefficients
phi = np.random.rand(num_sites)  # Placeholder for real Wannier function values

# Compute structure factor
S_nm = np.zeros((num_sites, num_sites), dtype=complex)
for i, Ri in enumerate(lattice_vectors):
    for j, Rj in enumerate(lattice_vectors):
        phase = np.exp(1j * (Gx * (Ri[0] - Rj[0]) + Gy * (Ri[1] - Rj[1])))
        S_nm[i, j] = np.sum(phi[i] * np.conj(phi[j]) * phase) / num_sites

# Compute Coulomb matrix in TB basis
V_TB = np.real(np.sum(S_nm * V_G[..., np.newaxis, np.newaxis], axis=(0, 1)))

print("Coulomb potential in Tight-Binding basis:\n", np.round(V_TB, 3))
```

# Small implementation example
1. Use G-vectors
In a periodic system, the Coulomb potential in reciprocal space $V(q)$ must be expanded using reciprocal lattice vectors G:

$$
V\left(k, k^{\prime}\right)=\sum_G V(G) \delta_{k^{\prime}, k+G}
$$

Instead of defining $V(q)$ only at discrete k-points, we must expand it over G-vectors to ensure periodicity.
This ensures that $V(q)$ is defined within the first Brillouin zone (BZ) while still incorporating contributions from higher reciprocal lattice vectors G.

2. Coulomb Potential in Reciprocal Space with G-Vectors
The screened Coulomb potential in periodic 2D or 3D systems is typically written as:

$$
V(G)=\frac{4 \pi e^2}{\epsilon|G|^2}
$$

This accounts for the periodicity of the crystal, ensuring the Coulomb interaction is properly expanded.

3. Implementation in Python
We now incorporate the G-vector expansion into the BSE Coulomb potential.

## Step 1: Define the Reciprocal Lattice and G-Vectors

```python
import numpy as np
import scipy.linalg as la

# Lattice parameters
a = 1.0  # Lattice constant
N = 50   # Number of k-points in the BZ

# Define reciprocal lattice G-vectors (1D case, generalize for 2D/3D)
G_max = 3  # Cutoff for reciprocal lattice vectors
G_vals = 2 * np.pi * np.arange(-G_max, G_max+1) / a  # Reciprocal lattice vectors

# Define k-space grid in the first BZ
k_vals = np.linspace(-np.pi/a, np.pi/a, N, endpoint=False)
```

## Step 2: Define the Coulomb Interaction Using G-Vectors

```python
# Define screened Coulomb potential in reciprocal space with G-expansion
def coulomb_potential_G(G, epsilon=2.5, q_TF=0.1):
    return np.where(np.abs(G) > 1e-6, 4 * np.pi * 1.44 / (epsilon * (G**2 + q_TF**2)), 0)

# Compute V(G) for all reciprocal lattice vectors
V_G = coulomb_potential_G(G_vals)

# Build full V(k, k') using G-expansion
V_k_kprime = np.zeros((N, N))

# Sum over G vectors to construct V(k, k')
for i, k in enumerate(k_vals):
    for j, k_prime in enumerate(k_vals):
        G_diff = k_prime - k  # Find the corresponding G
        closest_G = G_vals[np.argmin(np.abs(G_vals - G_diff))]  # Find closest G in grid
        V_k_kprime[i, j] = coulomb_potential_G(closest_G)
```
## Step 3: Transform Coulomb Potential to Bloch Basis
Now, we project this onto the tight-binding Bloch states.

```python
# Construct TB Hamiltonian in k-space
t = 1.0  # Hopping parameter
H_k = -2 * t * np.cos(k_vals * a)  # Dispersion for 1D tight-binding model

# Diagonalize to get eigenvalues and eigenvectors
E_tb = H_k
U_tb = np.eye(N)  # Identity because this is a simple model

# Transform Coulomb interaction to Bloch basis
V_Bloch = U_tb.T @ V_k_kprime @ U_tb
```

## Step 4: Construct and Solve the BSE Hamiltonian

```python
# Select conduction and valence bands
v_idx, c_idx = 0, -1  # First valence, last conduction band
E_v, E_c = E_tb[v_idx], E_tb[c_idx]

# Construct BSE Hamiltonian
H_BSE = np.diag(E_c - E_v) - V_Bloch

# Solve for excitonic states
exciton_energies, exciton_states = la.eigh(H_BSE)

print("Exciton Binding Energies:", np.round(exciton_energies[:5], 3))
```