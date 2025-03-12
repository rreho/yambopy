# Bloch Theorem
Let $T_\mathbf{R}$ be a translation operator of vector $\mathbf{R}$. By definition $T_\mathbf{R}$ commutes with itself for different $\mathbf{R}$ $[T_\mathbf{R}, T_{\mathbf{R^\prime}}] = 0$.

If it commutes also with the Hamiltonian, $[T_\mathbf{R}, H] = 0 $ then they posses a common set of eigenstates.

Given a general function $f(\mathbf{r})$ satisfying the boundary conditions of the crystal, we can expand on a plane wave basis as follow: $f(\mathbf{r}) = \sum_{\mathbf{q}} C_{\mathbf{q}} e^{i\mathbf{q}\mathbf{r}}$. where $\mathbf{q}$ is a wavevector that can be inside or outside the Brillouin Zone.

The eigenvalues of $T_\mathbf{R}$ can be found as follow

$$
T_{\mathbf{R}} f(\boldsymbol{r})=T_{\mathbf{R}} \sum_{\boldsymbol{q}} C_{\boldsymbol{q}} e^{i \boldsymbol{q} \cdot \mathbf{r}}=\sum_{\boldsymbol{q}} C_{\boldsymbol{q}} e^{i \mathbf{q} \cdot \mathbf{r}} e^{i \boldsymbol{q} \cdot \mathbf{R}}=t_R \sum_{\boldsymbol{q}} C_{\boldsymbol{q}} e^{i \mathbf{q} \cdot \mathbf{r}}
$$ (eq:translationoperatoreig)

$t_R$ is the eigenvalue. Note that the final step is not trivial. In order for the last equal to hold, $e^{i\mathbf{q}\cdot\mathbf{R}}$ must be a constant: $\mathbf{q} \cdot \mathbf{R}=2 \pi n+\text { constant }$ and implies that $\mathbf{q} = \mathbf{k}+\mathbf{G}$, with $\mathbf{G}$ a reciprocal lattice vector satisfing $\mathbf{G}\cdot\mathbf{R} = 2\pi n$.
The eigenvalue is therefore $t_R = e^{i\mathbf{k}\cdot\mathbf{R}}$ and eigenvector could be any plane wave with momentum $q$.
A general eigenvector associated with this eigenvalue can be written as 

$$
f_{\mathbf{k}}(r)=\sum_G C_{\mathbf{k}+\mathbf{G}} e^{i(\mathbf{k}+\mathbf{G}) \cdot \mathbf{r}}=e^{i \mathbf{k} \cdot \mathbf{r}} \sum_G C_{\mathbf{k}+\mathbf{G}} e^{i \mathbf{G} \cdot \mathbf{r}}=e^{i \mathbf{k} \cdot \mathbf{r}} u_{\mathbf{k}}(r)
$$ (eq:fkofr)

The infinite sum over $\mathbf{q}$ is now reduced to an (infinite) sum over $\mathbf{G}$. Notice that, $f_\mathbf{k} = f_{\mathbf{k}+\mathbf{G}} \forall \mathbf{G} $.

To obtain the coefficients $C_{\mathbf{k+G}}$ we need to diagonalize the Hamiltonian in the space of all plane waves of momentum $\mathbf{k} +\mathbf{G}$

**Bloch's thorem:** The eigenstates $f_\mathbf{k}$ of a periodic Hamiltonian can be written as a product of a periodic function with a plane wave of momentum $\mathbf{k}$ restricted to be in the first BZ $f_{\mathbf{k}}(\boldsymbol{r})=u_{\mathbf{k}}(\boldsymbol{r}) e^{i \mathbf{k} \cdot \mathbf{r}}$ with $u_\mathbf{k}(\mathbf{r})$ periodic in $\mathbf{k}$ and $\mathbf{r}$. Furthermore, $f_{\mathbf{k}}(\mathbf{r}+\mathbf{R})= e^{i\mathbf{k}\mathbf{R}}f_{\mathbf{k}}(\mathbf{r})$.

# Solution of the Schroedinger equation
Given a general eigenfuction $\psi(\mathbf{r})$, solution of the Schroedinger equation $H\psi=E\psi$, we can expand it in a finite basis set ($\phi_i(\mathbf{r})_{i=1, \ldots, N}$) as $\psi(\mathbf{r})=\sum_{i=1}^N C_i \phi_i(\mathbf{r})$.
We can apply the minimization procedure of the Hamiltonian and we will arive at the following set of linear equations

$$
\sum_{j=1}^N\left[H_{i j}-E S_{i j}\right] C_j=0 ; \quad \forall i=1, \ldots, N
$$ (eq:schroedingerlcao)

where:
-  $H_{i j}=\int \phi_i^*(\mathbf{r}) H \phi_j(\mathbf{r}) d \mathbf{r}$
- S_{i j}=\int \phi_i^*(\mathbf{r}) \phi_j(\mathbf{r}) d \mathbf{r}

# Local potential
Assume we have a periodic local potential as $V(\mathbf{r}+\mathbf{R}) = V(\mathbf{r})$ then we can Fourier Transform it as follow

$$
V(\mathbf{r})=\sum_{\mathbf{G}} \hat{V}_{\mathbf{G}} e^{i \mathbf{G} \cdot \mathbf{r}}
$$

where $\hat{V}_{\mathbf{G}}=\int_{\text {cell }} V(\mathbf{r}) e^{-i \mathbf{G} \cdot \mathbf{r}} d \mathbf{r} / \Omega$, with $\Omega$ the unit cell volume

How can we define the $\mathbf{G}$'s?
The $\mathbf{G}$'s are reciprocal lattice vectors satisfying $\mathbf{G}\cdot\mathbf{R} = 2 n \pi $.
If $\left\{\mathbf{R}_1, \mathbf{R}_2, \mathbf{R}_3\right\}$ is the primitive cell basis then:

$\mathbf{G}_i \cdot \mathbf{R}_j=2 \pi \delta_{i j}$ with $\mathbf{G}_1=2 \pi\left(\mathbf{R}_2 \times \mathbf{R}_3\right) / \Omega$.

# Plane wave basis
A plane wave is $\phi_{\mathbf{k}}(\mathbf{r})=e^{i \mathbf{k} \cdot \mathbf{r}} / \sqrt{\Omega}$

Since we are expanding the wavefunction in a plane-wave basis, we assume:

$$
\psi_k(r) = \sum_G C_{k+G} \frac{e^{i (k+G) \cdot r}}{\sqrt{\Omega}}.
$$

The potential $V(r)$ is periodic, meaning it can be expanded in a Fourier series:

$$
V(r) = \sum_G V_G e^{i G \cdot r},
$$

where $G$ are **reciprocal lattice vectors** and \( V_G \) are the Fourier coefficients of \( V(r) \).

Now, let's look at the action of \( V(r) \) on a plane wave:

\[
V(r) e^{i k \cdot r} = \sum_G V_G e^{i G \cdot r} e^{i k \cdot r}.
\]

Using the exponent sum property:

\[
V(r) e^{i k \cdot r} = \sum_G V_G e^{i (k+G) \cdot r}.
\]

This shows that when the periodic potential \( V(r) \) acts on a plane wave \( e^{i k \cdot r} \), it **couples** it to another plane wave with wavevector \( k+G \).

---

## 2. Why is \( V_{k-q} \) Nonzero Only for Reciprocal Lattice Vectors?

When you write the matrix elements of the Hamiltonian in the plane-wave basis:

\[
H_{kq} = \langle e^{i k \cdot r} | H | e^{i q \cdot r} \rangle.
\]

Breaking it down:

\[
H_{kq} = \int e^{-i k \cdot r} \left( \frac{-\hbar^2}{2m} \nabla^2 + V(r) \right) e^{i q \cdot r} d^3r.
\]

Since plane waves are **eigenfunctions of the kinetic energy operator**, we get:

\[
H_{kq} = \delta_{kq} \frac{k^2}{2m} + \int e^{-i k \cdot r} V(r) e^{i q \cdot r} d^3r.
\]

Using the Fourier expansion of \( V(r) \):

\[
H_{kq} = \delta_{kq} \frac{k^2}{2m} + \sum_G V_G \int e^{-i k \cdot r} e^{i G \cdot r} e^{i q \cdot r} d^3r.
\]

\[
H_{kq} = \delta_{kq} \frac{k^2}{2m} + \sum_G V_G \int e^{-i (k-q) \cdot r} e^{i G \cdot r} d^3r.
\]

Since the integral gives a **nonzero contribution** only when:

\[
k - q = G.
\]