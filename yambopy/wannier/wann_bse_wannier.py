import numpy as np
from typing import Optional, Tuple

# Progress bar (fallback to no-op if tqdm not available)
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(x, **kwargs):
        return x

# Robust k-point lookup utilities with periodic KDTree
from yambopy.kpoints import build_ktree, find_kpt, make_kpositive

"""
BSE kernel Fourier transforms between Bloch and Wannier representations
=====================================================================

Conventions (reduced coordinates):
- Nk: number of k-points on a uniform grid closed under addition modulo 1.
- k, k', q, Q are reduced reciprocal vectors in [0,1)^D (D=2/3).
- q is always the difference k' - k taken on the same reduced grid.
- Wannier indices: mu (hole Wannier), nu (electron Wannier)
- Band indices: v (valence), c (conduction)
- Flattened indices:
  - t = (v, c) in [0, Nv*Nc)
  - m = (mu, nu) in [0, NWh*NWe)

Real-space indices (integers in reduced lattice basis):
- Sh = Rh - Rh'      (hole bra − ket)
- Se = Re - Re'      (electron bra − ket)
- R0 = Re' - Rh'     (relative ket e − ket h)

Formulas implemented (user notes):
- L_m,t(k) = U_val(mu, v; k−Q) * U_cond(nu, c; k)
- M_m,m'(k, q) = [ L(k) K(k, k+q) L(k+q)^\dagger ]_{m,m'}
- Wtilde_m,m'(Sh, Se, q) = (1/Nk) sum_k e^{-i2π[ (k−Q)·Sh + k·Se ]} M_m,m'(k, q)
- 𝓦_m,m'(k, k', Q) = (1/Nk) sum_{Sh,Se} e^{-i2π[ (k−Q)·Sh + k·Se ]} Wtilde_m,m'(Sh, Se, q=k'−k)
- K(k, k', Q) = L(k) 𝓦(k, k', Q) L(k')^\dagger
- W(Sh,Se,R0) = (1/Nk) sum_q e^{-i2π q·R0} Wtilde(Sh,Se,q)
- Wtilde(Sh,Se,q) = sum_{R0} e^{+i2π q·R0} W(Sh,Se,R0)

This module provides a minimal, transparent implementation of these transforms
without additional indirections. You must provide:
- kpoints: (Nk, D) reduced coordinates, closed under addition modulo 1
- Q: (D,) reduced vector
- U_val: (Nk, NWh, Nv)  Wannier expansion for valence subspace at each k
- U_cond: (Nk, NWe, Nc) Wannier expansion for conduction subspace at each k
- Sh_list: (NRh, D) integer reduced lattice vectors for Sh
- Se_list: (NRe, D) integer reduced lattice vectors for Se

Optional (will be auto-built if None):
- k_minus_Q: (Nk,) index of k−Q on the same grid
- k_plus_q: (Nk, Nk) table: for each (ik, iq) gives index of k+q

Notes:
- All exponentials use 2π and reduced coordinates.
- Unitarity of U matrices is assumed column-orthonormal (tall OK).
- We use einsum for clarity and performance.
"""


def _mod1(x: np.ndarray) -> np.ndarray:
    """Map reduced coordinates to [0,1) element-wise, preserving shape."""
    return np.mod(x, 1.0)


def _grid_hash(points: np.ndarray, decimals: int = 10) -> Tuple[dict, np.ndarray]:
    """Build a dict mapping rounded reduced coords to indices and return the rounded array."""
    pts = _mod1(np.asarray(points, float))
    r = np.round(pts, decimals=decimals)
    lut = {tuple(r[i]): i for i in range(r.shape[0])}
    return lut, r


def _index_of(points: np.ndarray, target: np.ndarray, lut: dict, decimals: int = 10) -> int:
    """Find index in points for target reduced coord using rounding. Raises on miss."""
    key = tuple(np.round(_mod1(target), decimals=decimals))
    if key not in lut:
        raise KeyError(f"Point {key} not in grid (decimals={decimals}).")
    return lut[key]


def build_k_minus_vector_indices(kpoints: np.ndarray, vec: np.ndarray, *, decimals: int = 10) -> np.ndarray:
    """For each k, return index of k−vec on the same reduced grid using periodic KDTree lookup.
    vec may be Q or q. Shape (Nk,)."""
    Nk = kpoints.shape[0]
    # Build periodic KDTree for robust lookup
    tree = build_ktree(np.asarray(kpoints, float))
    out = np.empty(Nk, dtype=int)
    # Map decimals to tolerance ~ half-ulp at given rounding precision
    # Use a reasonably loose tolerance to account for non-canonical k ranges and float noise
    tol = max(1e-5, 0.5 * 10.0**(-decimals))
    for ik in range(Nk):
        target = make_kpositive(np.asarray(kpoints[ik] - vec, float))
        out[ik] = int(find_kpt(tree, target, tol=tol))
    return out


def build_k_plus_q_table(kpoints: np.ndarray, *, decimals: int = 10) -> np.ndarray:
    """Return table (Nk, Nk) with table[ik, iq] = index of kpoints[ik] + kpoints[iq] modulo 1 using periodic KDTree lookup."""
    Nk = kpoints.shape[0]
    tree = build_ktree(np.asarray(kpoints, float))
    tab = np.empty((Nk, Nk), dtype=int)
    tol = max(1e-5, 0.5 * 10.0**(-decimals))
    for ik in range(Nk):
        k = np.asarray(kpoints[ik], float)
        for iq in range(Nk):
            q = np.asarray(kpoints[iq], float)
            target = make_kpositive(k + q)
            tab[ik, iq] = int(find_kpt(tree, target, tol=tol))
    return tab


def split_U_bloch_to_wann(U_bw: np.ndarray, val_bands: np.ndarray, cond_bands: np.ndarray) -> tuple:
    """
    Convert a Bloch->Wannier matrix U_bw (nk, nbnd, nwan) into the shapes expected by this module:
      - U_val: (nk, nwan, Nv) with columns for valence bands
      - U_cond: (nk, nwan, Nc) with columns for conduction bands
    Here U_bw[k][m, mu] maps Bloch band m at k to Wannier index mu.
    """
    U_bw = np.asarray(U_bw, complex)
    idx_v = np.asarray(val_bands, int)
    idx_c = np.asarray(cond_bands, int)
    # Take rows (bands) then transpose to (nwan, Nv/Nc)
    U_val = np.transpose(U_bw[:, idx_v, :], (0, 2, 1)).copy()
    U_cond = np.transpose(U_bw[:, idx_c, :], (0, 2, 1)).copy()
    return U_val, U_cond


class BSEWannierFT:
    def __init__(
        self,
        *,
        kpoints: np.ndarray,           # (Nk, D) reduced
        Q: np.ndarray,                 # (D,) reduced
        U_val: np.ndarray,             # (Nk, NWh, Nv)
        U_cond: np.ndarray,            # (Nk, NWe, Nc)
        Sh_list: np.ndarray,           # (NRh, D) ints
        Se_list: np.ndarray,           # (NRe, D) ints
        k_minus_Q: Optional[np.ndarray] = None,   # (Nk,)
        k_plus_q: Optional[np.ndarray] = None,    # (Nk, Nk)
        decimals: int = 10,
        precompute_L: bool = False,
        mem_dtype=np.complex64,
    ) -> None:
        
        # Basic shapes and inputs
        self.k = np.ascontiguousarray(kpoints, float)
        self.Nk, self.dim = self.k.shape
        self.Q = np.asarray(Q, float)
        self.Uv = np.asarray(U_val, complex)
        self.Uc = np.asarray(U_cond, complex)
        assert self.Uv.shape[0] == self.Nk and self.Uc.shape[0] == self.Nk
        self.NWh, self.Nv = self.Uv.shape[1], self.Uv.shape[2]
        self.NWe, self.Nc = self.Uc.shape[1], self.Uc.shape[2]
        self.Nm = self.NWh * self.NWe
        self.Nt = self.Nv * self.Nc
        self.Sh = np.asarray(Sh_list, int)
        self.Se = np.asarray(Se_list, int)
        self.NRh = self.Sh.shape[0]
        self.NRe = self.Se.shape[0]
        self.decimals = int(decimals)
        self.mem_dtype = mem_dtype
        self.precompute_L = bool(precompute_L)

        # k−Q map and k+q table
        self.kmQ = (
            build_k_minus_vector_indices(self.k, self.Q, decimals=self.decimals)
            if k_minus_Q is None else np.asarray(k_minus_Q, int)
        )
        self.kpq = (
            build_k_plus_q_table(self.k, decimals=self.decimals)
            if k_plus_q is None else np.asarray(k_plus_q, int)
        )

        # Precompute phase factors: ph_h(ik,irh) = e^{-i2π (k−Q)·Sh}, ph_e(ik,ire) = e^{-i2π k·Se}
        # Use lower-precision dtype to reduce RAM pressure.
        self.ph_h = np.exp(-2j * np.pi * ((self.k - self.Q) @ self.Sh.T)).astype(self.mem_dtype, copy=False)  # (Nk, NRh)
        self.ph_e = np.exp(-2j * np.pi * (self.k @ self.Se.T)).astype(self.mem_dtype, copy=False)             # (Nk, NRe)

        # L buffers (optionally large). In low-memory mode, build on-the-fly.
        if self.precompute_L:
            self.L = np.empty((self.Nk, self.Nm, self.Nt), dtype=self.mem_dtype)
            for ik in _tqdm(range(self.Nk), desc="precompute L", leave=False):
                Uvk = self.Uv[self.kmQ[ik]]   # (NWh, Nv) at k−Q
                Uck = self.Uc[ik]             # (NWe, Nc) at k
                tens = np.einsum('mv,nc->mnvc', Uvk, Uck, optimize=True)
                self.L[ik] = tens.reshape(self.Nm, self.Nt).astype(self.mem_dtype, copy=False)
            self.LH = np.conjugate(self.L).transpose(0, 2, 1)  # (Nk, Nt, Nm)
        else:
            self.L = None
            self.LH = None

    @classmethod
    def from_U_bloch_to_wann(
        cls,
        *,
        kpoints: np.ndarray,
        Q: np.ndarray,
        U_bloch_to_wann: np.ndarray,   # (Nk, Nbnd, Nw)
        val_bands: np.ndarray,         # (Nv,)
        cond_bands: np.ndarray,        # (Nc,)
        Sh_list: np.ndarray,
        Se_list: np.ndarray,
        k_minus_Q: Optional[np.ndarray] = None,
        k_plus_q: Optional[np.ndarray] = None,
        decimals: int = 10,
        precompute_L: bool = False,
        mem_dtype=np.complex64,
    ):
        """Alternate constructor that accepts the combined Bloch->Wannier rotations.
        It slices into valence/conduction subsets and transposes to match (Nw, Nv/Nc)."""
        U_val, U_cond = split_U_bloch_to_wann(U_bloch_to_wann, val_bands, cond_bands)
        return cls(
            kpoints=kpoints,
            Q=Q,
            U_val=U_val,
            U_cond=U_cond,
            Sh_list=Sh_list,
            Se_list=Se_list,
            k_minus_Q=k_minus_Q,
            k_plus_q=k_plus_q,
            decimals=decimals,
            precompute_L=precompute_L,
            mem_dtype=mem_dtype,
        )

    def _build_L_at(self, ik: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (L, LH) for k-index ik without caching huge arrays when precompute_L=False."""
        if self.precompute_L:
            return self.L[ik], self.LH[ik]
        Uvk = self.Uv[self.kmQ[ik]]  # (NWh,Nv)
        Uck = self.Uc[ik]            # (NWe,Nc)
        tens = np.einsum('mv,nc->mnvc', Uvk, Uck, optimize=True)
        L = tens.reshape(self.Nm, self.Nt).astype(self.mem_dtype, copy=False)
        LH = L.conj().T
        return L, LH

    @staticmethod
    def W_to_wtilde_at_q_arbitrary(W: np.ndarray, R0_list: np.ndarray, q_list: np.ndarray) -> np.ndarray:
        """Fourier-transform W(Sh,Se,R0) to Wtilde(Sh,Se,q) for an arbitrary list of q-vectors.
        Inputs:
          - W: (NRh,NRe,N0,Nm,Nm)
          - R0_list: (N0,D) integer reduced lattice vectors
          - q_list: (Nq,D) reduced q points (not necessarily on the original grid)
        Returns:
          - Wtilde: (NRh,NRe,Nq,Nm,Nm)
        """
        R0 = np.asarray(R0_list, int)
        q = np.asarray(q_list, float)
        assert W.ndim == 5 and W.shape[2] == R0.shape[0]
        # phase ph(q,a) = exp(+i2π q·R0_a)
        ph = np.exp(+2j * np.pi * (q @ R0.T))  # (Nq, N0)
        # einsum over R0 index 'a'
        Wtilde = np.einsum('rsanm, qa -> rsqnm', W, ph, optimize=True)
        return Wtilde

    @staticmethod
    def reconstruct_bloch_for_offgrid_q(
        *,
        Wtilde_q: np.ndarray,             # (NRh,NRe,Nm,Nm) for a single q
        kpoints: np.ndarray,              # (Nk,D)
        Sh_list: np.ndarray,              # (NRh,D)
        Se_list: np.ndarray,              # (NRe,D)
        q_vec: np.ndarray,                # (D,)
        Uv_k_minus_q: np.ndarray,         # (Nk,NWh,Nv)
        Uc_k: np.ndarray,                 # (Nk,NWe,Nc)
        Uv_k: np.ndarray,                 # (Nk,NWh,Nv)
        Uc_k_plus_q: np.ndarray,          # (Nk,NWe,Nc)
    ) -> np.ndarray:
        """Reconstruct K(k,k+q) for an off-grid q using provided U matrices.
        Returns K_per_k of shape (Nk, Nv, Nc, Nv, Nc).
        NOTE: This legacy routine uses phases with (k−q)·Sh; prefer reconstruct_bloch_for_qQ.
        """
        k = np.asarray(kpoints, float)
        Sh = np.asarray(Sh_list, int)
        Se = np.asarray(Se_list, int)
        Nk = k.shape[0]
        NRh = Sh.shape[0]
        NRe = Se.shape[0]
        NWh, Nv = Uv_k_minus_q.shape[1], Uv_k_minus_q.shape[2]
        NWe, Nc = Uc_k.shape[1], Uc_k.shape[2]
        Nm = NWh * NWe
        assert Wtilde_q.shape == (NRh, NRe, Nm, Nm)

        # Phases exp(+i2π[(k−q)·Sh + k·Se]) per k
        ph_h = np.exp(+2j * np.pi * ((k - q_vec) @ Sh.T))  # (Nk,NRh)
        ph_e = np.exp(+2j * np.pi * (k @ Se.T))            # (Nk,NRe)

        # Output container in transition basis then reshape
        Ktc_per_k = np.zeros((Nk, Nv * Nc, Nv * Nc), dtype=complex)

        # Precompute normalization factor Nk/(NRh*NRe)
        norm = Nk / float(NRh * NRe)

        for ik in range(Nk):
            # Sum over shells to get Wk block in Wannier-transition basis (Nm,Nm)
            phases = (ph_h[ik][:, None] * ph_e[ik][None, :])  # (NRh,NRe)
            Wk_block = norm * np.einsum('rs, rsnm -> nm', phases, Wtilde_q, optimize=True)

            # Build L(k) and L(k+q)
            tens_k = np.einsum('mv,nc->mnvc', Uv_k_minus_q[ik], Uc_k[ik], optimize=True)
            Lk = tens_k.reshape(Nm, Nv * Nc)
            tens_kp = np.einsum('mv,nc->mnvc', Uv_k[ik], Uc_k_plus_q[ik], optimize=True)
            Lkp = tens_kp.reshape(Nm, Nv * Nc)

            # Project back: K_tc = L^† Wk L'
            Ktc_per_k[ik] = Lk.conj().T @ Wk_block @ Lkp

        # Reshape to (Nk,Nv,Nc,Nv,Nc)
        return Ktc_per_k.reshape(Nk, Nv, Nc, Nv, Nc)

    @staticmethod
    def reconstruct_bloch_for_qQ(
        *,
        Wtilde_q: np.ndarray,             # (NRh,NRe,Nm,Nm) for a single q
        kpoints: np.ndarray,              # (Nk,D)
        Sh_list: np.ndarray,              # (NRh,D)
        Se_list: np.ndarray,              # (NRe,D)
        q_vec: np.ndarray,                # (D,)
        Q_vec: np.ndarray,                # (D,)
        Uv_k_minus_Q: np.ndarray,         # (Nk,NWh,Nv)
        Uc_k: np.ndarray,                 # (Nk,NWe,Nc)
        Uv_k_plus_q_minus_Q: np.ndarray,  # (Nk,NWh,Nv)
        Uc_k_plus_q: np.ndarray,          # (Nk,NWe,Nc)
    ) -> np.ndarray:
        """Reconstruct K(k,k+q; Q) using the general formula with (k−Q) in hole sector.
        Returns K_per_k of shape (Nk, Nv, Nc, Nv, Nc).
        """
        k = np.asarray(kpoints, float)
        Sh = np.asarray(Sh_list, int)
        Se = np.asarray(Se_list, int)
        Nk = k.shape[0]
        NRh = Sh.shape[0]
        NRe = Se.shape[0]
        NWh, Nv = Uv_k_minus_Q.shape[1], Uv_k_minus_Q.shape[2]
        NWe, Nc = Uc_k.shape[1], Uc_k.shape[2]
        Nm = NWh * NWe
        assert Wtilde_q.shape == (NRh, NRe, Nm, Nm)

        # Phases exp(+i2π[(k−Q)·Sh + k·Se]) per k
        ph_h = np.exp(+2j * np.pi * ((k - Q_vec) @ Sh.T))  # (Nk,NRh)
        ph_e = np.exp(+2j * np.pi * (k @ Se.T))            # (Nk,NRe)

        # Output container in transition basis then reshape
        Ktc_per_k = np.zeros((Nk, Nv * Nc, Nv * Nc), dtype=complex)

        # Precompute normalization factor Nk/(NRh*NRe)
        norm = Nk / float(NRh * NRe)

        for ik in range(Nk):
            # Sum over shells to get Wk block in Wannier-transition basis (Nm,Nm)
            phases = (ph_h[ik][:, None] * ph_e[ik][None, :])  # (NRh,NRe)
            Wk_block = norm * np.einsum('rs, rsnm -> nm', phases, Wtilde_q, optimize=True)

            # Build L(k) and L(k+q) with Uv evaluated at k−Q and k+q−Q
            tens_k = np.einsum('mv,nc->mnvc', Uv_k_minus_Q[ik], Uc_k[ik], optimize=True)
            Lk = tens_k.reshape(Nm, Nv * Nc)
            tens_kp = np.einsum('mv,nc->mnvc', Uv_k_plus_q_minus_Q[ik], Uc_k_plus_q[ik], optimize=True)
            Lkp = tens_kp.reshape(Nm, Nv * Nc)

            # Project back: K_tc = L^† Wk L'
            Ktc_per_k[ik] = Lk.conj().T @ Wk_block @ Lkp

        # Reshape to (Nk,Nv,Nc,Nv,Nc)
        return Ktc_per_k.reshape(Nk, Nv, Nc, Nv, Nc)

    # -------------- Forward: Bloch -> Wtilde(Sh,Se,q) --------------
    def bloch_to_wtilde(self, K: np.ndarray) -> np.ndarray:
        """Compute Wtilde(Sh,Se,q) from K(k,k',Q) with streaming to minimize RAM.

        Inputs:
        - K: (Nk, Nk, Nv, Nc, Nv, Nc) or (Nk, Nk, Nt, Nt)

        Output:
        - Wtilde: (NRh, NRe, Nk, Nm, Nm) with axes (Sh, Se, iq, m, m')
        """
        if K.ndim == 6:
            K = K.reshape(self.Nk, self.Nk, self.Nt, self.Nt)
        assert K.shape == (self.Nk, self.Nk, self.Nt, self.Nt)

        # Phase tensor P(ik, irh, ire) = ph_h(ik,irh) * ph_e(ik,ire)
        P = (self.ph_h[:, :, None] * self.ph_e[:, None, :]).astype(self.mem_dtype, copy=False)  # (Nk, NRh, NRe)

        # Allocate output; compute one iq-slab at a time to avoid storing M(k,iq,...) for all k
        Wtilde = np.zeros((self.NRh, self.NRe, self.Nk, self.Nm, self.Nm), dtype=self.mem_dtype)

        for iq in _tqdm(range(self.Nk), desc="Bloch->Wtilde (per q)", leave=False):
            # Accumulator for this q: (NRh,NRe,Nm,Nm)
            acc = np.zeros((self.NRh, self.NRe, self.Nm, self.Nm), dtype=self.mem_dtype)
            for ik in range(self.Nk):
                ikp = self.kpq[ik, iq]
                Lk, _LHk = self._build_L_at(ik)
                Lkp, LHkp = self._build_L_at(ikp)
                # M_block = L(k) K(ik,ikp) L(ikp)^†  -> (Nm,Nm)
                M_block = (Lk @ K[ik, ikp] @ LHkp).astype(self.mem_dtype, copy=False)
                # Weighted add over k phases for this (iq): acc += P[ik] (NRh,NRe) x M_block (Nm,Nm)
                acc += np.einsum('rs, nm -> rsnm', P[ik], M_block, optimize=True)
            Wtilde[:, :, iq] = (1.0 / self.Nk) * acc
        return Wtilde

    # -------------- Inverse: Wtilde -> Bloch --------------
    def wtilde_to_bloch(self, Wtilde: np.ndarray) -> np.ndarray:
        """Compute Bloch kernel K(k,k',Q) from Wtilde(Sh,Se,q) using streaming to limit RAM.

        Input:
        - Wtilde: (NRh, NRe, Nk, Nm, Nm)

        Output:
        - K: (Nk, Nk, Nv, Nc, Nv, Nc)
        """
        assert Wtilde.shape[:3] == (self.NRh, self.NRe, self.Nk)
        assert Wtilde.shape[3:] == (self.Nm, self.Nm)

        # Precompute phase tensor P = exp(-i2π[(k−Q)·Sh + k·Se])
        P = (self.ph_h[:, :, None] * self.ph_e[:, None, :]).astype(self.mem_dtype, copy=False)  # (Nk, NRh, NRe)

        # Output tensor in transition basis
        Ktc = np.zeros((self.Nk, self.Nk, self.Nt, self.Nt), dtype=self.mem_dtype)
        scale = self.Nk / float(self.NRh * self.NRe)

        # Stream over k and q to avoid holding W(k,k') for all pairs
        for ik in _tqdm(range(self.Nk), desc="Wtilde->Bloch (per k)", leave=False):
            phases = np.conjugate(P[ik])  # (NRh,NRe) -> exp(+i2π[(k−Q)·Sh + k·Se])
            Lk, LHk = self._build_L_at(ik)
            for iq in range(self.Nk):
                ikp = self.kpq[ik, iq]
                # 𝓦_block(k,k') = sum_{Sh,Se} phases * Wtilde(Sh,Se,iq)
                W_block = np.einsum('rs,rsnm->nm', phases, Wtilde[:, :, iq], optimize=True)
                W_block = (scale * W_block).astype(self.mem_dtype, copy=False)
                # Project: Ktc[ik,ikp] = L(k)^† 𝓦_block L(k')
                Lkp, _ = self._build_L_at(ikp)
                Ktc[ik, ikp] = LHk @ W_block @ Lkp

        K6 = Ktc.reshape(self.Nk, self.Nk, self.Nv, self.Nc, self.Nv, self.Nc)
        return K6

    # -------------- Optional: q-FT to/from R0 --------------
    def wtilde_to_W(self, Wtilde: np.ndarray, R0_list: np.ndarray) -> np.ndarray:
        """Inverse FT over q to get W(Sh,Se,R0).
        Inputs:
          - Wtilde: (NRh,NRe,Nk,Nm,Nm)
          - R0_list: (N0, D) ints
        Returns:
          - W: (NRh,NRe,N0,Nm,Nm)
        """
        assert Wtilde.shape[:3] == (self.NRh, self.NRe, self.Nk)
        R0 = np.asarray(R0_list, int)
        N0 = R0.shape[0]
        # ph_q(R0, iq) = e^{-i2π q·R0}, with q enumerated by k-grid
        q = self.k  # reuse k-grid as q-grid
        ph = np.exp(-2j * np.pi * (q @ R0.T))  # (Nk, N0)
        # sum over q: (NRh,NRe,Nk,Nm,Nm) with (Nk,N0) -> (NRh,NRe,N0,Nm,Nm)
        # NOTE: einsum subscripts cannot use digits; use 'a' for N0 index
        W = (1.0 / self.Nk) * np.einsum('rsqnm, qa -> rsanm', Wtilde, ph, optimize=True)
        return W

    def W_to_wtilde(self, W: np.ndarray, R0_list: np.ndarray) -> np.ndarray:
        """FT over R0 to get Wtilde(Sh,Se,q).
        Input:
          - W: (NRh,NRe,N0,Nm,Nm)
          - R0_list: (N0, D) ints
        Returns:
          - Wtilde: (NRh,NRe,Nk,Nm,Nm)
        """
        R0 = np.asarray(R0_list, int)
        N0 = R0.shape[0]
        assert W.shape[:3] == (self.NRh, self.NRe, N0)
        q = self.k
        ph = np.exp(+2j * np.pi * (q @ R0.T))  # (Nk, N0)
        # sum over R0: (NRh,NRe,N0,Nm,Nm) with (Nk,N0) -> (NRh,NRe,Nk,Nm,Nm)
        # NOTE: einsum subscripts cannot use digits; use 'a' for N0 index
        Wtilde = np.einsum('rsanm, qa -> rsqnm', W, ph, optimize=True)
        return Wtilde

    # -------------- HDF5-backed streaming I/O --------------
    def bloch_to_wtilde_hdf5(self, K: np.ndarray, h5_path: str, dataset: str = "Wtilde", overwrite: bool = True) -> None:
        """Stream Bloch->Wtilde to disk, writing per-q slabs to HDF5 to limit RAM.
        Dataset shape: (NRh, NRe, Nk, Nm, Nm), complex.
        """
        if K.ndim == 6:
            K = K.reshape(self.Nk, self.Nk, self.Nt, self.Nt)
        assert K.shape == (self.Nk, self.Nk, self.Nt, self.Nt)

        # Lazy import to avoid hard dependency
        import h5py  # type: ignore
        mode = 'w' if overwrite else 'a'
        with h5py.File(h5_path, mode) as h5:
            if dataset in h5 and overwrite:
                del h5[dataset]
            dset = h5.create_dataset(dataset, shape=(self.NRh, self.NRe, self.Nk, self.Nm, self.Nm), dtype=np.complex64)

            P = (self.ph_h[:, :, None] * self.ph_e[:, None, :]).astype(self.mem_dtype, copy=False)  # (Nk,NRh,NRe)
            for iq in _tqdm(range(self.Nk), desc="Bloch->Wtilde[h5] (per q)", leave=False):
                acc = np.zeros((self.NRh, self.NRe, self.Nm, self.Nm), dtype=self.mem_dtype)
                for ik in range(self.Nk):
                    ikp = self.kpq[ik, iq]
                    Lk, _ = self._build_L_at(ik)
                    _, LHkp = self._build_L_at(ikp)
                    M_block = (Lk @ K[ik, ikp] @ LHkp).astype(self.mem_dtype, copy=False)
                    acc += np.einsum('rs, nm -> rsnm', P[ik], M_block, optimize=True)
                dset[:, :, iq, :, :] = (1.0 / self.Nk) * acc

    def wtilde_hdf5_to_bloch(self, h5_path: str, dataset: str = "Wtilde") -> np.ndarray:
        """Stream Wtilde (stored per-q in HDF5) back to Bloch K(k,k',Q) with low RAM.
        Returns K6 tensor with shape (Nk,Nk,Nv,Nc,Nv,Nc).
        """
        import h5py  # type: ignore
        with h5py.File(h5_path, 'r') as h5:
            W = h5[dataset]
            # Output tensor in transition basis
            Ktc = np.zeros((self.Nk, self.Nk, self.Nt, self.Nt), dtype=self.mem_dtype)
            scale = self.Nk / float(self.NRh * self.NRe)

            P = (self.ph_h[:, :, None] * self.ph_e[:, None, :]).astype(self.mem_dtype, copy=False)
            for ik in _tqdm(range(self.Nk), desc="Wtilde[h5]->Bloch (per k)", leave=False):
                phases = np.conjugate(P[ik])
                Lk, LHk = self._build_L_at(ik)
                for iq in range(self.Nk):
                    ikp = self.kpq[ik, iq]
                    W_block = np.einsum('rs,rsnm->nm', phases, W[:, :, iq, :, :], optimize=True)
                    W_block = (scale * W_block).astype(self.mem_dtype, copy=False)
                    Lkp, _ = self._build_L_at(ikp)
                    Ktc[ik, ikp] = LHk @ W_block @ Lkp

        return Ktc.reshape(self.Nk, self.Nk, self.Nv, self.Nc, self.Nv, self.Nc)

    # -------------- Diagnostics --------------
    def check_unitarity(self, tol: float = 1e-8) -> Tuple[float, float]:
        """Return max deviations from unitarity for U_val and U_cond columns."""
        Iv = np.eye(self.Nv, dtype=complex)
        Ic = np.eye(self.Nc, dtype=complex)
        dv = 0.0
        dc = 0.0
        for k in range(self.Nk):
            V = self.Uv[k]
            C = self.Uc[k]
            dv = max(dv, float(np.linalg.norm(V.conj().T @ V - Iv, 2)))
            dc = max(dc, float(np.linalg.norm(C.conj().T @ C - Ic, 2)))
        return dv, dc


def kernel_realspace_roundtrip(
    K_band: np.ndarray,
    *,
    kpoints: np.ndarray,
    Q: np.ndarray,
    U_val: np.ndarray,
    U_cond: np.ndarray,
    Sh_list: np.ndarray,
    Se_list: np.ndarray,
    R0_list: Optional[np.ndarray] = None,
    centered: bool = False,
    decimals: int = 10,
):
    """Compute real-space kernel W(Sh,Se,R0) from band kernel K and reconstruct K.

    Returns (W, K_back, R0_used).
    If R0_list is None, build the full dual grid indices inferred from kpoints.
    """
    def _infer_grid_shape_from_k(kpts: np.ndarray, decimals: int = 10):
        k = np.mod(np.asarray(kpts, float), 1.0)
        r = np.round(k, decimals=decimals)
        xs = np.unique(r[:, 0]); ys = np.unique(r[:, 1]); zs = np.unique(r[:, 2])
        return len(xs), len(ys), len(zs)

    def _dual_indices(n: int, centered: bool) -> np.ndarray:
        return np.arange(-(n // 2), n - (n // 2), dtype=int) if centered else np.arange(n, dtype=int)

    def _build_dual_from_k(kpts: np.ndarray, centered: bool, decimals: int = 10) -> np.ndarray:
        nkx, nky, nkz = _infer_grid_shape_from_k(kpts, decimals)
        ix = _dual_indices(nkx, centered); iy = _dual_indices(nky, centered); iz = _dual_indices(nkz, centered)
        grid = np.array([[x, y, z] for x in ix for y in iy for z in iz], dtype=int)
        return grid

    kpoints = np.asarray(kpoints, float)
    Q = np.asarray(Q, float)

    if R0_list is None:
        R0_list = _build_dual_from_k(kpoints, centered=centered, decimals=decimals)

    # Build transformer and do the round-trip
    ft = BSEWannierFT(kpoints=kpoints, Q=Q, U_val=U_val, U_cond=U_cond, Sh_list=Sh_list, Se_list=Se_list, decimals=decimals)
    Wtilde = ft.bloch_to_wtilde(K_band)
    W = ft.wtilde_to_W(Wtilde, R0_list)
    Wtilde_back = ft.W_to_wtilde(W, R0_list)
    K_back = ft.wtilde_to_bloch(Wtilde_back)
    return W, K_back, R0_list
