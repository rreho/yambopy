import numpy as np
from typing import Optional, Tuple

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
    """For each k, return index of k−vec on the same reduced grid.
    vec may be Q or q. Shape (Nk,)."""
    Nk = kpoints.shape[0]
    lut, _ = _grid_hash(kpoints, decimals=decimals)
    out = np.empty(Nk, dtype=int)
    for ik in range(Nk):
        out[ik] = _index_of(kpoints, kpoints[ik] - vec, lut, decimals)
    return out


def build_k_plus_q_table(kpoints: np.ndarray, *, decimals: int = 10) -> np.ndarray:
    """Return table (Nk, Nk) with table[ik, iq] = index of kpoints[ik] + kpoints[iq] modulo 1."""
    Nk = kpoints.shape[0]
    lut, _ = _grid_hash(kpoints, decimals=decimals)
    tab = np.empty((Nk, Nk), dtype=int)
    for ik in range(Nk):
        k = kpoints[ik]
        for iq in range(Nk):
            q = kpoints[iq]
            tab[ik, iq] = _index_of(kpoints, k + q, lut, decimals)
    return tab


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
    ) -> None:
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
        self.ph_h = np.exp(-2j * np.pi * ( (self.k - self.Q) @ self.Sh.T ))  # (Nk, NRh)
        self.ph_e = np.exp(-2j * np.pi * ( self.k @ self.Se.T ))             # (Nk, NRe)

        # Precompute L(k) and its conjugate transpose for reuse
        # L(k)_{(mu,nu),(v,c)} = Uv(k−Q)_{mu,v} * Uc(k)_{nu,c}
        self.L = np.empty((self.Nk, self.Nm, self.Nt), dtype=complex)
        for ik in range(self.Nk):
            Uvk = self.Uv[self.kmQ[ik]]   # (NWh, Nv) at k−Q
            Uck = self.Uc[ik]             # (NWe, Nc) at k
            # Outer product on Wannier rows, band cols; then reshape to (Nm, Nt)
            # einsum: (mu,v),(nu,c) -> (mu,nu,v,c)
            tens = np.einsum('mv,nc->mnvc', Uvk, Uck, optimize=True)
            self.L[ik] = tens.reshape(self.Nm, self.Nt)
        self.LH = np.conjugate(self.L).transpose(0, 2, 1)  # (Nk, Nt, Nm)

    # -------------- Forward: Bloch -> Wtilde(Sh,Se,q) --------------
    def bloch_to_wtilde(self, K: np.ndarray) -> np.ndarray:
        """Compute Wtilde(Sh,Se,q) from K(k,k',Q).

        Inputs:
        - K: (Nk, Nk, Nv, Nc, Nv, Nc) or (Nk, Nk, Nt, Nt)

        Output:
        - Wtilde: (NRh, NRe, Nk, Nm, Nm) with axes (Sh, Se, iq, m, m')
        """
        if K.ndim == 6:
            K = K.reshape(self.Nk, self.Nk, self.Nt, self.Nt)
        assert K.shape == (self.Nk, self.Nk, self.Nt, self.Nt)

        # Build M(k, q) = L(k)^\dagger K(k, k+q) L(k+q)
        # For each iq (interpreted as a q index), collect k' = k+q via kpq.
        M = np.zeros((self.Nk, self.Nk, self.Nm, self.Nm), dtype=complex)
        for iq in range(self.Nk):
            for ik in range(self.Nk):
                ikp = self.kpq[ik, iq]  # k' = k+q
                # L(Nm,Nt) @ K(ik,ikp) (Nt,Nt) @ L(ikp)^† (Nt,Nm) -> (Nm,Nm)
                M[ik, iq] = self.L[ik] @ K[ik, ikp] @ self.LH[ikp]

        # Wtilde(Sh,Se,q) = sum_k e^{-i2π[(k−Q)·Sh + k·Se]} M(k,q)
        # ph_h: (Nk,NRh), ph_e: (Nk,NRe)
        # Contract over k with outer product of phases per (Sh,Se)
        Wtilde = np.zeros((self.NRh, self.NRe, self.Nk, self.Nm, self.Nm), dtype=complex)
        # Compute phase tensor P(ik, irh, ire) = ph_h(ik,irh) * ph_e(ik,ire)
        P = self.ph_h[:, :, None] * self.ph_e[:, None, :]  # (Nk, NRh, NRe)
        # Sum over k: Wtilde[irh,ire,iq] = (1/Nk) sum_k P[k,irh,ire] * M[k,iq]
        for iq in range(self.Nk):
            # weighted sum over k for each (Sh,Se)
            coeff = P.transpose(1, 2, 0)  # (NRh,NRe,Nk)
            # Accumulate with einsum over k: (NRh,NRe,Nk),(Nk,Nm,Nm)->(NRh,NRe,Nm,Nm)
            Wtilde[:, :, iq] = (1.0 / self.Nk) * np.einsum('rsk,knm->rsnm', coeff, M[:, iq], optimize=True)
        return Wtilde

    # -------------- Inverse: Wtilde -> Bloch --------------
    def wtilde_to_bloch(self, Wtilde: np.ndarray) -> np.ndarray:
        """Compute Bloch kernel K(k,k',Q) from Wtilde(Sh,Se,q).

        Input:
        - Wtilde: (NRh, NRe, Nk, Nm, Nm)

        Output:
        - K: (Nk, Nk, Nv, Nc, Nv, Nc)
        """
        assert Wtilde.shape[:3] == (self.NRh, self.NRe, self.Nk)
        assert Wtilde.shape[3:] == (self.Nm, self.Nm)

        # 𝓦(k,k',Q) = ∑_{Sh,Se} e^{+i2π[(k−Q)·Sh + k·Se]} Wtilde(Sh,Se,q=k'−k)
        # In general, the double sum over (Sh,Se) introduces a factor (NRh·NRe) when it collapses deltas.
        # To make inversion exact whenever the shells span the dual sets used in the forward, we scale by Nk/(NRh·NRe),
        # since the forward used 1/Nk.
        Wk = np.zeros((self.Nk, self.Nk, self.Nm, self.Nm), dtype=complex)
        # Precompute bra phase per (k,Sh,Se): P = exp(-i2π[(k−Q)·Sh + k·Se])
        P = self.ph_h[:, :, None] * self.ph_e[:, None, :]  # (Nk, NRh, NRe)
        for ik in range(self.Nk):
            # For each k, 𝓦(k, k+q) uses q-index iq
            phases = np.conjugate(P[ik])  # (NRh,NRe) -> exp(+i2π[(k−Q)·Sh + k·Se])
            for iq in range(self.Nk):
                ikp = self.kpq[ik, iq]
                Wk[ik, ikp] = np.einsum('rs,rsnm->nm', phases, Wtilde[:, :, iq], optimize=True)
        # Apply normalization to invert the forward's 1/Nk and compensate for shell cardinalities
        Wk *= (self.Nk / float(self.NRh * self.NRe))

        # K(k,k',Q) = L(k)^\dagger 𝓦(k,k',Q) L(k')
        Ktc = np.zeros((self.Nk, self.Nk, self.Nt, self.Nt), dtype=complex)
        for ik in range(self.Nk):
            for ikp in range(self.Nk):
                # K = L^† W L with shapes: L^†(Nt,Nm), W(Nm,Nm), L(Nm,Nt)
                Ktc[ik, ikp] = self.L[ik].T.conj() @ Wk[ik, ikp] @ self.L[ikp]
        # reshape to 6D (k,k',v,c,v',c')
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
