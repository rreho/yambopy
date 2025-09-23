import numpy as np
from typing import Dict, Tuple, Optional, Iterable
from tqdm import tqdm
try:
    import h5py  # For storage of compressed Wannier integrals
except Exception:  # pragma: no cover
    h5py = None

# -----------------------------------------------------------------------------
# BSE kernel: band(k,k') -> Wannier real-space integrals and back
# -----------------------------------------------------------------------------
# Conventions (k−q on valence, k on conduction)
# - k-grid: Nk points (reduced coordinates). Indexing arrays must be aligned
# - q: total (COM) momentum, in reduced coordinates
# - U_val[k] has shape (Nw_h, Nv), U_cond[k] has shape (Nw_e, Nc)
#   mapping band index -> Wannier index, consistent with:
#   psi_{n,k}(r) = sum_{R,mu} e^{i k·R} U_{mu n}^*(k) w_{R mu}(r)
# - Kernel in band basis is provided for transitions (v,k−q) -> (c,k)
#   K_band[k,kp,v,c,vp,cp]
#   or flattened to K_band[k,kp,tc,tp] with tc = v*Nc + c
# - The target real-space integrals are W[ΔRh,ΔRe, μh, μe, μh', μe']
#   representing ⟨Rh μh, Re μe | K | Rh' μh', Re' μe'⟩ with ΔR = R - R'
# - All phases and conjugations follow the convention above.
# -----------------------------------------------------------------------------


class BSEWannierTransformer:
    def __init__(
        self,
        kpoints: np.ndarray,  # (Nk, 3) reduced
        qvec: np.ndarray,     # (3,) reduced
        U_val: np.ndarray,    # (Nk, Nw_h, Nv)
        U_cond: np.ndarray,   # (Nk, Nw_e, Nc) evaluated on same grid as kpoints
        K_band: np.ndarray,   # (Nk, Nk, Nv, Nc, Nv, Nc) or (Nk, Nk, Nv*Nc, Nv*Nc)
        *,
        k_plus_q_indices: Optional[np.ndarray] = None,  # (Nk,) index of k+q on grid; if None, assume U_cond already corresponds to k+q
        delta_R_h_list: Optional[np.ndarray] = None,    # (NRh,3) integer lattice vectors
        delta_R_e_list: Optional[np.ndarray] = None,    # (NRe,3)
        prune_tol: float = 0.0,                         # zero-out small values when storing W
        norm_factor: float = None,                      # if None, defaults to 1/Nk^2
    ) -> None:
        self.kpoints = np.ascontiguousarray(kpoints, dtype=float)
        self.Nk = int(self.kpoints.shape[0])
        self.qvec = np.asarray(qvec, dtype=float)

        # Store original U (unshifted) and keep an internally aligned copy at k−q
        self.U_val_in = np.asarray(U_val)   # (Nk, Nw_h, Nv) at k
        self.U_val = self.U_val_in          # backward-compat alias
        self.U_cond_in = np.asarray(U_cond) # (Nk, Nw_e, Nc) at k

        # Shapes
        assert self.U_val_in.shape[0] == self.Nk, "U_val must have first axis Nk"
        assert self.U_cond_in.shape[0] == self.Nk, "U_cond must have first axis Nk"
        # Derive Nv/Nc from U, not from K_band (which may be 4D)
        self.Nw_h, self.Nv = int(self.U_val_in.shape[1]), int(self.U_val_in.shape[2])
        self.Nw_e, self.Nc = int(self.U_cond_in.shape[1]), int(self.U_cond_in.shape[2])
        # Note: U_v (Nw_h x Nv) and U_c (Nw_e x Nc) can be rectangular (tall). We only require column-orthonormality:
        # U_v^\u2020 U_v = I_Nv and U_c^\u2020 U_c = I_Nc, which implies L^\u2020 L = I_{NvNc}.

        # Kernel shape -> (Nk, Nk, Ntc, Ntc)
        if K_band.ndim == 6:
            assert K_band.shape[:2] == (self.Nk, self.Nk)
            #assert K_band.shape[2] == self.Nv and K_band.shape[3] == self.Nc
            #assert K_band.shape[4] == self.Nv and K_band.shape[5] == self.Nc
            self.Ntc = self.Nv * self.Nc
            self.K_band = self._reshape_kernel_to_tc(K_band)
        elif K_band.ndim == 4:
            assert K_band.shape[:2] == (self.Nk, self.Nk)
            self.Ntc = int(K_band.shape[2])
            assert self.Ntc == self.Nv * self.Nc, "Flattened kernel Ntc must equal Nv*Nc"
            self.K_band = np.asarray(K_band)
        else:
            raise ValueError("K_band must be 6D (Nk,Nk,Nv,Nc,Nv,Nc) or 4D (Nk,Nk,Ntc,Ntc)")

        # Map k -> k−q for valence rotations and k for conduction
        # Expect k_minus_q_indices to map k -> k−q on the same grid. If not provided,
        # build it from (kpoints, qvec). We treat U_val_in as given at k (not aligned).
        if k_plus_q_indices is None:
            self.k_minus_q = self._k_minus_q_indices_for_grid(self.kpoints, self.qvec)
            self.U_val_aligned = False
        else:
            # Backward compatibility: allow passing k−q indices via the same arg name
            assert k_plus_q_indices.shape == (self.Nk,)
            self.k_minus_q = k_plus_q_indices.astype(int)
            self.U_val_aligned = False

        # Keep original U as provided (aligned or not); no Q-phase is baked into W
        # We'll select between U_val_in[k] vs U_val_in[k_minus_q[k]] when building contractions.

        # ΔR lists
        if delta_R_h_list is None:
            delta_R_h_list = np.array([[0, 0, 0]], dtype=int)
        if delta_R_e_list is None:
            delta_R_e_list = np.array([[0, 0, 0]], dtype=int)
        self.delta_R_h = np.asarray(delta_R_h_list, dtype=int)
        self.delta_R_e = np.asarray(delta_R_e_list, dtype=int)
        self.NRh = int(self.delta_R_h.shape[0])
        self.NRe = int(self.delta_R_e.shape[0])

        self.norm_factor = (1.0 / (self.Nk * self.Nk)) if norm_factor is None else float(norm_factor)
        self.prune_tol = float(prune_tol)

        # Precompute phases for inverse FT used in compute_W (DO include q on the hole branch):
        # phase_h = e^{-i (k−q)·ΔR_h}, phase_e = e^{-i k·ΔR_e}
        # These phases are used only to extract W from K; W stored remains Q-independent.
        self.phase_h = np.exp(-2j * np.pi * ((self.kpoints - self.qvec) @ self.delta_R_h.T))  # (Nk, NRh)
        self.phase_e = np.exp(-2j * np.pi * (self.kpoints @ self.delta_R_e.T))  # (Nk, NRe)

        # Precompute L[k] and R[k'] mappings, flattened in transition space tc = v*Nc + c
        # L[k] shape (Nme, Ntc) where Nme = Nw_h * Nw_e
        self.Nme = self.Nw_h * self.Nw_e
        self._L = np.empty((self.Nk, self.Nme, self.Ntc), dtype=np.complex128)
        self._R = np.empty((self.Nk, self.Nme, self.Ntc), dtype=np.complex128)
        for k in range(self.Nk):
            # Outer product over (μh, μe) and (v,c) with broadcasting -> then reshape
            # Use U_val at k−q if mapping available; otherwise use as-provided
            # Always align valence at k−q using mapping
            Lv = self.U_val_in[self.k_minus_q[k]]
            Lc = self.U_cond_in[k]          # (Nw_e, Nc) at k
            Ltc = np.einsum('hv,ec->hevc', Lv, Lc, optimize=True).reshape(self.Nme, self.Ntc)
            self._L[k] = Ltc
            # For R: same definition; conjugation will be applied where needed during contractions
            self._R[k] = Ltc

        # Build a projector in Wannier-me space capturing the union span of L(k)
        # P_me = U_basis U_basis^H, where U_basis are eigenvectors of sum_k P(k) with eigenvalues > tol
        sumP = np.zeros((self.Nme, self.Nme), dtype=np.complex128)
        for k in range(self.Nk):
            Lk = self._L[k]
            sumP += Lk @ Lk.conj().T  # (Nme,Nme)
        # Hermitian by construction; eigen-decompose and keep significant subspace
        w, U = np.linalg.eigh(sumP)
        tol_proj = 1e-10 * max(1, self.Nk)
        mask = w > tol_proj
        if not np.any(mask):
            # Fallback: keep the largest eigenspace
            idx = np.argmax(w)
            U_basis = U[:, [idx]]
        else:
            U_basis = U[:, mask]
        self.P_me = U_basis @ U_basis.conj().T  # idempotent projector onto span{L(k)}

    def check_unitarity(self, tol: float = 1e-8) -> Dict[str, float]:
        """Return worst-case deviations from unitarity for U_val and U_cond over k.
        If U are rectangular, checks U^\u2020 U \u2248 I in band subspace.
        """
        max_dev_val = 0.0
        max_dev_cond = 0.0
        I_v = np.eye(self.Nv, dtype=np.complex128)
        I_c = np.eye(self.Nc, dtype=np.complex128)
        for k in range(self.Nk):
            V = self.U_val_in[k]
            C = self.U_cond_in[k]
            dev_v = np.linalg.norm(V.conj().T @ V - I_v, ord=2)
            dev_c = np.linalg.norm(C.conj().T @ C - I_c, ord=2)
            max_dev_val = max(max_dev_val, float(dev_v))
            max_dev_cond = max(max_dev_cond, float(dev_c))
        ok = (max_dev_val <= tol) and (max_dev_cond <= tol)
        if not ok:
            print(f"Warning: Unitarity deviations: val={max_dev_val:.2e}, cond={max_dev_cond:.2e}")
        return {"max_dev_val": max_dev_val, "max_dev_cond": max_dev_cond, "ok": float(bool(ok))}

    @staticmethod
    def _reshape_kernel_to_tc(K6: np.ndarray) -> np.ndarray:
        Nk = K6.shape[0]
        Nv, Nc = K6.shape[2], K6.shape[3]
        Ntc = Nv * Nc
        Ktc = K6.reshape(Nk, Nk, Nv, Nc, Nv, Nc)
        Ktc = np.transpose(Ktc, (0, 1, 2, 3, 4, 5)).reshape(Nk, Nk, Ntc, Ntc)
        return Ktc

    # ---------------------------------------------------------------------
    # Core: band -> Wannier real-space integrals
    # ---------------------------------------------------------------------
    def compute_W(self, chunk_k: int = 8, chunk_kp: int = 8) -> np.ndarray:
        """
        Compute Wannier-space two-particle integrals W[ΔRh,ΔRe, μh, μe, μh', μe'].

        Parameters
        ----------
        Returns
        -------
        W : ndarray (NRh, NRe, Nw_h, Nw_e, Nw_h, Nw_e)
        """

        W = np.zeros((self.NRh, self.NRe, self.Nw_h, self.Nw_e, self.Nw_h, self.Nw_e), dtype=np.complex128)

        # Block over k and k' to reduce peak memory while allowing BLAS-3 contractions
        for k0 in tqdm(range(0, self.Nk, chunk_k), 'Compute W loop over k0'):
            k1 = min(k0 + chunk_k, self.Nk)
            L_blk = self._L[k0:k1]  # (K, Nme, Ntc)
            ph_h_k = self.phase_h[k0:k1]  # (K, NRh) = e^{-i (k−q)·ΔRh}
            ph_e_k = self.phase_e[k0:k1]  # (K, NRe) = e^{-i k·ΔRe}

            for kp0 in range(0, self.Nk, chunk_kp):
                kp1 = min(kp0 + chunk_kp, self.Nk)
                R_blk = self._R[kp0:kp1]  # (Kp, Nme, Ntc)
                ph_h_kp = self.phase_h[kp0:kp1]  # (Kp, NRh)
                ph_e_kp = self.phase_e[kp0:kp1]  # (Kp, NRe)

                # Kernel block (K, Kp, Ntc, Ntc)
                K_blk = self.K_band[k0:k1, kp0:kp1]

                # Contract over transitions: for each k in K-block and kp in Kp-block
                # tmp[k, kp, Nme, Nme] = L[k] @ K[k,kp] @ R[kp]^H
                tmp = np.einsum(
                    'aix,abxy,byj->abij',
                    L_blk, K_blk, np.conjugate(R_blk).swapaxes(1, 2),
                    optimize=True,
                )  # (K, Kp, Nme, Nme) where the third operand is R^H

                # Apply phase factors for inverse FT and accumulate into ΔR indices
                for irh in range(self.NRh):
                    fh_k_neg = ph_h_k[:, irh]                 # e^{-i (k−q)·ΔRh}
                    fh_kp_pos = ph_h_kp[:, irh]               # e^{-i (k'−q)·ΔRh} (use negative exponent on k')
                    fh_kp_neg = ph_h_kp[:, irh]               # e^{-i (k'−q)·ΔRh}
                    for ire in range(self.NRe):
                        fe_k_neg = ph_e_k[:, ire]                 # e^{-i k·ΔRe}
                        fe_kp_pos = np.conjugate(ph_e_kp[:, ire]) # e^{+i k'·ΔRe}
                        fe_kp_neg = ph_e_kp[:, ire]               # e^{-i k'·ΔRe}

                        w_k = (fh_k_neg * fe_k_neg).astype(np.complex128)      # (K,)
                        w_kp = (fh_kp_pos * fe_kp_pos).astype(np.complex128)    # (Kp,)
                        acc = np.einsum('a,abij,b->ij', w_k, tmp, w_kp, optimize=True)

                        W[irh, ire] += acc.reshape(self.Nw_h, self.Nw_e, self.Nw_h, self.Nw_e)

        # Normalization
        W *= self.norm_factor

        # Project W onto the Wannier-me subspace spanned by L (handles rectangular U)
        # Apply P W P on each (ΔRh, ΔRe) block, then enforce Hermiticity for numerical stability
        P = getattr(self, 'P_me', None)
        if P is not None:
            for irh in range(self.NRh):
                for ire in range(self.NRe):
                    Wblk = W[irh, ire].reshape(self.Nme, self.Nme)
                    Wblk = P @ Wblk @ P
                    Wblk = 0.5 * (Wblk + Wblk.conj().T)
                    W[irh, ire] = Wblk.reshape(self.Nw_h, self.Nw_e, self.Nw_h, self.Nw_e)

        # Optional pruning
        if self.prune_tol > 0.0:
            mask = np.abs(W) < self.prune_tol
            W[mask] = 0.0

        self.W = W
        return W

    # ---------------------------------------------------------------------
    # Storage helpers
    # ---------------------------------------------------------------------
    def save_W(self, filename: str, metadata: Optional[Dict] = None) -> None:
        if h5py is None:
            raise RuntimeError("h5py not available. Please install h5py to enable saving.")
        if not hasattr(self, 'W'):
            raise RuntimeError("W not computed. Call compute_W() first.")
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('W', data=self.W, compression='gzip', compression_opts=4)
            f.create_dataset('delta_R_h', data=self.delta_R_h)
            f.create_dataset('delta_R_e', data=self.delta_R_e)
            f.create_dataset('kpoints', data=self.kpoints)
            f.create_dataset('qvec', data=self.qvec)
            f.create_dataset('U_val_shape', data=np.array(self.U_val_in.shape, dtype=int))
            f.create_dataset('U_cond_shape', data=np.array(self.U_cond_in.shape, dtype=int))
            f.attrs['norm_factor'] = self.norm_factor
            f.attrs['prune_tol'] = self.prune_tol
            if metadata:
                for k, v in metadata.items():
                    try:
                        f.attrs[k] = v
                    except TypeError:
                        # Store non-basic types as strings
                        f.attrs[k] = str(v)

    # ---------------------------------------------------------------------
    # Reconstruction: Wannier integrals -> band kernel at arbitrary k, k'
    # ---------------------------------------------------------------------
    def _k_minus_q_indices_for_grid(
        self,
        kpoints_new: np.ndarray,
        qvec_new: np.ndarray,
        *,
        decimals: int = 8,
    ) -> np.ndarray:
        """
        Build index mapping for k -> (k - q) on the SAME grid as kpoints_new.
        Works by rounding reduced coordinates to a fixed precision and using a hash map.
        """
        kpts = np.mod(np.asarray(kpoints_new, dtype=float), 1.0)
        kmq = np.mod(kpts - np.asarray(qvec_new, dtype=float), 1.0)
        Nk2 = kpts.shape[0]
        # Hash map: rounded tuple -> index
        def key(arr):
            return tuple(np.round(arr, decimals=decimals))
        lut = {key(kpts[i]): i for i in range(Nk2)}
        idx = np.empty(Nk2, dtype=int)
        for i in range(Nk2):
            tgt = key(kmq[i])
            if tgt not in lut:
                raise ValueError(
                    "Could not find (k - q) point on provided grid. Provide k_minus_q_indices_new explicitly or adjust rounding."
                )
            idx[i] = lut[tgt]
        return idx

    def assemble_kernel_band(
        self,
        kpoints_new: np.ndarray,
        U_val_new: np.ndarray,
        U_cond_new: np.ndarray,
        *,
        k_minus_q_indices_new: Optional[np.ndarray] = None,
        k_plus_q_indices_new: Optional[np.ndarray] = None,  # backward-compat alias for k−q mapping
        qvec_new: Optional[np.ndarray] = None,
        out_shape: str = 'vc',  # 'vc' -> (Nk,Nk,Nv,Nc,Nv,Nc); 'tc' -> (Nk,Nk,Ntc,Ntc)
        u_val_is_aligned: bool = False,  # if True, U_val_new[k] already corresponds to U_val(k−q)
    ) -> np.ndarray:
        """
        Reconstruct the band-basis kernel on a (possibly new) k-grid using stored W.

        Conventions:
        - Valence rotation at k - q, conduction rotation at k.
        - Phases: e^{+i (k−q)·ΔRh} e^{+i k·ΔRe}.
        - q can be overridden via qvec_new; defaults to self.qvec.
        - If u_val_is_aligned=True, U_val_new is assumed to be already aligned as U_v(k−q) indexed by k.
        """
        if not hasattr(self, 'W'):
            raise RuntimeError("W not computed/loaded.")

        q_use = self.qvec if qvec_new is None else np.asarray(qvec_new, dtype=float)

        kpts = np.asarray(kpoints_new, dtype=float)
        Nk2 = int(kpts.shape[0])
        Uv = np.asarray(U_val_new)
        Uc_in = np.asarray(U_cond_new)
        assert Uv.shape[0] == Nk2
        assert Uc_in.shape[0] == Nk2
        Nv, Nc = int(Uv.shape[2]), int(Uc_in.shape[2])
        assert Nv == self.Nv and Nc == self.Nc, "Nv/Nc mismatch"
        Ntc = Nv * Nc

        # Determine mapping for valence: k -> (k - q)
        if (not u_val_is_aligned):
            if k_minus_q_indices_new is not None:
                k_mq_idx = k_minus_q_indices_new
            elif k_plus_q_indices_new is not None:
                # Back-compat: treat as k−q mapping
                k_mq_idx = k_plus_q_indices_new
            else:
                k_mq_idx = self._k_minus_q_indices_for_grid(kpts, q_use)
        else:
            k_mq_idx = None  # not used

        # Precompute phases on new grid using q_use
        ph_h = np.exp(2j * np.pi * ((kpts - q_use) @ self.delta_R_h.T))  # (Nk2, NRh), forward bra
        ph_e = np.exp(2j * np.pi * (kpts @ self.delta_R_e.T))            # (Nk2, NRe), forward bra
        ph_h_kp = np.conjugate(ph_h)                                     # (Nk2, NRh), forward ket uses conj
        ph_e_kp = np.conjugate(ph_e)                                     # (Nk2, NRe), forward ket uses conj

        # Align U on new grid: valence at k−q, conduction at k
        if u_val_is_aligned:
            Uv_aligned = Uv
        else:
            Uv_aligned = Uv[k_mq_idx]
        Uc_aligned = Uc_in

        # Build L on new grid
        Nw_h, Nw_e = self.Nw_h, self.Nw_e
        Nme = Nw_h * Nw_e
        L_new = np.empty((Nk2, Nme, Ntc), dtype=np.complex128)
        for k in tqdm(range(Nk2), desc=f"Assembling kernel on {Nk2} points"):
            L_new[k] = np.einsum('hv,ec->hevc', Uv_aligned[k], Uc_aligned[k], optimize=True).reshape(Nme, Ntc)

        # Kernel out
        K_out = np.zeros((Nk2, Nk2, Ntc, Ntc), dtype=np.complex128)

        # Vectorized Fourier-sum with chunking to control memory
        # K[k,k'] = sum_{ΔRh,ΔRe} e^{+i (k−q)·ΔRh} e^{+i k·ΔRe} L[k]^H · W[ΔRh,ΔRe] · L[k']
        # Precompute L^H
        Lh = np.conjugate(L_new).transpose(0, 2, 1)  # (Nk2, Ntc, Nme)

        # Choose chunks for k and k'
        ck = min(Nk2, 64)
        ckp = min(Nk2, 64)

        for irh in range(self.NRh):
            ph_h_k = ph_h[:, irh]                 # (Nk2,)
            ph_h_kp_conj = ph_h_kp[:, irh]        # (Nk2,)
            for ire in range(self.NRe):
                Wblk = self.W[irh, ire].reshape(Nme, Nme)  # (Nme,Nme)
                ph_k = (ph_h_k * ph_e[:, ire]).astype(np.complex128)            # (Nk2,)
                ph_kp_conj = (ph_h_kp_conj * ph_e_kp[:, ire]).astype(np.complex128)  # (Nk2,)

                # Loop over k and k' blocks
                for k0 in range(0, Nk2, ck):
                    k1 = min(k0 + ck, Nk2)
                    # A_blk: (K, Ntc, Nme) = (ph_k * Lh) @ Wblk
                    A_blk = (ph_k[k0:k1, None, None] * Lh[k0:k1])              # (K,Ntc,Nme)
                    A_blk = np.einsum('kxm,mn->kxn', A_blk, Wblk, optimize=True)

                    for kp0 in range(0, Nk2, ckp):
                        kp1 = min(kp0 + ckp, Nk2)
                        # B_blk: (Kp, Nme, Ntc) = (ph_kp_conj * L)
                        B_blk = (ph_kp_conj[kp0:kp1, None, None] * L_new[kp0:kp1])  # (Kp,Nme,Ntc)
                        # Accumulate: (K,Ntc,Nme) x (Kp,Nme,Ntc) -> (K,Kp,Ntc,Ntc)
                        K_out[k0:k1, kp0:kp1] += np.einsum('kxm,lmy->klxy', A_blk, B_blk, optimize=True)

        if out_shape == 'vc':
            K_out = K_out.reshape(Nk2, Nk2, Nv, Nc, Nv, Nc)
        elif out_shape == 'tc':
            pass
        else:
            raise ValueError("out_shape must be 'vc' or 'tc'")
        return K_out

    # ---------------------------------------------------------------------
    # Assemble BSE H (resonant, Tamm-Dancoff) and diagonalize
    # ---------------------------------------------------------------------
    def assemble_H2P(
        self,
        eps_k: np.ndarray,             # (Nk, Nv+Nc) band energies aligned with U, or dict with 'val','cond'
        *,
        eps_is_split: bool = False,    # if True, eps_k is a dict {'val': (Nk,Nv), 'cond': (Nk,Nc)}
        out_eig: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build resonant BSE Hamiltonian H2P at q (Tamm-Dancoff) on original k-grid and diagonalize.

        Returns
        -------
        H2P : ndarray (Nk*Nv*Nc, Nk*Nv*Nc)
        eigvals, eigvecs : if out_eig True
        """
        Nk = self.Nk
        Nv, Nc = self.Nv, self.Nc
        Ntc = Nv * Nc
        dim = Nk * Ntc

        # Diagonal single-particle term ΔE_{(v,c,k)}
        if eps_is_split:
            ev = np.asarray(eps_k['val'])  # (Nk,Nv)
            ec = np.asarray(eps_k['cond'])  # (Nk,Nc)
        else:
            eps = np.asarray(eps_k)
            assert eps.shape[1] >= (Nv + Nc)
            ev = eps[:, :Nv]
            ec = eps[:, Nv:Nv + Nc]
        # Build ΔE flat: conduction at k, valence at k−q
        if self.k_minus_q is None:
            k_mq_idx = np.arange(Nk)
        else:
            k_mq_idx = self.k_minus_q
        dE = (ec[:, None, :] - ev[k_mq_idx][:, :, None]).reshape(Nk * Nv * Nc)

        # Interaction term on original grid using W and original U
        K_band_orig = self.assemble_kernel_band(
            self.kpoints,
            self.U_val_in,
            self.U_cond_in,
            k_minus_q_indices_new=self.k_minus_q,
            out_shape='tc',
            u_val_is_aligned=self.U_val_aligned,
        )

        # Build H2P = diag(dE) + K (occupation factors can be introduced outside if needed)
        H2P = np.zeros((dim, dim), dtype=np.complex128)
        # Place diagonal
        idx = np.arange(dim)
        H2P[idx, idx] = dE
        # Add interaction
        H2P += K_band_orig.reshape(Nk * Ntc, Nk * Ntc)

        if not out_eig:
            return H2P, None, None
        # Diagonalize Hermitian matrix
        evals, evecs = np.linalg.eigh(H2P)
        return H2P, evals, evecs

    # ---------------------------------------------------------------------
    # Validation utilities
    # ---------------------------------------------------------------------
    def validate_reconstruction(
        self,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        verbose: bool = True,
        u_val_is_aligned_override: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Reconstruct K on the original grid and compare with input K_band.
        Also report per-(k,k') max error, hermiticity residual, and try a phase-only check with identity U.
        If u_val_is_aligned_override is provided, use it instead of self.U_val_aligned during reconstruction.
        """
        u_align = False if u_val_is_aligned_override is None else bool(u_val_is_aligned_override)
        K_rec = self.assemble_kernel_band(
            self.kpoints,
            self.U_val_in,
            self.U_cond_in,
            k_minus_q_indices_new=self.k_minus_q,
            out_shape='tc',
            u_val_is_aligned=u_align,
        )
        K_ref = self.K_band
        diff = K_rec - K_ref
        num = np.linalg.norm(diff.ravel())
        den = np.linalg.norm(K_ref.ravel()) + 1e-30
        rel = num / den
        ok = np.allclose(K_rec, K_ref, atol=atol, rtol=rtol)

        def herm_residual(K: np.ndarray) -> float:
            # Hermiticity: K[k,k'] should equal K[k',k]^H
            K_swap = np.swapaxes(np.swapaxes(K, 0, 1), 2, 3).conj()
            return float(np.linalg.norm((K - K_swap).ravel()) / (np.linalg.norm(K.ravel()) + 1e-30))

        if verbose:
            print(f"Reconstruction relative error: {rel:.3e}; allclose={ok}; u_val_is_aligned_used={u_align}")
            print(f"Hermiticity residual (ref): {herm_residual(K_ref):.3e}; (rec): {herm_residual(K_rec):.3e}")
            # Per-(k,k') diagnostics
            Nk = self.Nk
            per = np.zeros((Nk, Nk))
            for k in range(Nk):
                for kp in range(Nk):
                    d = diff[k, kp]
                    per[k, kp] = np.linalg.norm(d)
            kmx, kpmx = divmod(per.argmax(), per.shape[1])
            print(f"Max-block error at (k={kmx}, k'={kpmx}): {per[kmx,kpmx]:.3e}")
        
        # Phase-only diagnostic: set U to identity shape-matched if possible
        try:
            Nv, Nc = self.Nv, self.Nc
            Uv_id = np.zeros_like(self.U_val_in)
            Uc_id = np.zeros_like(self.U_cond_in)
            for k in range(self.Nk):
                Uv_id[k, :Nv, :Nv] = np.eye(Nv)
                Uc_id[k, :Nc, :Nc] = np.eye(Nc)
            K_phase = self.assemble_kernel_band(
                self.kpoints, Uv_id, Uc_id,
                k_minus_q_indices_new=self.k_minus_q,
                out_shape='tc', u_val_is_aligned=u_align,
            )
            rel_phase = np.linalg.norm((K_phase - self.K_band).ravel()) / (np.linalg.norm(self.K_band.ravel()) + 1e-30)
            if verbose:
                print(f"Phase-only relative error (identity U): {rel_phase:.3e}")
        except Exception as e:
            if verbose:
                print(f"Phase-only diagnostic skipped: {e}")
        
        return {
            'rel_error': float(rel),
            'ok': float(bool(ok)),
        }


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def build_delta_R_list(R_list: np.ndarray, R_cut: Optional[float] = None, lattice: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive ΔRh and ΔRe candidate shells from a list of lattice vectors R_list (integer 3-vectors).
    If R_cut is provided, keep only those within |R| <= R_cut (in Cartesian using lattice if provided or L2 in reduced).
    """
    R = np.asarray(R_list, dtype=int)
    if R_cut is not None:
        if lattice is not None:
            R_cart = R @ lattice  # (NR,3)
            keep = np.linalg.norm(R_cart, axis=1) <= R_cut
        else:
            keep = np.linalg.norm(R, axis=1) <= R_cut
        R = R[keep]
    # For now use the same list for ΔRh and ΔRe
    return R.copy(), R.copy()