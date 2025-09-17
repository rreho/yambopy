# Utilities for aligning G-spheres and computing overlaps of periodic Bloch parts in G-space
#
# This module provides:
# - change_gsphere: map coefficients defined on one G-set to another G-set with zero padding
# - build_union_gsphere: construct common G-set and index maps
# - reindex_with_shift: implement c_{k+G0}(G) = c_k(G+G0) via label shift
# - compute_overlap: end-to-end overlap with optional k->k+G0 shift and symmetry
#
# Notes
# -----
# - All G-vectors are assumed to be integer Miller indices in reduced (crystal) coordinates
# - k-vectors are in reduced coordinates
# - Coefficients arrays may have arbitrary leading dimensions; the last dimension must be NG
# - Symmetry support expects the rotation in reduced coordinates and optional fractional translation
#   consistent with wfdb.apply_symm implementation

from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np

Array = np.ndarray

INVALID_G_MARK = 2147483646

# ---------- helpers ----------
def _ensure_intG(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G)
    if not np.issubdtype(G.dtype, np.integer):
        # allow floats that are integer within tolerance
        if np.allclose(G, np.rint(G), atol=1e-8):
            G = np.rint(G).astype(int)
        else:
            raise ValueError("G vectors must be integer Miller indices or integer-valued floats")
    return np.ascontiguousarray(G.astype(int))

def pack_triplet_to_int64(G: np.ndarray, shift=21) -> np.ndarray:
    """
    Pack 3 small integers into one int64 key.
    shift: number of bits for each component (tune to fit range)
    Works for moderate ranges: components must fit into (2^shift)-1.
    """
    G = np.asarray(G, dtype=np.int64)
    if G.ndim != 2 or G.shape[1] != 3:
        raise ValueError("G must be (N,3)")
    # add offset to allow negative values
    mx = np.max(np.abs(G))
    # choose offset = 2^(shift-1)
    offset = 1 << (shift - 1)
    a = (G[:,0] + offset).astype(np.int64)
    b = (G[:,1] + offset).astype(np.int64)
    c = (G[:,2] + offset).astype(np.int64)
    return (a << (2*shift)) | (b << shift) | c

def build_union_gsphere(G1, G2):
    G1 = _ensure_intG(G1)
    G2 = _ensure_intG(G2)
    # vectorized unique via packing
    keys1 = pack_triplet_to_int64(G1)
    keys2 = pack_triplet_to_int64(G2)
    all_keys = np.concatenate([keys1, keys2])
    allG = np.vstack([G1, G2])
    uniq_keys, inv_idx = np.unique(all_keys, return_index=True)
    G_union = allG[inv_idx]
    # sort lexicographically deterministic
    order = np.lexsort((G_union[:,2], G_union[:,1], G_union[:,0]))
    G_union = G_union[order]
    # build map from key->idx in union
    union_keys = pack_triplet_to_int64(G_union)
    key2idx = {int(k): i for i, k in enumerate(union_keys)}
    # map original arrays
    idx1 = np.array([key2idx.get(int(k), -1) for k in keys1], dtype=int)
    idx2 = np.array([key2idx.get(int(k), -1) for k in keys2], dtype=int)
    # But idx1, idx2 map from original -> union. The previous API returned for each G_union index the index into G1.
    # produce idx1_to_union: for each union index, index into G1 or -1
    rev1 = -np.ones(G_union.shape[0], dtype=int)
    rev2 = -np.ones(G_union.shape[0], dtype=int)
    for orig_i, u in enumerate(idx1):
        if u >= 0:
            rev1[u] = orig_i
    for orig_i, u in enumerate(idx2):
        if u >= 0:
            rev2[u] = orig_i
    return G_union, rev1, rev2


# ---------- change_gsphere vectorized ----------
def change_gsphere(G_source, c_source, G_target):
    Gs = _ensure_intG(G_source)
    Gt = _ensure_intG(G_target)
    c_source = np.asarray(c_source)
    assert c_source.shape[-1] == Gs.shape[0]
    # build maps via packed keys
    keys_src = pack_triplet_to_int64(Gs)
    keys_tgt = pack_triplet_to_int64(Gt)
    # map each target key to source index (vectorized)
    key2idx = {int(k): i for i, k in enumerate(keys_src)}
    idx_map = np.array([key2idx.get(int(k), -1) for k in keys_tgt], dtype=int)

    out_shape = c_source.shape[:-1] + (Gt.shape[0],)
    c_tgt = np.zeros(out_shape, dtype=c_source.dtype)
    present = idx_map >= 0
    if np.any(present):
        c_tgt[..., present] = c_source[..., idx_map[present]]
    return c_tgt

# ---------- reindex_with_shift with optional remap ----------
def reindex_with_shift(G, c, Gshift, remap=False):
    G = _ensure_intG(G)
    Gshift = _ensure_intG(Gshift.reshape(1,3))[0]
    if not remap:
        # purely relabel G as G - Gshift; coefficients unchanged
        return (G - Gshift), c
    # remap: produce c_shifted s.t. c_shifted[i] = c[j] with G[j] = G[i] + Gshift
    keys = pack_triplet_to_int64(G)
    key2idx = {int(k): i for i, k in enumerate(keys)}
    # target keys for every G entry: (G + Gshift)
    target_keys = pack_triplet_to_int64(G + Gshift)
    idx_map = np.array([key2idx.get(int(k), -1) for k in target_keys], dtype=int)
    c_shifted = np.zeros_like(c)
    present = idx_map >= 0
    if np.any(present):
        c_shifted[..., present] = c[..., idx_map[present]]
    return G, c_shifted

def _apply_symmetry_to_coeffs(kvec: Array,
                              G: Array,
                              c: Array,
                              sym: Dict) -> Tuple[Array, Array, Array]:
    """
    Apply symmetry (in reduced coordinates) to k, G, c.

    sym keys:
    - sym_red: (3,3) int rotation in reduced basis
    - tau_red: (3,) fractional translation in reduced basis (defaults to 0)
    - time_rev: bool (defaults to False)
    - su2: optional (2,2) for spinor rotation; if provided, c is expected to have a spinor axis before the last axis

    Returns k_sym, G_sym, c_sym
    """
    sym_red = np.asarray(sym.get('sym_red'))
    tau_red = np.asarray(sym.get('tau_red', np.zeros(3)))
    time_rev = bool(sym.get('time_rev', False))
    su2 = sym.get('su2', None)

    # Rotate k and G (consistent with wfdb.apply_symm using sym_red as integer in reduced coords)
    k_sym = sym_red.T @ kvec
    G_sym = G @ sym_red

    # Phases from translation: exp(-i 2pi (k'·tau + G'·tau))
    phase = np.exp(-1j * 2 * np.pi * (k_sym @ tau_red)) * np.exp(-1j * 2 * np.pi * (G_sym @ tau_red))

    c_sym = c.copy()
    # Optional spinor rotation
    if su2 is not None:
        su2 = np.asarray(su2, dtype=c.dtype)
        # Expect c shape (..., Nspinor, NG)
        if c_sym.ndim < 2:
            raise ValueError("Spinor SU(2) provided but coefficients lack spinor axis")
        # Apply on the last spinor axis (assume it is the penultimate axis)
        c_sym = (su2 @ c_sym.reshape((-1, c_sym.shape[-2], c_sym.shape[-1]))).reshape(c.shape)

    if time_rev:
        c_sym = c_sym.conj()

    # Apply G-dependent phase on last axis
    c_sym = c_sym * phase[..., None]
    return k_sym, G_sym, c_sym


def _infer_fft_grid_from_G(Gs: Array) -> Array:
    """Infer a minimal FFT grid (Nx,Ny,Nz) that can hold all Miller indices in Gs.
    Grid is computed as max_i - min_i + 1 per axis.
    """
    Gs = np.asarray(Gs, dtype=int)
    mn = Gs.min(axis=0)
    mx = Gs.max(axis=0)
    return (mx - mn + 1).astype(int)


def _grid_indices_for_G(G: Array, grid: Array) -> Tuple[Array, Array, Array]:
    """Convert possibly negative Miller indices to [0,N) grid indices using wrap like wfdb.to_real_space."""
    G = np.asarray(G, dtype=int)
    Nx = np.where(G[:, 0] >= 0, G[:, 0], G[:, 0] + grid[0])
    Ny = np.where(G[:, 1] >= 0, G[:, 1], G[:, 1] + grid[1])
    Nz = np.where(G[:, 2] >= 0, G[:, 2], G[:, 2] + grid[2])
    return Nx, Ny, Nz


def remap_with_box(G_source: Array, c_source: Array, G_target: Array, grid: Optional[Array] = None,
                   Gshift: Optional[Array] = None) -> Array:
    """
    Remap coefficients from G_source to values sampled at G_target (+Gshift) using a dense box index mapping.
    This mirrors the ABINIT `sphere` insertion/extraction logic without going through FFT.

    If grid is None, a minimal grid covering both G_source and (G_target+Gshift) is inferred.
    """
    G_source = np.asarray(G_source, dtype=int)
    G_target = np.asarray(G_target, dtype=int)
    c_source = np.asarray(c_source)
    assert c_source.shape[-1] == G_source.shape[0]

    if Gshift is None:
        Gt_eff = G_target
    else:
        Gt_eff = G_target + np.asarray(Gshift, dtype=int)

    allG = np.vstack([G_source, Gt_eff])
    if grid is None:
        grid = _infer_fft_grid_from_G(allG)
    grid = np.asarray(grid, dtype=int)

    # Build map from grid positions of source G to coefficient indices
    Nx_s, Ny_s, Nz_s = _grid_indices_for_G(G_source, grid)
    # Build a hash map using a 1D linear index
    lin_mul_y = grid[2]
    lin_mul_x = grid[1] * grid[2]
    lin_idx_src = Nx_s * lin_mul_x + Ny_s * lin_mul_y + Nz_s
    pos2idx = {int(p): i for i, p in enumerate(lin_idx_src)}

    # Prepare output remapped coefficients
    c_tgt = np.zeros(c_source.shape[:-1] + (G_target.shape[0],), dtype=c_source.dtype)
    Nx_t, Ny_t, Nz_t = _grid_indices_for_G(Gt_eff, grid)
    lin_idx_tgt = Nx_t * lin_mul_x + Ny_t * lin_mul_y + Nz_t

    present_mask = np.array([int(p) in pos2idx for p in lin_idx_tgt], dtype=bool)
    if np.any(present_mask):
        src_indices = np.array([pos2idx[int(p)] for p in lin_idx_tgt[present_mask]], dtype=int)
        c_tgt[..., present_mask] = c_source[..., src_indices]
    return c_tgt


def compute_overlap(kvec1: Array,
                    G1: Array,
                    c1: Array,
                    kvec2: Array,
                    G2: Array,
                    c2: Array,
                    Gshift: Optional[Array] = None,
                    sym: Optional[Dict] = None,
                    use_box: bool = False,
                    box_grid: Optional[Array] = None) -> Array:
    """
    Compute overlap <u_{n,k1} | u_{m,k2(+Gshift)}>, aligning G-spheres and handling optional symmetry and k->k+G0.

    Parameters
    ----------
    kvec1, kvec2 : (3,) reduced
    G1, G2 : (N1,3), (N2,3) int reduced
    c1, c2 : (..., N1), (..., N2) complex. Leading axes (e.g., nband, nspinor) are preserved and broadcast across the dot.
    Gshift : optional (3,) int. If given, implements extraction at G_target+Gshift.
    sym : optional dict. If provided, apply symmetry to (k2,G2,c2) before shift. See _apply_symmetry_to_coeffs.
    use_box : if True, remap onto a common dense box using remap_with_box, which mirrors ABINIT sphere logic.
    box_grid : optional (Nx,Ny,Nz) grid to use when use_box=True; if None, inferred from data.

    Returns
    -------
    overlap : array with shape equal to the broadcast of c1 and c2 leading axes (excluding last).
    """
    kvec1 = np.asarray(kvec1)
    kvec2 = np.asarray(kvec2)
    G1 = np.asarray(G1, dtype=int)
    G2 = np.asarray(G2, dtype=int)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    assert G1.shape[1] == 3 and G2.shape[1] == 3
    assert c1.shape[-1] == G1.shape[0]
    assert c2.shape[-1] == G2.shape[0]

    # Apply symmetry to (k2,G2,c2) if provided
    if sym is not None:
        kvec2, G2, c2 = _apply_symmetry_to_coeffs(kvec2, G2, c2, sym)

    # Require k1 == k2 up to a small mismatch (handled outside) after symmetry
    kdiff = kvec1 - kvec2
    G0 = np.rint(kdiff).astype(int)
    kdiff -= G0
    if np.max(np.abs(kdiff)) > 1e-8:
        lead_shape = np.broadcast_shapes(c1.shape[:-1], c2.shape[:-1])
        return np.zeros(lead_shape, dtype=np.result_type(c1, c2))

    # Align on a common set and compute dot
    if use_box:
        # Build common target as union of labels; use box remap for both operands
        Gunion, _, _ = build_union_gsphere(G1, G2)
        c1a = remap_with_box(G1, c1, Gunion, grid=box_grid)
        # For c2, apply optional Gshift during extraction
        c2a = remap_with_box(G2, c2, Gunion, grid=box_grid, Gshift=(np.asarray(Gshift, dtype=int) if Gshift is not None else None))
    else:
        # Label-based reindexing (fast path)
        if Gshift is not None:
            G2 = G2 - np.asarray(Gshift, dtype=int)
        Gunion, _, _ = build_union_gsphere(G1, G2)
        c1a = change_gsphere(G1, c1, Gunion)
        c2a = change_gsphere(G2, c2, Gunion)

    overlap = np.tensordot(c1a.conj(), c2a, axes=([-1], [-1]))
    return overlap