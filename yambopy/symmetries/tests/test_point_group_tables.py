"""
Consistency of the point-group character tables and of the class assignment in get_pg_info
for all 32 crystallographic point groups, in the standard and in a rotated orientation.

For every irrep r the operator P_r = (d_r / |G|) sum_g chi_r(g)* D(g) must be a projector on any
representation D, and sum_r P_r = 1.  Here D = polar vector + rank-2 + rank-3 tensors + axial vector,
which contains every irrep of every crystallographic point group.  A wrong character or a wrong
class assignment (e.g. two C2 classes swapped) breaks P_r^2 = P_r.
"""
import io
import contextlib
import numpy as np
import pytest
from yambopy.symmetries.point_group_ops import get_pg_info, pg_to_symels

PGS = ('C1 Ci C2 Cs C2h D2 C2v D2h C4 S4 C4h D4 C4v D2d D4h C3 S6 D3 C3v D3d '
       'C6 C3h C6h D6 C6v D3h D6h T Th O Td Oh').split()


def _rep(R):
    z = np.zeros
    return np.block([[R, z((3, 39))],
                     [z((9, 3)), np.kron(R, R), z((9, 30))],
                     [z((27, 12)), np.kron(np.kron(R, R), R), z((27, 3))],
                     [z((3, 39)), np.linalg.det(R) * R]])


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.mark.parametrize('pg', PGS)
@pytest.mark.parametrize('seed', [None, 1])
def test_character_projectors(pg, seed):
    S = np.array([s.rrep for s in pg_to_symels(pg)], float)
    U = np.eye(3) if seed is None else _rotation(seed)
    L = np.einsum('ij,njk,lk->nil', U, S, U)
    with contextlib.redirect_stdout(io.StringIO()):
        label, classes, class_dict, ctab, irreps = get_pg_info(L)
    assert label == pg
    ctab = np.atleast_2d(np.asarray(ctab))
    chi = np.zeros((len(irreps), len(L)), complex)
    for ic, ops in class_dict.items():
        chi[:, ops] = ctab[:, ic][:, None]
    D = np.array([_rep(R) for R in L])
    total = np.zeros((42, 42), complex)
    for r in range(len(irreps)):
        # every row is a genuine irrep: sum_g |chi|^2 = |G|
        assert np.isclose(np.sum(np.abs(chi[r])**2), len(L)), irreps[r]
        P = chi[r, 0].real / len(L) * np.einsum('g,gij->ij', chi[r].conj(), D)
        assert np.allclose(P @ P, P, atol=1e-8), irreps[r]
        total += P
    assert np.allclose(total, np.eye(42), atol=1e-8)
