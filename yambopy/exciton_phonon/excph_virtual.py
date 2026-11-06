##
## Authors: SB
##
import numpy as np
import os
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.bse.exciton_matrix_elements import eh_X_matelem
from yambopy.kpoints import build_ktree, find_kpt
from yambopy.exciton_phonon.excph_matrix_elements import save_or_load_dmat, rotate_Akcv_Q
from tqdm import tqdm

def exciton_phonon_virtual_matelem(latdb,elphdb,wfdb,Qrange=None,BSE_dir='bse',BSE_Lin_dir=None,
                                   neigs=-1,dmat_mode='run',save_files=True,exph_file='Ex-ph-virtual.npz',
                                   overwrite=False,diagonal_only=False,second_vertex='minus_q'):
    """
    This function calculates the virtual two-phonon exciton matrix elements

        M_{S S'}(Q,q,nu',nu) = < S,Q | dV_{-q nu'} dV_{q nu} | S',Q >
                             = sum_t < S,Q | dV_{-q nu'} | t > < t | dV_{q nu} | S',Q >

    i.e. the exciton S' at Q scatters with phonon (q,nu) into an intermediate
    electron-hole pair t=(c,v,k) at Q+q and back to Q with phonon (-q,nu').
    The sum runs over all pair states of the BSE band window (no energy
    denominator). The diagonal S=S' is the "return to the initial state" element.

    - Q is the exciton momentum
    - q is the phonon momentum of the first vertex (the second one carries -q)

    Parameters
    ----------
    latdb : YamboLatticeDB
        The YamboLatticeDB object which contains the lattice information.
    elphdb : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements. Its q-grid must contain -q for every q.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    Qrange : int list or 'full' or 'full-IBZ', optional
        Exciton Qpoint index range [iQ_initial, iQ_final] (python counting).
        If 'full', it covers all k-points in the FBZ.
        If 'full-IBZ', it covers only k-points in the IBZ.
        Default is [0,1] (Gamma point only).
    BSE_dir : str, optional
        The name of the folder which contains the BSE calculation. Default is 'bse'.
    BSE_Lin_dir : str, optional
        The name of the folder which contains the BSE Q=0 calculation. Default is BSE_dir.
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    dmat_mode : str, optional
        If 'save', print dmats on .npy file for faster recalculation. If 'load', load from .npy file. Else, calculate Dmats at runtime.
    save_files : bool, optional
        If True, the matrix elements will be saved in .npz file `exph_file` with metadata. Default is True.
    exph_file : str, optional
        Output .npz file. Default is 'Ex-ph-virtual.npz'.
    overwrite : bool, optional
        If False and `exph_file` is found, the matrix elements will be loaded from file. Default is False.
    diagonal_only : bool, optional
        If True, only keep S=S'. Default is False (full nexc x nexc block at Q).
    second_vertex : str, optional
        Operator used for the second vertex (momentum transfer -q):
        - 'minus_q' : dV_{-q nu'} read from the elph database at -q (default).
        - 'adjoint' : dV_{q nu'}^dag. Identical to 'minus_q' if the phonon gauge obeys
          e_{-q} = e_q^*, otherwise it differs by a phase/rotation within degenerate
          modes. Use this for the coherent process where the same phonon (q,nu')
          is emitted (absorbed) after (q,nu) is absorbed (emitted).

    Returns
    -------
    exph_mat : ndarray (Ha^2)
        [nQ, nq, nmodes (nu', second vertex), nmodes (nu, first vertex), nexc (S), nexc (S')]
        or [..., nexc] if diagonal_only. The nQ axis is suppressed for a single Q.
    """
    if Qrange is None: Qrange = [0,1]

    # Define indices to iterate over
    if Qrange == 'full':
        Q_indices = range(wfdb.nkBZ)
    elif Qrange == 'full-IBZ':
        Q_indices = [wfdb.kptBZidx(wfdb.kpts_iBZ[i]) for i in range(wfdb.nkpoints)]
    else:
        Q_indices = range(Qrange[0], Qrange[1])

    exph_file_path = exph_file if exph_file.endswith('.npz') else exph_file.replace('.npy', '.npz')
    if os.path.exists(exph_file_path) and overwrite==False:
        print(f'Loading virtual EXCPH matrices from {exph_file_path}...')
        return np.load(exph_file_path)['M']

    # Load exc dbs
    excitons_nc_path = os.path.join(BSE_dir, 'excitons.nc')
    if os.path.exists(excitons_nc_path):
        print(f'Loading excitons from {excitons_nc_path}...')
        exdbs = YamboExcitonDB.load_excitons_nc(latdb, excitons_nc_path)
    else:
        import glob, re
        files = glob.glob(os.path.join(BSE_dir, 'ndb.BS_diago_Q*'))
        q_indices = sorted([int(re.search(r'ndb\.BS_diago_Q(\d+)', f).group(1)) for f in files if re.search(r'ndb\.BS_diago_Q(\d+)', f)])

        exdbs = [None] * max(q_indices)
        for iq in q_indices:
            exdbs[iq-1] = YamboExcitonDB.from_db_file(latdb,filename='ndb.BS_diago_Q%d' % iq,folder=BSE_dir,\
                                                      Load_WF=True, neigs=neigs)

        min_bnd_bse = np.min(exdbs[q_indices[0]-1].unique_vbands)
        max_bnd_bse = np.max(exdbs[q_indices[0]-1].unique_cbands)+1
        assert (wfdb.min_bnd == min_bnd_bse) and ((wfdb.min_bnd + wfdb.nbands) == max_bnd_bse), \
            "Error: BSE bands mismatch. Given bands range : [%d, %d]. Bse band range found (expected) : [%d %d]" %(
                wfdb.min_bnd,wfdb.min_bnd + wfdb.nbands, min_bnd_bse,max_bnd_bse)

    # get D matrices
    Dmats = save_or_load_dmat(wfdb,mode=dmat_mode,dmat_file='Dmats.npy')

    # -q index for every q of the elph grid
    qtree = build_ktree(elphdb.qpoints)
    iq_minus = find_kpt(qtree, -elphdb.qpoints)

    # Calculation
    print('Calculating virtual EXCPH matrix elements...')
    exph_mat = []
    Q_points = []
    for iQ in Q_indices:
        Q_in = wfdb.kBZ[iQ]
        Q_points.append(Q_in)
        exph_mat.append( exciton_phonon_virtual_matelem_iQ(elphdb,wfdb,exdbs,Dmats,iq_minus,\
                                                           BSE_Lin_dir=BSE_Lin_dir,Q_in=Q_in,
                                                           diagonal_only=diagonal_only,
                                                           second_vertex=second_vertex) )
    # IO
    if len(exph_mat)<2: exph_mat = exph_mat[0] # single Q-point calculation (suppress axis)
    else:               exph_mat = np.array(exph_mat) #[nQ,nq,nmodes',nmodes,nexc,nexc]

    if save_files:
        print(f'Virtual excph coupling file saved to {exph_file_path}')
        np.savez(exph_file_path, M=exph_mat, Q_init=np.array(Q_points),
                 qpoints=elphdb.qpoints, iq_minus=iq_minus, second_vertex=second_vertex,
                 kBZ=wfdb.kBZ, kpoints_indexes=latdb.kpoints_indexes,
                 symmetry_indexes=latdb.symmetry_indexes)

    return exph_mat

def exciton_phonon_virtual_matelem_iQ(elphdb,wfdb,exdbs,Dmats,iq_minus,BSE_Lin_dir=None,
                                      Q_in=np.zeros(3),diagonal_only=False,second_vertex='minus_q'):
    """
    Virtual two-phonon exciton matrix elements for one exciton momentum Q.

    Returns [nq, nmodes (nu'), nmodes (nu), nexc (S), nexc (S')] (or [..., nexc] if diagonal_only).
    See exciton_phonon_virtual_matelem.
    """
    assert second_vertex in ['minus_q', 'adjoint'], "Allowed values for second_vertex are 'minus_q', 'adjoint'"
    Ak = rotate_Akcv_Q(wfdb, exdbs, Q_in, Dmats, folder=BSE_Lin_dir)
    bse_bnds_range = [wfdb.min_bnd,wfdb.min_bnd + wfdb.nbands]
    exph_mat = []
    for iq in tqdm(range(elphdb.nq), desc='computing phonon momentum'):
        qpt = elphdb.qpoints[iq]
        # g(q) : <k+q|dV_q|k>, (nmodes, nk, nspin, final, initial)
        elph_q = elphdb.read_iq(iq,bands_range=bse_bnds_range,convention='standard')[1].transpose(1,0,2,4,3)
        # g(-q) : <k-q|dV_-q|k>
        if second_vertex == 'minus_q':
            elph_mq = elphdb.read_iq(iq_minus[iq],bands_range=bse_bnds_range,convention='standard')[1].transpose(1,0,2,4,3)
        else: elph_mq = None
        exph_mat.append( virtual_X_matelem(Q_in, qpt, Ak, elph_q, elph_mq, wfdb.kBZ,
                                           diagonal_only=diagonal_only, ktree=wfdb.ktree) )

    ## 0.5 per vertex for Ry to Ha
    return 0.25 * np.array(exph_mat) #[nq,nmodes (nu'),nmodes (nu),nexc (S),nexc (S')]

def virtual_X_matelem(exe_kvec, O_qvec, Ak, Omn_q, Omn_mq, kpts, diagonal_only=False, ktree=None):
    """
    Compute < S,Q | O(-q) O(q) | S',Q > in the Tamm-Dancoff approximation, summing
    over all intermediate electron-hole pairs (c,v,k) at Q+q.

    Parameters
    ----------
    exe_kvec : array_like
        Exciton momentum Q in crystal coordinates.
    O_qvec : array_like
        Momentum q of the first vertex in crystal coordinates.
    Ak : array_like
        Exciton wavefunction at Q with shape (n_exe_states, 1, ns, nk, nc, nv).
    Omn_q : array_like
        (nlambda, nk, nspin, m_bnd, n_bnd) with Omn_q = < k+q, m | O(q) | k, n >.
    Omn_mq : array_like or None
        (nlambda, nk, nspin, m_bnd, n_bnd) with Omn_mq = < k-q, m | O(-q) | k, n >.
        If None, O(-q) = O(q)^dag is used for the second vertex.
    kpts : array_like
        K-points used to construct the BSE with shape (nk, 3) in crystal coordinates.
    diagonal_only : bool, optional
        If True, only S=S' is computed.
    ktree : KDtree, optional
        If None, will build internally, else use the user provided

    Returns
    -------
    ex_O_mat : ndarray
        (nlambda (second vertex), nlambda (first vertex), n_exe_states (S), n_exe_states (S'))
        or (nlambda, nlambda, n_exe_states) if diagonal_only.
    """
    if ktree is None : ktree = build_ktree(kpts)
    #
    # X = O(q) |S',Q> and Y = O(-q)^dag |S,Q>, both pair vectors at Q+q: [nlambda, nexc, ntrans]
    X = eh_X_matelem(exe_kvec, O_qvec, Ak, Omn_q, kpts, contribution='b', ktree=ktree)
    if Omn_mq is None:
        Y = X
    else:
        # adjoint of O(-q), which again transfers +q:
        # <k+q,m|O(-q)^dag|k,n> = conj(<k,n|O(-q)|k+q,m>) = conj(Omn_mq[k+q,n,m])
        idx_k_plus_q = find_kpt(ktree, kpts + O_qvec[None, :])
        Omn_adj = Omn_mq[:, idx_k_plus_q].swapaxes(-1,-2).conj()
        Y = eh_X_matelem(exe_kvec, O_qvec, Ak, Omn_adj, kpts, contribution='b', ktree=ktree)
    # contract over the intermediate pair states
    if diagonal_only: return np.einsum('ast,bst->abs', Y.conj(), X)
    else:             return np.einsum('ast,bpt->absp', Y.conj(), X)
