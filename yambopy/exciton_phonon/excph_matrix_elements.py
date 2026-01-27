##
## Authors: MN (FP adapted)
##
import numpy as np
import os
from netCDF4 import Dataset
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.bse.exciton_matrix_elements import exciton_X_matelem
from yambopy.bse.rotate_excitonwf import rotate_exc_wf
from tqdm import tqdm
from yambopy.lattice import red_car

def exciton_phonon_matelem(latdb,elphdb,wfdb,Qrange=None,BSE_dir='bse',BSE_Lin_dir=None,
                           neigs=-1,dmat_mode='run',save_files=True,exph_file='Ex-ph.npy',overwrite=False,
                           save_excitons=False,save_lattice=False):
    """
    This function calculates the exciton-phonon matrix elements

    - Q is the exciton momentum
    - q is the phonon momentum
    - exc_in represent the "initial" exciton states in the scattering process (at mom. Q)
    - exc_out represents the "final" exciton states in the scattering process (at mom. Q+q)

    Parameters
    ----------
    latdb : YamboLatticeDB
        The YamboLatticeDB object which contains the lattice information.
    elphdb : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    BSE_dir : str, optional
        The name of the folder which contains the BSE calculation. Default is 'bse'.
    BSE_Lin_dir : str, optional
        The name of the folder which contains the BSE Q=0 calculation (for optical spectra). Default is BSE_dir.
    Qrange : int list or 'full', optional
        Exciton Qpoint index range [iQ_initial, iQ_final] (python counting). 
        If 'full', it covers all k-points in the FBZ. Default is [0,1] (Gamma point only).
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    dmat_mode : str, optional
        If 'save', print dmats on .npy file for faster recalculation. If 'load', load from .npy file. Else, calculate Dmats at runtime.
    save_files : bool, optional
        If True, the matrix elements will be saved in .npz file `exph_file` with metadata. Default is True.
    overwrite : bool, optional
        If False and `exph_file` is found, the matrix elements will be loaded from file. Default is False.
    save_excitons : bool, optional
        If True, save all BSE eigenvectors and energies for the full BZ in a single `excitons.nc` file. Default is False.
    save_lattice : bool, optional
        If True, save lattice and symmetry information in a single `lattice.nc` file. Default is False.
    """
    if Qrange is None: Qrange = [0,1]
    if Qrange == 'full': Qrange = [0, wfdb.nkBZ]

    # Check if we just need to load
    if exph_file.endswith('.nc'): exph_file_path = exph_file
    else: exph_file_path = exph_file if exph_file.endswith('.npz') else exph_file.replace('.npy', '.npz')

    if os.path.exists(exph_file_path) and overwrite==False:
        print(f'Loading EXCPH matrices from {exph_file_path}...')
        if exph_file_path.endswith('.nc'):
            with Dataset(exph_file_path, 'r') as f:
                G_tmp = f.variables['G'][:]
                exph_mat_loaded = G_tmp[...,0] + 1j*G_tmp[...,1]
                # If it was saved with nQ=1, G might be 4D or 5D depending on how it was saved.
                # Here we return it as is.
        else:
            data = np.load(exph_file_path)
            exph_mat_loaded = data['G']
        return exph_mat_loaded

    # Load exc dbs
    exdbs = []
    excitons_nc_path = os.path.join(BSE_dir, 'excitons.nc')
    if os.path.exists(excitons_nc_path):
        print(f'Loading excitons from {excitons_nc_path}...')
        exdbs = YamboExcitonDB.load_excitons_nc(latdb, excitons_nc_path)
    else:
        for ik in range(wfdb.nkpoints):
            filename = 'ndb.BS_diago_Q%d' % (ik+1)
            excdb = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=BSE_dir,\
                                                Load_WF=True, neigs=neigs)
            exdbs.append(excdb)
            #
            # NM : Add a sanity check to avoid a disasterous consequence
            # if the user gives wrong bse band indices.
            min_bnd_bse = np.min(excdb.unique_vbands)
            max_bnd_bse = np.max(excdb.unique_cbands)+1
            assert (wfdb.min_bnd == min_bnd_bse) and ((wfdb.min_bnd + wfdb.nbands) == max_bnd_bse), \
                print("Error: BSE bands mismatch. Given bands range : [%d, %d]. " %(
                    wfdb.min_bnd,wfdb.min_bnd + wfdb.nbands) +
                    "Bse band range found (expected) : [%d %d]" %( min_bnd_bse,max_bnd_bse))

    # get D matrices
    Dmats = save_or_load_dmat(wfdb,mode=dmat_mode,dmat_file='Dmats.npy')

    # Calculation
    print('Calculating EXCPH matrix elements...')
    exph_mat = []
    Q_points = []
    for iQ in tqdm(range(Qrange[0],Qrange[1])):
        Q_in = wfdb.kBZ[iQ]
        Q_points.append(Q_in)
        exph_mat.append( exciton_phonon_matelem_iQ(elphdb,wfdb,exdbs,Dmats,\
                                                   BSE_Lin_dir=BSE_Lin_dir,Q_in=Q_in,neigs=neigs) )
    # IO
    if len(exph_mat)<2: exph_mat = exph_mat[0] # single Q-point calculation (suppress axis)
    else:               exph_mat = np.array(exph_mat) #[nQ,nq,nmodes,nexc_in (Qexc),nexc_out (Qexc+q)]
    
    if save_excitons:
        print('Expanding and saving excitons for the full BZ to excitons.nc...')
        full_exdbs = []
        for iQ in tqdm(range(wfdb.nkBZ)):
            Qpt = wfdb.kBZ[iQ] # reduced
            #Qpt_car = red_car(Qpt, latdb.rlat)
            # Get rotated Akcv
            Ak_rot = rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats)
            
            # Identify IBZ index
            idx_BZQ = wfdb.kptBZidx(Qpt)
            iQ_iBZ = latdb.kpoints_indexes[idx_BZQ]
            ibz_db = exdbs[iQ_iBZ]
            
            # Create rotated YamboExcitonDB
            # Flatten Ak_rot back to eigenvectors format
            eigenvectors_rot = ibz_db.flatten_Akcv(Ak_rot)
            
            full_exdb = YamboExcitonDB(latdb, str(iQ+1), ibz_db.eigenvalues, 
                                       ibz_db.l_residual, ibz_db.r_residual,
                                       spin_pol=ibz_db.spin_pol, red_qpoint=Qpt,
                                       table=ibz_db.table, eigenvectors=eigenvectors_rot)
            full_exdbs.append(full_exdb)

        YamboExcitonDB.save_excitons_nc(full_exdbs, 'excitons.nc')
    
    if save_lattice:
        print('Saving lattice to lattice.nc...')
        latdb.save_nc('lattice.nc')

    if save_files:
        if exph_file.endswith('.nc'):
            exph_file_path = exph_file
            print(f'Excph coupling file saved to {exph_file_path}')
            with Dataset(exph_file_path, 'w', format='NETCDF4') as f:
                # Dimensions
                f.createDimension('complex', 2)
                f.createDimension('q_phonon', elphdb.nq)
                f.createDimension('q_coords', 3)
                f.createDimension('Q_init', len(Q_points))
                f.createDimension('Q_coords', 3)
                
                # Exph matrix elements
                # exph_mat shape: [nQ, nq, nmodes, nexc_in, nexc_out]
                if exph_mat.ndim == 5: dims_G = ['Q_init', 'q_phonon']
                else:                  dims_G = ['q_phonon']

                for i, dim in enumerate(exph_mat.shape[len(dims_G):]):
                    dim_name = f'dim_G_{i}'
                    f.createDimension(dim_name, dim)
                    dims_G.append(dim_name)
                dims_G.append('complex')
                
                G_var = f.createVariable('G', 'f8', dims_G)
                G_var[..., 0] = exph_mat.real
                G_var[..., 1] = exph_mat.imag
                
                # Q_init
                Q_init_var = f.createVariable('Q_init', 'f8', ('Q_init', 'Q_coords'))
                Q_init_var[:] = np.array(Q_points)
                
                # q_phonon
                q_ph_var = f.createVariable('q_phonon', 'f8', ('q_phonon', 'q_coords'))
                q_ph_var[:] = elphdb.qpoints
                
        else:
            exph_file_path = exph_file if exph_file.endswith('.npz') else exph_file.replace('.npy', '.npz')
            print(f'Excph coupling file saved to {exph_file_path}')
            np.savez(exph_file_path, G=exph_mat, Q_init=np.array(Q_points), q_phonon=elphdb.qpoints)
    
    return exph_mat

def exciton_phonon_matelem_iQ(elphdb,wfdb,exdbs,Dmats,BSE_Lin_dir=None,
                              Q_in=np.zeros(3),neigs=-1,dmat_mode='run'): 
    """
    This function calculates the exciton-phonon matrix element per Q 

    - Q is the exciton momentum
    - q is the phonon momentum
    - exc_in represent the "initial" exciton states in the scattering process (at mom. Q)
    - exc_out represents the "final" exciton states in the scattering process (at mom. Q+q)

    Parameters
    ----------
    elphdb : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    exdbs : YamboExcitonDB list
        List of Q+q YamboExcitonDB objects containing the BSE calculation
    BSE_Lin_dir : str, optional
        The name of the folder which contains the BSE q=0 calculation (for optical spectra). Default is exdbs[Q].
    Q_in : np.ndarray, optional
        Excitonic momentum in reduced units. Default np.array([0.0,0.0,0.0]) 
    neigs : int, optional
        Number of excitonic states included in calculation. Default is -1 (all).
    """
    latdb = wfdb.ydb
    # Determine Lkind(in)
    Ak = rotate_Akcv_Q(wfdb, exdbs, Q_in, Dmats, folder=BSE_Lin_dir)
    # Compute ex-ph
    exph_mat = []
    bse_bnds_range = [wfdb.min_bnd,wfdb.min_bnd + wfdb.nbands]
    for iq in range(elphdb.nq):
        ph_eig, elph_mat = elphdb.read_iq(iq,bands_range=bse_bnds_range,convention='standard')
        elph_mat = elph_mat.transpose(1,0,2,4,3)
        #
        Akq = rotate_Akcv_Q(wfdb, exdbs, Q_in + elphdb.qpoints[iq], Dmats) # q+Q
        tmp_exph = exciton_X_matelem(Q_in, elphdb.qpoints[iq], \
                                     Akq, Ak, elph_mat, wfdb.kBZ, \
                                     contribution='b', diagonal_only=False, ktree=wfdb.ktree)
        exph_mat.append(tmp_exph)

    ## 0.5 for Ry to Ha
    exph_mat = 0.5 * np.array(exph_mat).transpose(0,1,3,2) #[nq,nmodes,nexc_in (Qexc),nexc_out (Qexc+q)]

    return exph_mat

def save_or_load_dmat(wfdb, mode='run', dmat_file='Dmats.npy'):
    """
    Save or load Dmats to/from .npy file `dmat_file` for faster recalculation.

     If mode=='save', print dmats on .npy file for faster recalculation. 
     If mode=='load', load from .npy file. 
     Else, calculate Dmats at runtime.
    """
    if dmat_file[-4:]!='.npy': dmat_file = dmat_file+'.npy'
    if mode=='save':
        print('Saving D matrices...')
        Dmats = wfdb.Dmat()
        np.save(dmat_file,Dmats)
        return Dmats
    elif mode=='load': 
        print('Loading D matrices...')
        if not os.path.exists(dmat_file):
            raise FileNotFoundError(f"Cannot load '{dmat_file}' - file does not exist.")
        Dmats_loaded = np.load(dmat_file)
        return Dmats_loaded
    else:
        return wfdb.Dmat()


def rotate_Akcv_Q(wfdb, exdbs, Qpt, Dmats, folder=None):
    '''
    Qpt reduced coordinates in BZ or whatever
    '''
    latdb = wfdb.ydb
    idx_BZQ = wfdb.kptBZidx(Qpt)
    iQ_isymm = latdb.symmetry_indexes[idx_BZQ]
    iQ_iBZ = latdb.kpoints_indexes[idx_BZQ]
    trev  = (iQ_isymm >= len(latdb.sym_car) / (1 + int(np.rint(latdb.time_rev))))
    symm_mat_red = latdb.lat@latdb.sym_car[iQ_isymm]@np.linalg.inv(latdb.lat)
    exe_iQIBZ = wfdb.kpts_iBZ[iQ_iBZ]
    #
    if folder is not None:
        neigs = len(exdbs[0].eigenvalues)
        filename = 'ndb.BS_diago_Q%d' % (iQ_iBZ+1)
        excdbin = YamboExcitonDB.from_db_file(latdb,filename=filename,folder=folder,\
                                              Load_WF=True, neigs=neigs)
        AQibz = excdbin.get_Akcv()
    else : AQibz = exdbs[iQ_iBZ].get_Akcv()
    #
    AQ_rot = rotate_exc_wf(AQibz,symm_mat_red,wfdb.kBZ,exe_iQIBZ,Dmats[iQ_isymm],trev,wfdb.ktree)
    
    return AQ_rot
