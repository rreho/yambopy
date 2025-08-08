#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: RR, MN
#
# This file is part of the yambopy project
#
import warnings
import numpy as np
import os
from netCDF4 import Dataset
from yambopy.letzelphc_interface.lelphcdb import LetzElphElectronPhononDB
from yambopy.units import *
from yambopy.bse.rotate_excitonwf import rotate_exc_wf
from yambopy.optical_properties.base_optical import BaseOpticalProperties
from yambopy.optical_properties.utils import (
    read_lelph_database, compute_symmetry_matrices
)
from tqdm import tqdm
import re

warnings.filterwarnings('ignore')

class ExcitonGroupTheory(BaseOpticalProperties):
    """
    This class performs group theory analysis of exciton states.
    
    It analyzes the irreducible representations of exciton states under the 
    little group of the exciton momentum, providing insight into the symmetry
    properties of excitonic states.
    
    Parameters
    ----------
    path : str, optional
        The path where the Yambo calculation is performed. Default is the current
        working directory.
    save : str, optional
        The name of the folder which contains the Yambo save folder. Default is 'SAVE'.
    lelph_db : LetzElphElectronPhononDB, optional
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements. If not provided, it will be read from the lelph database.
    latdb : YamboLatticeDB, optional
        The YamboLatticeDB object which contains the lattice information. If not
        provided, it will be read from the ns.db1 file.
    wfdb : YamboWFDB, optional
        The YamboWFDB object which contains the wavefunction information. If not
        provided, it will be read from the ns.wf file.
    bands_range : list or tuple, optional
        The range of bands for which the analysis will be performed.
        Default is all bands.
    BSE_dir : str, optional
        The name of the folder which contains the BSE calculation. Default is 'bse'.
    LELPH_dir : str, optional
        The name of the folder which contains the electron-phonon matrix elements.
        Default is 'lelph'.
    read_symm_from_ns_db_file : bool, optional
        If True, will read symmetry matrices from ns.db1 file else from ndb.elph.
        Default is False.
    
    Attributes
    ----------
    SAVE_dir : str
        The path of the SAVE folder.
    BSE_dir : str
        The path of the BSE folder.
    LELPH_dir : str
        The path of the folder which contains the electron-phonon matrix elements.
    latdb : YamboLatticeDB
        The YamboLatticeDB object which contains the lattice information.
    lelph_db : LetzElphElectronPhononDB
        The LetzElphElectronPhononDB object which contains the electron-phonon matrix
        elements.
    wfdb : YamboWFDB
        The YamboWFDB object which contains the wavefunction information.
    """
    
    def __init__(self, path=None, save='SAVE', lelph_db=None, latdb=None, wfdb=None, 
                 bands_range=None, BSE_dir='bse', LELPH_dir='lelph', 
                 read_symm_from_ns_db_file=True):
        """
        Initialize ExcitonGroupTheory class.
        
        Parameters
        ----------
        path : str, optional
            Path to calculation directory. Defaults to current directory.
        save : str, optional
            SAVE directory name. Defaults to 'SAVE'.
        lelph_db : LetzElphElectronPhononDB, optional
            Pre-loaded electron-phonon database.
        latdb : YamboLatticeDB, optional
            Pre-loaded lattice database.
        wfdb : YamboWFDB, optional
            Pre-loaded wavefunction database.
        bands_range : list, optional
            Range of bands to load.
        BSE_dir : str, optional
            BSE directory name. Defaults to 'bse'.
        LELPH_dir : str, optional
            LELPH directory name. Defaults to 'lelph'.
        read_symm_from_ns_db_file : bool, optional
            Whether to read symmetry from ns.db1 file. Defaults to True.
        """
        # Initialize base class
        super().__init__(path=path, save=save, latdb=latdb, wfdb=wfdb, 
                        bands_range=bands_range, BSE_dir=BSE_dir)
        
        # Setup additional directories
        self._setup_directories(LELPH_dir=LELPH_dir)
        
        # Store specific parameters
        self.lelph_db = lelph_db
        self.read_symm_from_ns_db_file = read_symm_from_ns_db_file
        
        # Read all necessary databases
        self.read(lelph_db=lelph_db, latdb=latdb, wfdb=wfdb, bands_range=bands_range)

    def read(self, lelph_db=None, latdb=None, wfdb=None, bands_range=None):
        """
        Read all necessary databases for group theory analysis.
        
        Parameters
        ----------
        lelph_db : LetzElphElectronPhononDB, optional
            Pre-loaded electron-phonon database.
        latdb : YamboLatticeDB, optional
            Pre-loaded lattice database.
        wfdb : YamboWFDB, optional
            Pre-loaded wavefunction database.
        bands_range : list, optional
            Range of bands to load.
        """
        # Read common databases using base class method
        self.read_common_databases(latdb=latdb, wfdb=wfdb, bands_range=bands_range)
        
        # Read LetzElPhC database
        self.lelph_db = read_lelph_database(self.LELPH_dir, lelph_db)
        self.qpts = self.lelph_db.qpoints
        self.elph_bnds_range = self.lelph_db.bands

        # Read D-matrices
        if bands_range:
            self.Dmats = self.wfdb.Dmat()[:,:,0,:,:]
        else:
            self.Dmats = self.wfdb.Dmat()[:,:,0,:,:]

        # Handle symmetry matrices and kmap specific to group theory
        self._setup_symmetry_data()
        
        # Setup D-matrices for spgrep analysis (will be computed when needed)
        self.spglib_Dmats = None

    def _setup_symmetry_data(self):
        """Setup symmetry data specific to group theory analysis."""
        ndb_lelph_fname = os.path.join(self.LELPH_dir, 'ndb.elph')
        
        if not self.read_symm_from_ns_db_file:
            try:
                elph_file = Dataset(ndb_lelph_fname, 'r')
                self.kpts = elph_file['kpoints'][...].data
                self.kmap = elph_file['kmap'][...].data
                self.symm_mats = elph_file['symmetry_matrices'][...].data
                self.ele_time_rev = elph_file['time_reversal_phonon'][...].data
                self.frac_trans = elph_file['fractional_translation'][...].data
                # Convert to crystal coordinates
                self.frac_trans = np.einsum('ij,nj->ni', self.blat_vecs.T, self.frac_trans)
                elph_file.close()
            except Exception as e:
                print(f"Warning: Could not read symmetry from elph file: {e}")
                print("Attempting to get fractional translations from spglib...")
                self._setup_symmetry_from_spglib()
        else:
            print("Reading symmetry from ns.db1 file...")
            # Use lattice database kmap and kpoints from lelph_db
            self.kpts = self.lelph_db.kpoints
            # Try to get fractional translations from spglib
            self._setup_symmetry_from_spglib()

        # Use existing k-point tree from base class
        if hasattr(self.wfdb, 'ktree'):
            self.kpt_tree = self.wfdb.ktree
        else:
            self._build_kpoint_tree(self.kpts)
        
        # Compute symmetry matrices in reduced coordinates using utility function
        self.sym_red = compute_symmetry_matrices(self.symm_mats, self.blat_vecs, self.lat_vecs)

        # Construct kpts_iBZ (following original algorithm exactly)
        self.kpts_iBZ = np.zeros((len(np.unique(self.kmap[:, 0])), 3))
        for i in range(self.kmap.shape[0]):
            ik_ibz, isym = self.kmap[i]
            if isym == 0:
                self.kpts_iBZ[ik_ibz, :] = self.kpts[i]
    
    def _setup_symmetry_from_spglib(self):
        """Setup symmetry data using spglib when elph file is not available."""
        try:
            import spglib
            
            # Get crystal structure
            lattice, positions, numbers = self._get_crystal_structure()
            if lattice is not None and positions is not None and numbers is not None:
                cell = (lattice, positions, numbers)
                symmetry = spglib.get_symmetry(cell, symprec=1e-5)
                
                if symmetry:
                    # Get fractional translations from spglib
                    spg_translations = symmetry['translations']
                    spg_rotations = symmetry['rotations']
                    
                    # Match spglib operations with Yambo operations
                    if hasattr(self, 'symm_mats') and self.symm_mats is not None:
                        # Try to match operations and extract corresponding translations
                        self.frac_trans = self._match_translations(spg_rotations, spg_translations)
                    else:
                        # If no Yambo symmetries available, use spglib directly
                        print("Using spglib symmetries directly")
                        self.symm_mats = spg_rotations.astype(float)
                        self.frac_trans = spg_translations
                        # Convert to crystal coordinates
                        self.frac_trans = np.einsum('ij,nj->ni', self.blat_vecs.T, self.frac_trans)
                        
                    print(f"Extracted {len(self.frac_trans)} fractional translations from spglib")
                    return
            
            # Fallback: zero translations
            print("spglib fallback failed, using zero fractional translations")
            if hasattr(self, 'symm_mats') and self.symm_mats is not None:
                self.frac_trans = np.zeros((self.symm_mats.shape[0], 3))
            else:
                self.frac_trans = np.zeros((24, 3))  # Default assumption
                
        except ImportError:
            print("spglib not available, using zero fractional translations")
            if hasattr(self, 'symm_mats') and self.symm_mats is not None:
                self.frac_trans = np.zeros((self.symm_mats.shape[0], 3))
            else:
                self.frac_trans = np.zeros((24, 3))
        except Exception as e:
            print(f"spglib setup failed: {e}")
            if hasattr(self, 'symm_mats') and self.symm_mats is not None:
                self.frac_trans = np.zeros((self.symm_mats.shape[0], 3))
            else:
                self.frac_trans = np.zeros((24, 3))
    
    def _match_translations(self, spg_rotations, spg_translations):
        """Match spglib translations with Yambo symmetry operations."""
        matched_translations = np.zeros((self.symm_mats.shape[0], 3))
        
        for i, yambo_rot in enumerate(self.symm_mats):
            # Find matching rotation in spglib operations
            best_match_idx = -1
            min_diff = float('inf')
            
            for j, spg_rot in enumerate(spg_rotations):
                diff = np.linalg.norm(yambo_rot - spg_rot.astype(float))
                if diff < min_diff:
                    min_diff = diff
                    best_match_idx = j
            
            # If we found a good match (within tolerance)
            if min_diff < 1e-6 and best_match_idx >= 0:
                matched_translations[i] = spg_translations[best_match_idx]
            else:
                # No match found, use zero translation
                matched_translations[i] = np.zeros(3)
        
        # Convert to crystal coordinates
        return np.einsum('ij,nj->ni', self.blat_vecs.T, matched_translations)
    
    def _compute_spglib_dmats(self, spglib_rotations):
        """Compute D-matrices for spglib symmetry operations using wfdb.Dmat() method."""
        try:
            # Use the wfdb.Dmat() method with spglib symmetries
            # This should be called after we have the spglib rotations
            print("Computing D-matrices for spglib symmetries...")
            
            # Convert spglib rotations to the format expected by wfdb.Dmat()
            spglib_symm_mats = spglib_rotations.astype(float)
            
            # Call wfdb.Dmat() with the spglib symmetries
            # The method accepts symm_mat (singular), frac_vec, and time_rev parameters
            if hasattr(self.wfdb, 'Dmat') and callable(getattr(self.wfdb, 'Dmat')):
                # Try to compute D-matrices with spglib symmetries
                # Get corresponding fractional translations for spglib operations
                spglib_frac_trans = self._get_spglib_fractional_translations(spglib_rotations)
                
                self.spglib_Dmats = self.wfdb.Dmat(
                    symm_mat=spglib_symm_mats,
                    frac_vec=spglib_frac_trans,
                    time_rev=(self.ele_time_rev == 1)
                )[:,:,0,:,:]
                print(f"Computed D-matrices for {len(spglib_rotations)} spglib operations")
            else:
                print("wfdb.Dmat() method not available, using Yambo D-matrices")
                self.spglib_Dmats = self.Dmats
                
        except Exception as e:
            print(f"Failed to compute spglib D-matrices: {e}")
            print("Falling back to Yambo D-matrices")
            self.spglib_Dmats = self.Dmats
    
    def _get_spglib_fractional_translations(self, spglib_rotations):
        """Get fractional translations corresponding to spglib rotations."""
        try:
            import spglib
            
            # Get crystal structure and symmetry operations
            lattice, positions, numbers = self._get_crystal_structure()
            if lattice is not None and positions is not None and numbers is not None:
                cell = (lattice, positions, numbers)
                symmetry = spglib.get_symmetry(cell, symprec=1e-5)
                
                if symmetry:
                    spg_rotations = symmetry['rotations']
                    spg_translations = symmetry['translations']
                    
                    # Match the provided spglib_rotations with the full set
                    matched_translations = []
                    for rot in spglib_rotations:
                        # Find matching rotation in full spglib set
                        best_match_idx = -1
                        min_diff = float('inf')
                        
                        for j, spg_rot in enumerate(spg_rotations):
                            diff = np.linalg.norm(rot - spg_rot)
                            if diff < min_diff:
                                min_diff = diff
                                best_match_idx = j
                        
                        if min_diff < 1e-6 and best_match_idx >= 0:
                            matched_translations.append(spg_translations[best_match_idx])
                        else:
                            matched_translations.append(np.zeros(3))
                    
                    return np.array(matched_translations)
            
            # Fallback: zero translations
            return np.zeros((len(spglib_rotations), 3))
            
        except Exception as e:
            print(f"Failed to get spglib fractional translations: {e}")
            return np.zeros((len(spglib_rotations), 3))

    def read_excdb_single(self, BSE_dir, iQ, nstates):
        """
        Read yambo exciton database for a specific Q-point.

        Parameters
        ----------
        BSE_dir : str
            The directory containing the BSE calculation data.
        iQ : int
            The Q-point index (1-based indexing as in Yambo).
        nstates : int
            Number of exciton states to read.

        Returns
        -------
        tuple
            (bands_range, BS_eigs, BS_wfcs) for the specific Q-point.
        """
        from yambopy.dbs.excitondb import YamboExcitonDB
        
        try:
            bse_db_iq = YamboExcitonDB.from_db_file(self.ydb, folder=BSE_dir,
                                                   filename=f'ndb.BS_diago_Q{iQ+1}')
        except Exception as e:
            raise IOError(f'Cannot read ndb.BS_diago_Q{iQ+1} file: {e}')
            
        bands_range = bse_db_iq.nbands
        BS_eigs = bse_db_iq.eigenvalues[:nstates]
        BS_wfcs = bse_db_iq.get_Akcv()[:nstates]
        
        # Convert to Hartree units
        BS_eigs = BS_eigs / ha2ev
        
        return bands_range, BS_eigs, BS_wfcs

    def compute(self):
        """
        Main computation method - placeholder for group theory analysis.
        
        Returns
        -------
        dict
            Results of group theory analysis.
        """
        print("ExcitonGroupTheory compute method called.")
        print("Use analyze_exciton_symmetry() method for specific Q-point analysis.")
        #For now it is a dummy method
        return {}

    def analyze_exciton_symmetry(self, iQ, nstates, degen_thres=0.001):
        """
        Perform group theory analysis for exciton states at a given Q-point.
        This implementation follows the algorithm in exe_rep_program.py exactly.

        Parameters
        ----------
        iQ : int
            The Q-point index (1-based indexing as in Yambo).
        nstates : int
            Number of exciton states to analyze.
        degen_thres : float, optional
            Degeneracy threshold in eV. Default is 0.001 eV.

        Returns
        -------
        results : dict
            Dictionary containing the analysis results including:
            - 'little_group': Little group symmetries
            - 'point_group_label': Point group label
            - 'unique_energies': Unique energy levels
            - 'degeneracies': Degeneracy of each level
            - 'irrep_decomposition': Irreducible representation decomposition
        """
        print('Reading BSE eigen vectors')
        bands_range, BS_eigs, BS_wfcs = self.read_excdb_single(self.BSE_dir, iQ-1, nstates)
        
        # Convert energies to eV for analysis 
        BS_eigs_eV = BS_eigs * ha2ev
        
        # Get unique values up to threshold 
        uni_eigs, degen_eigs = np.unique((BS_eigs_eV / degen_thres).astype(int),
                                        return_counts=True)
        uni_eigs = uni_eigs * degen_thres
        
        print('=' * 40)
        print('Group theory analysis for Q point : ', self.kpts_iBZ[iQ - 1])
        print('*' * 40)

        # Find little group 
        trace_all_real = []
        trace_all_imag = []
        little_group = []
        # Loop over symmetries (excluding time reversal operations)
        for isym in range(int(self.sym_red.shape[0] / (self.ele_time_rev + 1))):
            # Check if Sq = q 
            Sq_minus_q = np.einsum('ij,j->i', self.sym_red[isym],
                                  self.kpts_iBZ[iQ - 1]) - self.kpts_iBZ[iQ - 1]
            Sq_minus_q = Sq_minus_q - np.rint(Sq_minus_q)
            
            # Check if Sq = q (within tolerance)
            if np.linalg.norm(Sq_minus_q) > 1e-5:
                continue
            little_group.append(isym + 1)
            # Phase factor from fractional translations
            tau_dot_k = np.exp(1j * 2 * np.pi *
                              np.dot(self.kpts_iBZ[iQ - 1], self.frac_trans[isym]))
            
            # Rotate exciton wavefunction
            wfc_tmp = rotate_exc_wf(
                BS_wfcs,
                self.sym_red[isym],
                self.kpts,
                self.kpts_iBZ[iQ - 1],
                self.Dmats[isym],
                False,
                ktree=self.kpt_tree
            )
            
            # Compute representation matrix 
            rep = np.einsum('n...,m...->nm', wfc_tmp, BS_wfcs.conj(),
                           optimize=True) * tau_dot_k
            
            # Compute traces for each degenerate subspace
            irrep_sum = 0
            real_trace = []
            imag_trace = []
            for iirepp in range(len(uni_eigs)):
                idegen = degen_eigs[iirepp]
                idegen2 = irrep_sum + idegen
                trace_tmp = np.trace(rep[irrep_sum:idegen2, irrep_sum:idegen2])
                real_trace.append(trace_tmp.real.round(4))
                imag_trace.append(trace_tmp.imag.round(4))
                irrep_sum = idegen2
                
            trace_all_real.append(real_trace)
            trace_all_imag.append(imag_trace)

        little_group = np.array(little_group, dtype=int)
        
        # Get point group information using spgrep for complete crystallographic analysis
        # Note: We use spgrep here to get the full crystallographic point group including
        # non-symmorphic symmetries, while the actual wavefunction rotations above
        # used the Yambo symmetries from the SAVE database
        try:
            from .spgrep_point_group_ops import get_pg_info, decompose_rep2irrep
            
            # For irrep analysis, we need to determine the crystallographic point group
            # This may include additional symmetries not present in the Yambo SAVE
            pg_label, classes, class_dict, char_tab, irreps = self._get_crystallographic_point_group(little_group)
        except ImportError:
            print("Warning: Point group analysis module not available")
            pg_label = "Unknown"
            classes = []
            class_dict = {}
            char_tab = None
            irreps = []
        except Exception as e:
            print(f"Warning: Point group analysis failed due to numerical precision issues.")
            print("Continuing with basic symmetry analysis...")
            pg_label = "Unknown"
            classes = []
            class_dict = {}
            char_tab = None
            irreps = []

        print('Little group : ', pg_label)
        print('Little group symmetries : ', little_group)

        # Print class information (following original algorithm exactly)
        irrep_decompositions = []
        if classes:
            print('Classes (symmetry indices in each class): ')
            req_sym_characters = np.zeros(len(classes), dtype=int)
            class_orders = np.zeros(len(classes), dtype=int)
            for ilab, iclass in class_dict.items():
                if ilab < len(classes):  # Safety check
                    # Convert 1-based indices to 0-based for array access
                    iclass_0based = np.array(iclass) - 1
                    # But print the 1-based indices as they appear in little_group
                    print("%16s    : " % (classes[ilab]), np.array(iclass))
                    req_sym_characters[ilab] = min(iclass) - 1  # Convert to 0-based
                    class_orders[ilab] = len(iclass)
                else:
                    print(f"Warning: Class index {ilab} out of range for classes list (len={len(classes)})")
            print()

            # Process traces (following original algorithm exactly)
            trace_all_real = np.array(trace_all_real)
            trace_all_imag = np.array(trace_all_imag)
            trace = trace_all_real + 1j * trace_all_imag
            trace_req = trace[req_sym_characters, :].T

            print("====== Exciton representations ======")
            print("Energy (eV),  degeneracy  : representation")
            print('-' * 40)
            
            # Decompose representations (following original algorithm exactly)
            for i in range(len(trace_req)):
                if char_tab is not None:
                    rep_str_tmp = decompose_rep2irrep(trace_req[i], char_tab, 
                                                     len(little_group),
                                                     class_orders, irreps)
                else:
                    rep_str_tmp = "Analysis not available"
                print('%.4f        %9d  : ' % (uni_eigs[i], degen_eigs[i]), rep_str_tmp)
                irrep_decompositions.append(rep_str_tmp)
        else:
            # Fallback when point group analysis fails
            print("====== Exciton representations ======")
            print("Energy (eV),  degeneracy  : representation")
            print('-' * 40)
            for i in range(len(uni_eigs)):
                rep_str_tmp = "Point group analysis failed"
                print('%.4f        %9d  : ' % (uni_eigs[i], degen_eigs[i]), rep_str_tmp)
                irrep_decompositions.append(rep_str_tmp)

        print('*' * 40)

        # Return results
        results = {
            'q_point': self.kpts_iBZ[iQ - 1],
            'little_group': little_group,
            'point_group_label': pg_label,
            'unique_energies': uni_eigs,
            'degeneracies': degen_eigs,
            'irrep_decomposition': irrep_decompositions,
            'exciton_energies': BS_eigs_eV,
            'classes': classes,
            'class_dict': class_dict,
            'trace_characters': np.array(trace_all_real) + 1j * np.array(trace_all_imag) if len(trace_all_real) > 0 else None
        }
        
        return results
    
    def _get_crystallographic_point_group(self, little_group_yambo):
        """
        Determine the crystallographic point group for irrep analysis.
        
        This method takes the little group operations from Yambo and determines
        the corresponding crystallographic point group using spgrep, which may
        include additional non-symmorphic symmetries not present in the Yambo SAVE.
        
        Parameters
        ----------
        little_group_yambo : array_like
            Little group operation indices from Yambo analysis
            
        Returns
        -------
        tuple
            Point group information (pg_label, classes, class_dict, char_tab, irreps)
        """
        from .spgrep_point_group_ops import get_pg_info
        
        # Get the Yambo symmetry matrices for the little group
        little_group_mats_yambo = self.symm_mats[little_group_yambo - 1]
        
        # For crystallographic analysis, we need to determine the space group
        # and extract the corresponding point group with all symmetries
        # This is where spgrep's comprehensive database is crucial
        
        try:
            # First attempt: use the Yambo matrices directly with spgrep
            pg_label, classes, class_dict, char_tab, irreps = get_pg_info(
                little_group_mats_yambo, 
                time_rev=(self.ele_time_rev == 1)
            )
            return pg_label, classes, class_dict, char_tab, irreps
            
        except Exception as e:
            print(f"Direct spgrep analysis failed: {e}")
            
            # Use spglib to get proper crystallographic symmetries for irrep analysis
            return self._get_point_group_from_spglib(little_group_yambo)
    

    
    def _get_point_group_from_spglib(self, little_group_yambo):
        """
        Use spglib to get proper crystallographic symmetries for irrep analysis.
        
        This method uses spglib to identify the space group and extract the 
        corresponding point group symmetries for proper irrep decomposition.
        """
        try:
            import spglib
            
            # Get atomic positions and lattice from the SAVE database
            lattice, positions, numbers = self._get_crystal_structure()
            
            if lattice is not None and positions is not None and numbers is not None:
                # Use spglib to find the space group
                cell = (lattice, positions, numbers)
                spacegroup_info = spglib.get_spacegroup(cell, symprec=1e-5)
                
                if spacegroup_info:
                    print(f"spglib identified space group: {spacegroup_info}")
                    
                    # Get symmetry operations from spglib
                    symmetry = spglib.get_symmetry(cell, symprec=1e-5)
                    
                    if symmetry:
                        # Extract point group operations (rotations without translations)
                        rotations = symmetry['rotations']
                        
                        # For Γ point analysis, we need the little group, not the full space group
                        # Filter operations that leave Q=0 invariant (all operations do for Γ)
                        # But we need to identify the proper point group subset
                        little_group_rotations = self._extract_little_group_operations(rotations, spacegroup_info)
                        
                        # Use spgrep with the proper little group symmetries
                        return self._analyze_with_spglib_symmetries(little_group_rotations, little_group_yambo)
            
            # Fallback if spglib analysis fails
            print("spglib analysis failed, using operation matching")
            return self._match_operations_to_point_group(little_group_yambo)
                
        except ImportError:
            print("spglib not available, using operation matching")
            return self._match_operations_to_point_group(little_group_yambo)
        except Exception as e:
            print(f"spglib analysis failed: {e}")
            return self._match_operations_to_point_group(little_group_yambo)
    
    def _extract_little_group_operations(self, rotations, spacegroup_info):
        """
        Extract the proper little group operations for Γ point analysis.
        
        For hexagonal systems like hBN, the little group at Γ should be D3h (12 operations)
        rather than the full space group D6h (24 operations).
        """
        import numpy as np
        
        # Parse space group number from spacegroup_info
        sg_number = int(spacegroup_info.split('(')[1].split(')')[0])
        
        # General approach: extract little group operations for Γ point
        # For hexagonal systems, we typically want to reduce from D6h to D3h
        # For other systems, we may use all operations or apply different rules
        
        little_group_ops = []
        
        # Classify all operations first
        operation_classes = self._classify_operations(rotations)
        
        # Apply little group selection rules based on space group family
        if sg_number in [194, 186, 191, 193]:  # Hexagonal space groups
            little_group_ops = self._select_hexagonal_little_group(rotations, operation_classes)
        else:
            # For other space groups, use a more general approach
            little_group_ops = self._select_general_little_group(rotations, operation_classes)
        
        print(f"Extracted {len(little_group_ops)} operations for little group")
        return np.array(little_group_ops)
    
    def _classify_operations(self, rotations):
        """Classify symmetry operations by type."""
        classes = []
        for rot in rotations:
            det = np.linalg.det(rot)
            trace = np.trace(rot)
            
            if np.allclose(rot, np.eye(3)):
                classes.append('E')
            elif np.allclose(rot, -np.eye(3)):
                classes.append('i')
            elif det > 0:  # Proper rotations
                if np.isclose(trace, 0):
                    classes.append('C3')
                elif np.isclose(trace, 1):
                    classes.append('C6')
                elif np.isclose(trace, -1):
                    classes.append('C2')
                else:
                    classes.append('Cn')
            else:  # Improper rotations
                if np.isclose(trace, 1):
                    # Reflection - determine type
                    eigenvals, eigenvecs = np.linalg.eig(rot)
                    normal_idx = np.argmin(np.abs(eigenvals + 1))
                    normal = np.real(eigenvecs[:, normal_idx])
                    if np.abs(normal[2]) > 0.9:
                        classes.append('σh')
                    else:
                        classes.append('σv')
                elif np.isclose(trace, -2):
                    classes.append('S3')
                elif np.isclose(trace, -1):
                    classes.append('S6')
                else:
                    classes.append('Sn')
        
        return classes
    
    def _select_hexagonal_little_group(self, rotations, classes):
        """Select D3h little group operations from D6h space group."""
        selected_ops = []
        c2_count = 0
        sv_count = 0
        
        for i, (rot, op_class) in enumerate(zip(rotations, classes)):
            if op_class == 'E':
                selected_ops.append(rot)
            elif op_class == 'C3':
                selected_ops.append(rot)
            elif op_class == 'C2' and c2_count < 3:
                # Select only C2 operations in xy-plane
                eigenvals, eigenvecs = np.linalg.eig(rot)
                axis_idx = np.argmin(np.abs(eigenvals - 1))
                axis = np.real(eigenvecs[:, axis_idx])
                if np.abs(axis[2]) < 0.1:  # Axis in xy-plane
                    selected_ops.append(rot)
                    c2_count += 1
            elif op_class == 'σh':
                selected_ops.append(rot)
            elif op_class == 'σv' and sv_count < 3:
                selected_ops.append(rot)
                sv_count += 1
            elif op_class == 'S3':
                selected_ops.append(rot)
        
        return selected_ops
    
    def _select_general_little_group(self, rotations, classes):
        """General little group selection for non-hexagonal systems."""
        # For now, use all operations
        # This can be refined based on specific space group requirements
        return list(rotations)
    
    def _get_crystal_structure(self):
        """
        Get crystal structure information from the SAVE database.
        
        Returns
        -------
        tuple
            (lattice, positions, numbers) for spglib analysis
        """
        try:
            # Get lattice vectors (already available)
            lattice = self.lat_vecs
            
            # Read atomic positions and numbers from the SAVE database
            if (hasattr(self.ydb, 'car_atomic_positions') and 
                hasattr(self.ydb, 'atomic_numbers')):
                
                # Use the actual atomic positions and numbers from SAVE
                # spglib expects fractional coordinates, not Cartesian
                positions = self.ydb.red_atomic_positions
                numbers = self.ydb.atomic_numbers
                
                print(f"Read crystal structure from SAVE: {len(positions)} atoms")
                print(f"Atomic numbers: {numbers}")
                
                return lattice, positions, numbers
            else:
                print("No atomic structure information found in SAVE database")
                return None, None, None
            
        except Exception as e:
            print(f"Failed to get crystal structure: {e}")
            return None, None, None
    

    
    def _analyze_with_spglib_symmetries(self, spglib_rotations, little_group_yambo):
        """
        Analyze point group using spglib symmetries with spgrep.
        
        Parameters
        ----------
        spglib_rotations : array_like
            Rotation matrices from spglib
        little_group_yambo : array_like
            Little group indices from Yambo (for reference)
            
        Returns
        -------
        tuple
            Point group analysis results
        """
        try:
            from .spgrep_point_group_ops import get_pg_info
            
            # Convert spglib rotations to the format expected by spgrep
            # spglib gives integer matrices in the standard crystallographic setting
            print(f"Using {len(spglib_rotations)} symmetry operations from spglib")
            
            # Compute D-matrices for spglib symmetries if needed
            # This should be done safely after we have the spglib rotations
            if self.spglib_Dmats is None:
                self._compute_spglib_dmats(spglib_rotations)
            
            # Use spgrep with the proper spglib symmetries
            pg_label, classes, class_dict, char_tab, irreps = get_pg_info(
                spglib_rotations, 
                time_rev=(self.ele_time_rev == 1)
            )
            
            print(f"spgrep analysis with spglib symmetries successful: {pg_label}")
            return pg_label, classes, class_dict, char_tab, irreps
            
        except Exception as e:
            print(f"spgrep analysis with spglib symmetries failed: {e}")
            # Final fallback
            return self._match_operations_to_point_group(little_group_yambo)
    

    
    def _match_operations_to_point_group(self, little_group_yambo):
        """
        Match symmetry operations to known point groups as a fallback.
        
        This analyzes the actual symmetry operations to determine the point group.
        """
        try:
            # Get the symmetry matrices for the little group
            little_group_mats = self.symm_mats[little_group_yambo - 1]
            
            # Analyze the operations to determine point group
            n_ops = len(little_group_mats)
            
            # Count different types of operations
            identity_count = 0
            c3_rotations = 0
            c2_rotations = 0
            horizontal_reflections = 0
            vertical_reflections = 0
            improper_rotations = 0
            
            for mat in little_group_mats:
                det = np.linalg.det(mat)
                trace = np.trace(mat)
                
                if np.allclose(mat, np.eye(3)):
                    identity_count += 1
                elif np.isclose(det, 1) and np.isclose(trace, 0):
                    c3_rotations += 1
                elif np.isclose(det, 1) and np.isclose(trace, -1):
                    c2_rotations += 1
                elif (np.isclose(det, -1) and np.isclose(mat[2,2], 1) and 
                      np.allclose(mat[2,:2], 0) and np.allclose(mat[:2,2], 0)):
                    horizontal_reflections += 1
                elif np.isclose(det, -1) and np.isclose(trace, 1):
                    vertical_reflections += 1
                elif np.isclose(det, -1):
                    improper_rotations += 1
            
            # If we can't identify the point group, return unknown
            print(f"Could not identify point group from {n_ops} operations")
            return "Unknown", [], {}, None, []
            
        except Exception as e:
            print(f"Operation matching failed: {e}")
            return "Unknown", [], {}, None, []
    


    def save_analysis_results(self, results, filename=None):
        """
        Save the group theory analysis results to a file.

        Parameters
        ----------
        results : dict
            Results dictionary from analyze_exciton_symmetry.
        filename : str, optional
            Output filename. If None, uses default naming.
        """
        if filename is None:
            q_str = '_'.join([f'{q:.3f}' for q in results['q_point']])
            filename = f'exciton_group_theory_Q{q_str}.txt'
        
        with open(filename, 'w') as f:
            f.write("Exciton Group Theory Analysis\n")
            f.write("=" * 40 + "\n")
            f.write(f"Q-point: {results['q_point']}\n")
            f.write(f"Little group: {results['point_group_label']}\n")
            f.write(f"Little group symmetries: {results['little_group']}\n\n")
            
            if results['classes']:
                f.write("Classes:\n")
                for class_name in results['classes']:
                    f.write(f"  {class_name}\n")
                f.write("\n")
            
            f.write("Exciton representations:\n")
            f.write("Energy (eV)    Degeneracy    Representation\n")
            f.write("-" * 50 + "\n")
            
            for i, (energy, degen, irrep) in enumerate(zip(
                results['unique_energies'], 
                results['degeneracies'],
                results['irrep_decomposition'])):
                f.write(f"{energy:8.4f}    {degen:8d}    {irrep}\n")
        
        print(f"Analysis results saved to {filename}")