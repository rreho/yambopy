# Copyright (C) 2025
# All rights reserved.
#
# This file is part of yambopy
#
# Author: Riccardo Reho
#
"""
Interface between Yambo, Wannier90 and external codes for computing
screening matrix elements in the Wannier basis.

This module implements the computation of:
1. Density matrices ρ_q^nR(r) in real/G-space
2. Bare potentials V_nm(R) 
3. Screened potentials W_nm(R)
"""

import numpy as np
import scipy.fft
from netCDF4 import Dataset
from yambopy.dbs.wfdb import YamboWFDB
from yambopy.dbs.em1sdb import YamboStaticScreeningDB
from yambopy.kpoints import build_ktree, find_kpt
from typing import Optional, Tuple, Union
import os


class WannierYamboInterface:
    """
    Main class to interface between Yambo wavefunctions and Wannier90 data.
    
    This class computes screening matrix elements in the Wannier basis by:
    - Reading Bloch wavefunctions from Yambo (u_mk in G-space)
    - Reading Wannier rotations U_mn from Wannier90
    - Computing rotated wavefunctions ũ_nk = Σ_m U_mn(k) u_mk
    - Computing density matrices ρ_q^nR(r)
    - Computing bare and screened potentials V_nm(R), W_nm(R)
    
    Attributes:
        wfdb: YamboWFDB object containing Bloch wavefunctions
        U_matrix: Wannier rotation matrices [nk, nbands, nwann]
        nwann: Number of Wannier functions
        nkpts: Number of k-points
        kmesh: K-point mesh dimensions [nk1, nk2, nk3]
    """
    
    def __init__(self, 
                 save_path: str = '.', 
                 wannier_path: str = '.',
                 seedname: str = 'wannier90',
                 bands_range: Optional[list] = None):
        """
        Initialize the interface.
        
        Args:
            save_path: Path to Yambo SAVE directory
            wannier_path: Path to Wannier90 files
            seedname: Seedname for Wannier90 files
            bands_range: Range of bands to load from Yambo [min, max)
        """
        self.save_path = save_path
        self.wannier_path = wannier_path
        self.seedname = seedname
        
        # Load Yambo wavefunctions
        print("Loading Yambo wavefunctions...")
        if bands_range is None:
            self.wfdb = YamboWFDB(save=save_path)
        else:
            self.wfdb = YamboWFDB(save=save_path, bands_range=bands_range)
        
        self.nkpts = self.wfdb.nkpoints
        self.nbands = self.wfdb.nbands
        self.ngvecs = self.wfdb.ng
        
        # K-points in crystal coordinates
        self.kpts = self.wfdb.kpts_iBZ  # IBZ k-points
        self.kpts_BZ = self.wfdb.kBZ     # Full BZ k-points
        self.ktree = self.wfdb.ktree      # KDTree for k-point searching
        
        # Lattice information
        self.lat = self.wfdb.ydb.lat
        self.alat = self.wfdb.ydb.alat
        self.rlat = self.wfdb.ydb.rlat
        
        # Will be loaded later
        self.U_matrix = None
        self.nwann = None
        self.R_vectors = None
        self.nR = None
        
        print(f"Loaded {self.nkpts} k-points (IBZ) with {self.nbands} bands")
        print(f"Max G-vectors: {self.ngvecs}")
    
    def load_U_matrix_from_chk(self):
        """
        Load Wannier U matrix from .chk file.
        
        TODO: Implement this using wannier90 chk format
        """
        raise NotImplementedError("Reading from .chk file not yet implemented")
    
    def load_U_matrix_from_umat(self):
        """
        Load Wannier U matrix from .mat files, combining disentanglement and rotation.
        
        This method loads the Wannier90 transformation matrices and combines them:
        - If u_dis.mat exists: U_bloch_to_wann = U_dis @ U  (entangled case)
        - If only u.mat exists: U_bloch_to_wann = U  (isolated manifold)
        
        The resulting matrix transforms Bloch states to Wannier functions:
        |psi_n^wann> = sum_m U_bloch_to_wann[k]_{m,n} |psi_m^Bloch(k)>
        
        """
        from yambopy.wannier.wann_io import WannierUMatrices
        
        seedname_full = os.path.join(self.wannier_path, self.seedname)
        
        # Use WannierUMatrices class to handle both u_dis.mat and u.mat properly
        print(f"Loading Wannier transformation matrices from {self.seedname}...")
        umatrices = WannierUMatrices(seedname_full)
        
        # Get the properly combined Bloch->Wannier transformation
        # This is U_dis @ U if u_dis exists, otherwise just U
        self.U_matrix = umatrices.U_bloch_to_wann()
        self.nwann = umatrices.nwan
        
        nk_umat = umatrices.nk
        nbnd_umat = umatrices.nbnd
        
        if umatrices.U_dis is not None:
            print(f"  ✓ Loaded u_dis.mat (disentanglement) and u.mat (rotation)")
            print(f"  ✓ Combined: U_bloch_to_wann = U_dis @ U")
        else:
            print(f"  ✓ Loaded u.mat only (isolated manifold)")
        
        # Consistency checks
        if nk_umat != self.nkpts:
            raise ValueError(f"k-point mismatch: U matrix has {nk_umat} k-points, "
                           f"but wfdb has {self.nkpts}")
        
        if nbnd_umat != self.nbands:
            print(f"Warning: U matrix has {nbnd_umat} bands, but wfdb has {self.nbands} bands")
            if nbnd_umat < self.nbands:
                print(f"Using only first {nbnd_umat} bands from wfdb")
                self.nbands = nbnd_umat
        
        print(f"Loaded U matrix: nk={nk_umat}, nbands={nbnd_umat}, nwann={self.nwann}")
    
    def set_R_vectors(self, R_vectors: np.ndarray):
        """
        Set the R vectors for real-space representation.
        
        Args:
            R_vectors: Array of R vectors in crystal coordinates [nR, 3]
        """
        self.R_vectors = np.array(R_vectors)
        self.nR = len(self.R_vectors)
        print(f"Set {self.nR} R vectors")
    
    def load_R_vectors_from_hr(self):
        """
        Load R vectors from Wannier90 _hr.dat file.
        """
        from yambopy.wannier.wann_io import HR
        
        print(f"Loading R vectors from {self.seedname}_hr.dat...")
        hr = HR(os.path.join(self.wannier_path, self.seedname))
        
        self.R_vectors = hr.hop  # Shape [nR, 3]
        self.nR = len(self.R_vectors)
        self.ws_deg = hr.ws_deg  # Wigner-Seitz degeneracy factors
        
        print(f"Loaded {self.nR} R vectors from HR file")
    

    
    def compute_rho_q_nR(self, 
                         q_vec: np.ndarray,
                         R_vec: np.ndarray,
                         ispin: int = 0,
                         return_gspace: bool = True,
                         use_ibz_only: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the density matrix ρ_q^nR for a given q and R.
        
        The density matrix in real space is computed as:
        ρ_q^nR(r) = (1/Nk) * exp(-iq·R) * Σ_k ũ*_nk(r) ũ_n,k+q(r)
        
        where ũ_nk are the Wannier-rotated periodic parts of Bloch wavefunctions.
        
        This implementation uses a real-space approach (Option A):
        1. Load periodic parts u_mk(G) at k and k+q from Yambo (G-space)
        2. Apply Wannier transformation: ũ_nk(G) = Σ_m U_mn(k) u_mk(G)
        3. FFT to real space: ũ_nk(r)
        4. Compute density: ρ_n(r) = ũ*_nk(r) × ũ_n,k+q(r)
        5. FFT back to G-space if requested
        
        This avoids the G-vector matching problem by working in real space.
        
        Args:
            q_vec: q-vector in crystal coordinates
            R_vec: R-vector in crystal coordinates
            ispin: Spin index
            return_gspace: If True, return in G-space; if False, return in real space
            use_ibz_only: If True, use only IBZ k-points
            
        Returns:
            If return_gspace:
                rho_G: Density in G-space [nwann, ngvecs]
                gvecs: G-vectors [ngvecs, 3]
            Else:
                rho_r: Density in real space [nwann, nx, ny, nz]
        """
        if self.U_matrix is None:
            raise ValueError("U matrix not loaded. Call load_U_matrix_from_umat() first.")
        
        # Phase factor exp(-iq·R)
        phase_qR = np.exp(-2j * np.pi * np.dot(q_vec, R_vec))
        fft_grid = self.wfdb.fft_box
        cel_vol = abs(np.linalg.det(self.wfdb.ydb.lat.T))
        
        # Initialize density accumulator
        rho_r = np.zeros((self.nwann, *fft_grid), dtype=np.complex128)
        
        # Preallocate FFT arrays (reuse for all k-points to avoid allocations)
        tmp_fft_k = np.zeros((self.nwann, self.wfdb.nspinor, *fft_grid), dtype=np.complex128)
        tmp_fft_kpq = np.zeros((self.nwann, self.wfdb.nspinor, *fft_grid), dtype=np.complex128)
        
        # Loop over k-points (vectorize where possible)
        for ik in range(self.nkpts):
            # Find k+q in BZ
            kpq_vec = self.kpts[ik] + q_vec
            ikpq_bz = find_kpt(self.ktree, kpq_vec)
            ikpq_ibz = self.wfdb.ydb.kpoints_indexes[ikpq_bz]
            
            # Load wavefunctions: [nspin, nbands, nspinor, ngvecs]
            wf_k, gvecs_k = self.wfdb.get_iBZ_wf(ik)
            wf_kpq, gvecs_kpq = self.wfdb.get_iBZ_wf(ikpq_ibz)
            #GVEC = K+Q+G-K+Q
            
            # Extract spin: [nbands, nspinor, ngvecs]
            unk_k = wf_k[ispin]
            unk_kpq = wf_kpq[ispin]
            
            # Apply Wannier transformation: ũ = Σ_m U*_mn u_m
            # Shape: [nwann, nspinor, ngvecs]
            u_tilde_k = np.einsum('mn,msi->nsi', self.U_matrix[ik].conj(), unk_k)
            u_tilde_kpq = np.einsum('mn,msi->nsi', self.U_matrix[ikpq_ibz].conj(), unk_kpq)
            
            # Batch FFT: Fill preallocated arrays (all Wannier bands at once)
            # Clear arrays
            tmp_fft_k[:] = 0
            tmp_fft_kpq[:] = 0
            
            # Vectorized G-vector index computation (much faster than loop)
            Nx_k = np.where(gvecs_k[:, 0] >= 0, gvecs_k[:, 0], gvecs_k[:, 0] + fft_grid[0])
            Ny_k = np.where(gvecs_k[:, 1] >= 0, gvecs_k[:, 1], gvecs_k[:, 1] + fft_grid[1])
            Nz_k = np.where(gvecs_k[:, 2] >= 0, gvecs_k[:, 2], gvecs_k[:, 2] + fft_grid[2])
            
            Nx_kpq = np.where(gvecs_kpq[:, 0] >= 0, gvecs_kpq[:, 0], gvecs_kpq[:, 0] + fft_grid[0])
            Ny_kpq = np.where(gvecs_kpq[:, 1] >= 0, gvecs_kpq[:, 1], gvecs_kpq[:, 1] + fft_grid[1])
            Nz_kpq = np.where(gvecs_kpq[:, 2] >= 0, gvecs_kpq[:, 2], gvecs_kpq[:, 2] + fft_grid[2])
            
            # Fill FFT arrays using advanced indexing (broadcasts over nwann, nspinor)
            tmp_fft_k[:, :, Nx_k, Ny_k, Nz_k] = u_tilde_k
            tmp_fft_kpq[:, :, Nx_kpq, Ny_kpq, Nz_kpq] = u_tilde_kpq
            
            # Batched FFT: Transform all Wannier bands + spinors with ONE FFT call
            # Shape: [nwann, nspinor, nx, ny, nz]
            u_k_r = scipy.fft.ifftn(tmp_fft_k, norm="forward", axes=(2, 3, 4)) / np.sqrt(cel_vol)
            u_kpq_r = scipy.fft.ifftn(tmp_fft_kpq, norm="forward", axes=(2, 3, 4)) / np.sqrt(cel_vol)
            # multiply by phase -iGR*u_kpq_r?
            # Sum over spinors: [nwann, nx, ny, nz]
            u_k_r_total = u_k_r.sum(axis=1)
            u_kpq_r_total = u_kpq_r.sum(axis=1)
            
            # Accumulate density: ρ(r) = ũ*(r) × ũ(r)
            rho_r += u_k_r_total.conj() * u_kpq_r_total
        
        # Normalize and apply phase
        rho_r = phase_qR * rho_r / self.nkpts
        
        if return_gspace:
            # Convert to G-space using FFT
            # Use a representative G-vector set (from first k-point)
            gvecs_ref = self.wfdb.gvecs[0]
            rho_G = self._realspace_to_gspace(rho_r, gvecs_ref)
            
            return rho_G, gvecs_ref
        else:
            return rho_r
    
    def compute_all_rho_q_R(self, 
                            q_vec: np.ndarray,
                            ispin: int = 0,
                            return_gspace: bool = True) -> np.ndarray:
        """
        Compute density matrices for all R vectors at a given q.
        
        Args:
            q_vec: q-vector in crystal coordinates
            ispin: Spin index
            return_gspace: If True, return in G-space
            
        Returns:
            rho: Density matrices [nR, nwann, ngvecs] or [nR, nwann, nx, ny, nz]
        """
        if self.R_vectors is None:
            raise ValueError("R vectors not set. Call set_R_vectors or load_R_vectors_from_hr.")
        
        if return_gspace:
            rho_all = np.zeros((self.nR, self.nwann, self.ngvecs), dtype=np.complex128)
        else:
            # Determine real-space grid size
            fft_grid = self.wfdb.fft_box
            rho_all = np.zeros((self.nR, self.nwann, *fft_grid), dtype=np.complex128)
        
        for iR, R_vec in enumerate(self.R_vectors):
            print(f"Computing rho for q-point (R={iR+1}/{self.nR})", end='\r')
            
            result = self.compute_rho_q_nR(q_vec, R_vec, ispin, return_gspace)
            
            if return_gspace:
                rho_all[iR, :, :] = result[0]
            else:
                rho_all[iR, :, :, :, :] = result
        
        print()  # New line after progress
        return rho_all
    
    def compute_all_rho_qgrid_R(self,
                                 q_grid: np.ndarray,
                                 ispin: int = 0,
                                 return_gspace: bool = True) -> np.ndarray:
        """
        Compute density matrices for a grid of q-points and all R vectors.
        
        This is the main function to compute the full ρ_q^{nR} object where:
        - q runs over a grid of q-points (typically the same as k-grid)
        - R runs over all R vectors from the Wannier Hamiltonian
        - n is the Wannier band index
        
        Args:
            q_grid: Array of q-vectors in crystal coordinates, shape [nq, 3]
            ispin: Spin index
            return_gspace: If True, return in G-space; otherwise real-space
            
        Returns:
            rho: Density matrices with shape:
                - If return_gspace: [nq, nR, nwann, ngvecs]
                - If not return_gspace: [nq, nR, nwann, nx, ny, nz]
        """
        if self.R_vectors is None:
            raise ValueError("R vectors not set. Call set_R_vectors or load_R_vectors_from_hr.")
        
        nq = len(q_grid)
        
        if return_gspace:
            rho_all = np.zeros((nq, self.nR, self.nwann, self.ngvecs), dtype=np.complex128)
        else:
            fft_grid = self.wfdb.fft_box
            rho_all = np.zeros((nq, self.nR, self.nwann, *fft_grid), dtype=np.complex128)
        
        print(f"Computing rho for {nq} q-points × {self.nR} R-vectors = {nq*self.nR} total calculations")
        print(f"Grid size: {self.nwann} Wannier bands, {self.nkpts} k-points (IBZ)")
        
        # Loop over q-points
        for iq, q_vec in enumerate(q_grid):
            print(f"\nProcessing q-point {iq+1}/{nq}: q = [{q_vec[0]:.4f}, {q_vec[1]:.4f}, {q_vec[2]:.4f}]")
            
            # Loop over R vectors
            for iR, R_vec in enumerate(self.R_vectors):
                print(f"  R = {iR+1}/{self.nR}", end='\r')
                
                result = self.compute_rho_q_nR(q_vec, R_vec, ispin, return_gspace)
                
                if return_gspace:
                    rho_all[iq, iR, :, :] = result[0]
                else:
                    rho_all[iq, iR, :, :, :, :] = result
        
        print("\n" + "="*60)
        print(f"Completed! Shape: {rho_all.shape}")
        return rho_all
    
    def get_kgrid_as_qgrid(self) -> np.ndarray:
        """
        Get the k-grid (IBZ) to use as q-grid.
        
        This is commonly used since the q-grid for screening is often
        the same as the k-grid used in the DFT/GW calculation.
        
        Returns:
            q_grid: Array of q-vectors [nkpts, 3] in crystal coordinates
        """
        return self.kpts.copy()
    
    def _gspace_to_realspace(self, rho_G: np.ndarray, gvecs: np.ndarray) -> np.ndarray:
        """
        Convert density from G-space to real space using FFT.
        
        Args:
            rho_G: Density in G-space [nwann, ngvecs]
            gvecs: G-vectors [ngvecs, 3]
            
        Returns:
            rho_r: Density in real space [nwann, nx, ny, nz]
        """
        # Use wfdb's to_real_space method
        fft_grid = self.wfdb.fft_box
        nwann = rho_G.shape[0]
        
        rho_r = np.zeros((nwann, *fft_grid), dtype=np.complex128)
        
        for n in range(nwann):
            # Convert each Wannier function's density
            rho_r[n, :, :, :] = self.wfdb.to_real_space(
                rho_G[n, :], gvecs, grid=fft_grid
            )
        
        return rho_r
    
    def _realspace_to_gspace(self, rho_r: np.ndarray, gvecs: np.ndarray) -> np.ndarray:
        """
        Convert density from real space to G-space using FFT.
        
        This is the inverse of to_real_space from wfdb.
        
        Args:
            rho_r: Density in real space [nwann, nx, ny, nz]
            gvecs: G-vectors [ngvecs, 3]
            
        Returns:
            rho_G: Density in G-space [nwann, ngvecs]
        """
        import scipy.fft
        
        fft_grid = rho_r.shape[1:]  # (nx, ny, nz)
        nwann = rho_r.shape[0]
        ngvecs = len(gvecs)
        
        # Get cell volume
        cel_vol = abs(np.linalg.det(self.wfdb.ydb.lat.T))
        
        rho_G = np.zeros((nwann, ngvecs), dtype=np.complex128)
        
        for n in range(nwann):
            # Forward FFT (inverse of ifftn)
            # to_real_space does: ifftn(..., norm="forward") / sqrt(cel_vol)
            # So we need: fftn(..., norm="forward") * sqrt(cel_vol)
            rho_fft = scipy.fft.fftn(rho_r[n], norm="forward") * np.sqrt(cel_vol)
            
            # Extract G-vector components
            Nx_vals = np.where(gvecs[:, 0] >= 0, gvecs[:, 0], gvecs[:, 0] + fft_grid[0])
            Ny_vals = np.where(gvecs[:, 1] >= 0, gvecs[:, 1], gvecs[:, 1] + fft_grid[1])
            Nz_vals = np.where(gvecs[:, 2] >= 0, gvecs[:, 2], gvecs[:, 2] + fft_grid[2])
            
            rho_G[n, :] = rho_fft[Nx_vals, Ny_vals, Nz_vals]
        
        return rho_G
    
    def load_screening_db(self, em1s_path: str = '.', filename: str = 'ndb.em1s'):
        """
        Load Yambo static screening database.
        
        Args:
            em1s_path: Path to em1s database
            filename: Name of em1s database file
        """
        print(f"Loading screening database from {em1s_path}/{filename}...")
        self.em1s_db = YamboStaticScreeningDB(
            save=self.save_path,
            em1s=em1s_path,
            filename=filename,
            do_not_read_cutoff= True
        )
        print(f"Loaded screening for {self.em1s_db.nqpoints} q-points")
    
    def find_q_in_screening_db(self, q_vec: np.ndarray, tol: float = 1e-5) -> int:
        """
        Find the index of q-point in the screening database.
        
        The q-vector is provided in crystal coordinates and compared against
        the q-points stored in the em1s database (also in crystal coordinates).
        
        Args:
            q_vec: q-vector in crystal coordinates [3]
            tol: Tolerance for matching q-points
            
        Returns:
            iq: Index of q-point in em1s database
            
        Raises:
            ValueError: If q-point not found in database
        """
        if not hasattr(self, 'em1s_db'):
            raise ValueError("Screening database not loaded. Call load_screening_db first.")
        
        kmap = np.zeros((self.wfdb.nkBZ,2), dtype=int)
        kmap[:,0]=self.wfdb.ydb.kpoints_indexes
        kmap[:,1]=self.wfdb.ydb.symmetry_indexes
        self.kmap=kmap

        iq_BZ = find_kpt(self.ktree, q_vec)
        iq_ibz, isym = self.kmap[iq_BZ]

        return iq_ibz
        # # Wrap q-vector to [0, 1)
        # q_wrapped = q_vec - np.floor(q_vec)
        
        # # Compare with all q-points in database (also in crystal coordinates)
        # for iq, q_db in enumerate(self.em1s_db.red_qpoints):
        #     q_db_wrapped = q_db - np.floor(q_db)
            
        #     # Check if they match (considering periodic boundary conditions)
        #     diff = q_wrapped - q_db_wrapped
        #     diff = diff - np.round(diff)  # Wrap to [-0.5, 0.5]
            
        #     if np.linalg.norm(diff) < tol:
        #         return iq
        
        # raise ValueError(f"q-point {q_vec} not found in screening database. "
        #                 f"Available q-points: {self.em1s_db.nqpoints}")
    
    def align_gvectors(self, gvecs_rho: np.ndarray, iq: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align G-vectors between density (from wfdb) and screening database.
        
        Both G-vectors from wfdb and em1s_db.red_gvectors are in crystal/reduced 
        coordinates (integers). The em1s database may have fewer G-vectors than wfdb
        (different cutoff), so we need to find which ones match.
        
        This function finds which G-vectors from em1s_db correspond to the
        G-vectors in gvecs_rho.
        
        Args:
            gvecs_rho: G-vectors from density in crystal coordinates [ngvecs_rho, 3]
            iq: Index of q-point in em1s database
            
        Returns:
            indices_in_em1s: Indices of matching G-vectors in em1s_db [ngvecs_rho]
            mask_valid: Boolean mask for valid matches [ngvecs_rho]
        """
        if not hasattr(self, 'em1s_db'):
            raise ValueError("Screening database not loaded. Call load_screening_db first.")
        
        # Get G-vectors from em1s in reduced coordinates
        gvecs_em1s_red = self.em1s_db.red_gvectors  # [ngvecs_em1s, 3]
        ngvecs_rho = len(gvecs_rho)
        ngvecs_em1s = len(gvecs_em1s_red)
        
        # Round to integer (they should already be integers in reduced coords)
        gvecs_rho_int = np.round(gvecs_rho).astype(int)
        gvecs_em1s_int = np.round(gvecs_em1s_red).astype(int)
        
        # Build a dictionary for fast lookup: G_vector -> index in em1s
        gvec_to_idx = {}
        for ig_em1s, gvec in enumerate(gvecs_em1s_int):
            key = tuple(gvec)
            gvec_to_idx[key] = ig_em1s
        
        # Find matches
        indices_in_em1s = np.full(ngvecs_rho, -1, dtype=int)
        mask_valid = np.zeros(ngvecs_rho, dtype=bool)
        
        for ig_rho, gvec in enumerate(gvecs_rho_int):
            key = tuple(gvec)
            if key in gvec_to_idx:
                indices_in_em1s[ig_rho] = gvec_to_idx[key]
                mask_valid[ig_rho] = True
        
        return indices_in_em1s, mask_valid
    
    def compute_bare_potential_V(self, rho_q_nR: np.ndarray, q_vec: np.ndarray, 
                                  gvecs_rho: np.ndarray) -> np.ndarray:
        """
        Compute bare Coulomb potential V_q^nR(G) from density.
        
        The bare Coulomb potential in G-space is:
            V_q^nR(G) = v(q+G) * ρ_q^nR(G)
        
        where v(q+G) = 4π/|q+G|² is the bare Coulomb interaction.
        
        For G-vectors not present in the screening database, we compute
        v(q+G) directly using the bare 3D formula.
        
        Args:
            rho_q_nR: Density in G-space [nwann, ngvecs_rho]
            q_vec: q-vector in crystal coordinates [3]
            gvecs_rho: G-vectors in crystal coordinates [ngvecs_rho, 3]
            
        Returns:
            V_q_nR: Bare potential in G-space [nwann, ngvecs_rho]
        """
        if not hasattr(self, 'em1s_db'):
            raise ValueError("Screening database not loaded. Call load_screening_db first.")
        
        # Find q-point in em1s database
        iq = self.find_q_in_screening_db(q_vec)
        
        # Align G-vectors
        indices_in_em1s, mask_valid = self.align_gvectors(gvecs_rho, iq)
        
        # Initialize bare potential v(q+G)
        nwann = rho_q_nR.shape[0]
        ngvecs_rho = len(gvecs_rho)
        v_qG = np.zeros(ngvecs_rho, dtype=np.float64)
        
        # For G-vectors in em1s database, use stored sqrt(v)
        if np.any(mask_valid):
            sqrt_v_em1s = self.em1s_db.sqrt_V[iq, indices_in_em1s[mask_valid]]
            v_qG[mask_valid] = sqrt_v_em1s ** 2
        
        # For G-vectors not in em1s, compute bare potential directly
        if np.any(~mask_valid):
            # Convert to Cartesian coordinates for |q+G| calculation
            rlat = self.em1s_db.rlat  # Reciprocal lattice vectors
            q_cart = q_vec @ rlat  # q in Cartesian (atomic units)
            gvecs_cart = gvecs_rho[~mask_valid] @ rlat  # G in Cartesian
            
            # Compute |q+G| (multiply by 2π for actual reciprocal space)
            qpG_cart = 2.0 * np.pi * (q_cart[None, :] + gvecs_cart)
            qpG_norm = np.linalg.norm(qpG_cart, axis=1)
            
            # Avoid division by zero for G=0, q=0
            qpG_norm = np.where(qpG_norm < 1e-10, 1e-10, qpG_norm)
            
            # v(q+G) = 4π/|q+G|²
            v_qG[~mask_valid] = 4.0 * np.pi / (qpG_norm ** 2)
        
        # Apply potential: V = v * rho
        V_q_nR = rho_q_nR * v_qG[None, :]
        
        return V_q_nR
    
    def compute_screened_potential_W(self, rho_q_nR: np.ndarray, q_vec: np.ndarray,
                                     gvecs_rho: np.ndarray) -> np.ndarray:
        """
        Compute screened Coulomb potential W_q^nR(G) from density.
        
        The screened potential is:
            W_q^nR(G,G') = ε^{-1}(q; G,G') * v(q+G') * ρ_q^nR(G')
        
        where ε^{-1} is the inverse dielectric function from Yambo screening.
        
        Note: This computes the HEAD component (G=0) of the screened potential.
        For full local-field effects, one would need to sum over all G' components.
        
        Args:
            rho_q_nR: Density in G-space [nwann, ngvecs_rho]
            q_vec: q-vector in crystal coordinates [3]
            gvecs_rho: G-vectors in crystal coordinates [ngvecs_rho, 3]
            
        Returns:
            W_q_nR: Screened potential in G-space [nwann, ngvecs_rho]
        """
        if not hasattr(self, 'em1s_db'):
            raise ValueError("Screening database not loaded. Call load_screening_db first.")
        
        if not hasattr(self.em1s_db, 'X'):
            raise ValueError("Screening data not loaded. em1s database fragments not found.")
        
        # Find q-point in em1s database
        iq = self.find_q_in_screening_db(q_vec)
        
        # Align G-vectors
        indices_in_em1s, mask_valid = self.align_gvectors(gvecs_rho, iq)
        
        # Get screening: X[iq] is sqrt(v)*chi*sqrt(v)
        # We need eps^{-1} = 1 + v*chi = 1 + (sqrt(v)*chi*sqrt(v)) / (sqrt(v) * sqrt(v))
        sqrt_v_em1s = self.em1s_db.sqrt_V[iq, :]  # [ngvecs_em1s]
        X_iq = self.em1s_db.X[iq, :, :]  # [ngvecs_em1s, ngvecs_em1s]
        
        # Compute true chi: chi = X / (sqrt_v[G] * sqrt_v[G'])
        # Avoid division by zero
        sqrt_v_safe = np.where(sqrt_v_em1s < 1e-10, 1e-10, sqrt_v_em1s)
        chi = X_iq / (sqrt_v_safe[:, None] * sqrt_v_safe[None, :])
        
        # Compute eps^{-1} = I + v * chi
        v_em1s = sqrt_v_em1s ** 2
        eps_inv = np.eye(len(v_em1s)) + v_em1s[:, None] * chi
        
        # Initialize screened potential
        nwann = rho_q_nR.shape[0]
        ngvecs_rho = len(gvecs_rho)
        W_q_nR = np.zeros((nwann, ngvecs_rho), dtype=np.complex128)
        
        # For G-vectors in em1s database, apply full screening
        if np.any(mask_valid):
            idx_valid = indices_in_em1s[mask_valid]
            
            # W(G) = sum_{G'} eps^{-1}(G,G') * v(G') * rho(G')
            # Extract relevant submatrix of eps_inv
            eps_inv_sub = eps_inv[np.ix_(idx_valid, idx_valid)]  # [nvalid, nvalid]
            v_sub = v_em1s[idx_valid]  # [nvalid]
            
            # For each Wannier function
            for n in range(nwann):
                rho_valid = rho_q_nR[n, mask_valid]  # [nvalid]
                v_rho = v_sub * rho_valid  # [nvalid]
                W_q_nR[n, mask_valid] = eps_inv_sub @ v_rho  # [nvalid]
        
        # For G-vectors not in em1s, use bare potential (no screening data)
        if np.any(~mask_valid):
            print(f"Warning: {np.sum(~mask_valid)} G-vectors not found in screening DB. "
                  f"Using bare potential for these components.")
            V_bare = self.compute_bare_potential_V(rho_q_nR, q_vec, gvecs_rho)
            W_q_nR[:, ~mask_valid] = V_bare[:, ~mask_valid]
        
        return W_q_nR
    
    def compute_V_nm_R(self, rho_dict: dict, normalize: bool = True) -> dict:
        """
        Compute bare Coulomb matrix elements V_nm(R) in Wannier basis.
        
        Formula:
            V_nm(R) = (1/Nq) * sum_q e^{iq·R} * sum_G ρ*_q^{nR}(G) V_q^{m0}(G)
        
        where V_q^{m0}(G) = v(q+G) * ρ_q^{m0}(G) is the bare potential.
        
        Args:
            rho_dict: Precomputed densities from compute_rho_q_nR
                      Format: rho_dict[(q_tuple, R_tuple)] = (rho_q_nR, gvecs)
                      where q_tuple and R_tuple are tuples of crystal coordinates
            normalize: If True, normalize by 1/Nq and cell volume
        
        Returns:
            V_nm_dict: Dictionary with V_nm[R_tuple] = V_nm_R array [nwann, nwann]
        """
        if not hasattr(self, 'wfdb'):
            raise ValueError("Wavefunction database not loaded.")
        
        nwann = self.nwann
        
        # Cell volume in atomic units (bohr^3)
        cell_volume = abs(np.linalg.det(self.lat.T))
        # Extract unique R vectors and q vectors
        R_vecs = sorted(set(R for (q, R) in rho_dict.keys()))
        q_vecs_all = sorted(set(q for (q, R) in rho_dict.keys()))
        nq = len(q_vecs_all)
        
        print(f"Computing V_nm(R) for {len(R_vecs)} R-vectors...")
        print(f"Using {nq} q-points with cell volume = {cell_volume:.3f} bohr^3")
        if normalize:
            print(f"Normalization: 1/(Nq * Ω) = 1/({nq} * {cell_volume:.3f}) = {1.0/(nq*cell_volume):.6e}")
        
        if nq == 1:
            print("WARNING: Using only 1 q-point. This will give:")
            print("  - Same V_nm for all R vectors (no R-dependence)")
            print("  - Unphysical values dominated by G=0 at q=Γ")
            print("  Recommendation: Use a full q-point grid (e.g., same as k-grid)")
        
        V_nm_dict = {}
        
        for R_tuple in R_vecs:
            R_vec = np.array(R_tuple, dtype=np.float64)
            print(f"  R = {R_tuple}")
            V_nm_R = np.zeros((nwann, nwann), dtype=np.complex128)
            
            # Get all q-points that have this R vector
            q_vecs = sorted([q for (q, R) in rho_dict.keys() if R == R_tuple])
            
            for iq, q_tuple in enumerate(q_vecs):
                # Get precomputed densities
                rho_q_nR, gvecs_rho = rho_dict[(q_tuple, R_tuple)]
                rho_q_m0, gvecs_m0 = rho_dict[(q_tuple, (0, 0, 0))]
                
                # Verify G-vectors match
                if not np.allclose(gvecs_rho, gvecs_m0):
                    raise ValueError(f"G-vectors mismatch for q={q_tuple}, R={R_tuple}")
                
                # Compute Fourier phase factor: e^{iq·R}
                # q is in crystal coordinates, R is in lattice units
                # Phase = 2π * q·R
                q_vec = np.array(q_tuple, dtype=np.float64)
                phase = np.exp(1j * 2.0 * np.pi * np.dot(q_vec, R_vec))
                
                # Compute bare potential V_q^{m0}(G) = v(q+G) * ρ_q^{m0}(G)
                V_q_m0 = self.compute_bare_potential_V(rho_q_m0, q_vec, gvecs_m0)
                
                # Matrix element: V_nm += e^{iq·R} * sum_G ρ*_q^{nR}(G) * V_q^{m0}(G)
                # Shape: [nwann, 1, ngvecs] × [1, nwann, ngvecs] -> sum over G
                V_nm_q = np.sum(rho_q_nR[:, None, :].conj() * V_q_m0[None, :, :], axis=2)
                V_nm_R += phase * V_nm_q
            
            # Apply normalization: 1/(Nq * Ω)
            if normalize:
                V_nm_R /= (nq * cell_volume)
            
            V_nm_dict[R_tuple] = V_nm_R
            print(f"    ✓ Summed over {len(q_vecs)} q-points")
        
        return V_nm_dict
    
    def compute_W_nm_R(self, rho_dict: dict) -> dict:
        """
        Compute screened Coulomb matrix elements W_nm(R) in Wannier basis.
        
        Formula:
            W_nm(R) = V_nm(R) + sum_q sum_G V*_q^{nR}(G) Δρ_q^{n0}(G)
        
        where the induced density response is:
            Δρ_q^{n0}(G) = sum_{G'} χ_q(G,G') V_q^{n0}(G')
        
        and χ is the bare response function from the screening database.
        
        Note: The screening correction is DIAGONAL in Wannier indices (both use n).
        
        Args:
            rho_dict: Precomputed densities from compute_rho_q_nR
                      Format: rho_dict[(q_tuple, R_tuple)] = (rho_q_nR, gvecs)
        
        Returns:
            W_nm_dict: Dictionary with W_nm[R_tuple] = W_nm_R array [nwann, nwann]
        """
        if not hasattr(self, 'wfdb'):
            raise ValueError("Wavefunction database not loaded.")
        if not hasattr(self, 'em1s_db'):
            raise ValueError("Screening database not loaded. Call load_screening_db first.")
        
        nwann = self.nwann
        
        # Cell volume for normalization
        cell_volume = abs(np.linalg.det(wfdb.ydb.lat.T))
        
        # Start with bare Coulomb (without normalization yet)
        print(f"Computing W_nm(R) with screening...")
        V_nm_dict = self.compute_V_nm_R(rho_dict, normalize=False)
        W_nm_dict = {R: V_nm.copy() for R, V_nm in V_nm_dict.items()}
        
        # Extract unique R vectors and q vectors
        R_vecs = sorted(set(R for (q, R) in rho_dict.keys()))
        q_vecs_all = sorted(set(q for (q, R) in rho_dict.keys()))
        
        print(f"Adding screening corrections for {len(q_vecs_all)} q-points...")
        
        for iq, q_tuple in enumerate(q_vecs_all):
            if iq % max(1, len(q_vecs_all) // 10) == 0:
                print(f"  q-point {iq+1}/{len(q_vecs_all)}: q={q_tuple}")
            
            q_vec = np.array(q_tuple)
            
            # Find q in screening database
            try:
                iq_em1s = self.find_q_in_screening_db(q_vec)
            except ValueError:
                print(f"    Warning: q not in screening DB, skipping")
                continue
            
            # Get density at R=0 for all Wannier functions
            if (q_tuple, (0, 0, 0)) not in rho_dict:
                print(f"    Warning: ρ_q^{{n0}} not precomputed, skipping")
                continue
            
            rho_q_n0, gvecs_n0 = rho_dict[(q_tuple, (0, 0, 0))]  # [nwann, ngvecs]
            
            # Compute bare potential V_q^{n0}(G) for each Wannier function
            V_q_n0 = self.compute_bare_potential_V(rho_q_n0, q_vec, gvecs_n0)  # [nwann, ngvecs]
            
            # Get screening χ from em1s database
            indices_in_em1s, mask_valid = self.align_gvectors(gvecs_n0, iq_em1s)
            
            if not np.any(mask_valid):
                print(f"    Warning: No G-vectors overlap with screening DB")
                continue
            
            sqrt_v_em1s = self.em1s_db.sqrt_V[iq_em1s, :]
            X_iq = self.em1s_db.X[iq_em1s, :, :]
            
            # Compute χ_q(G,G')
            sqrt_v_safe = np.where(sqrt_v_em1s < 1e-10, 1e-10, sqrt_v_em1s)
            chi_q = X_iq / (sqrt_v_safe[:, None] * sqrt_v_safe[None, :])
            
            idx_valid = indices_in_em1s[mask_valid]
            ngvecs_em1s = len(sqrt_v_em1s)
            
            # Compute induced density Δρ_q^{n0}(G) for each Wannier function n
            Delta_rho_q_n0 = np.zeros((nwann, ngvecs_em1s), dtype=np.complex128)
            
            for n in range(nwann):
                # Extract V_q^{n0} for valid G-vectors
                V_q_n0_valid = np.zeros(ngvecs_em1s, dtype=np.complex128)
                V_q_n0_valid[idx_valid] = V_q_n0[n, mask_valid]
                
                # Apply χ: Δρ_q^{n0} = χ @ V_q^{n0}
                Delta_rho_q_n0[n, :] = chi_q @ V_q_n0_valid
            
            # Now add correction for each R vector
            for R_tuple in R_vecs:
                if (q_tuple, R_tuple) not in rho_dict:
                    continue
                
                # Compute Fourier phase factor: e^{iq·R}
                R_vec = np.array(R_tuple, dtype=np.float64)
                phase = np.exp(1j * 2.0 * np.pi * np.dot(q_vec, R_vec))
                
                rho_q_nR, gvecs_nR = rho_dict[(q_tuple, R_tuple)]
                
                # Compute V_q^{nR}(G)
                V_q_nR = self.compute_bare_potential_V(rho_q_nR, q_vec, gvecs_nR)  # [nwann, ngvecs]
                
                # Extract valid components
                V_q_nR_valid = np.zeros((nwann, ngvecs_em1s), dtype=np.complex128)
                
                # Align G-vectors between nR and em1s
                indices_nR_in_em1s, mask_nR_valid = self.align_gvectors(gvecs_nR, iq_em1s)
                idx_nR_valid = indices_nR_in_em1s[mask_nR_valid]
                V_q_nR_valid[:, idx_nR_valid] = V_q_nR[:, mask_nR_valid]
                
                # Correction: e^{iq·R} * sum_G V*_q^{nR}(G) Δρ_q^{n0}(G)
                # This is DIAGONAL in n, and needs proper normalization
                for n in range(nwann):
                    correction_n = np.sum(V_q_nR_valid[n, :].conj() * Delta_rho_q_n0[n, :])
                    W_nm_dict[R_tuple][n, n] += phase * correction_n
        
        # Apply the same normalization as V_nm: 1/(Nq * Ω)
        cell_volume = abs(np.linalg.det(self.lat.T))
        nq = len(q_vecs_all)
        for R_tuple in W_nm_dict:
            W_nm_dict[R_tuple] /= (nq * cell_volume)
        
        print(f"✓ W_nm(R) computed with screening.")
        return W_nm_dict
    
    def __repr__(self):
        """WannierYamboInterface."""
        lines = []
        lines.append("=" * 60)
        lines.append("Wannier-Yambo Interface")
        lines.append("=" * 60)
        lines.append(f"K-points (IBZ): {self.nkpts}")
        lines.append(f"Bands: {self.nbands}")
        lines.append(f"Wannier functions: {self.nwann if self.nwann else 'Not loaded'}")
        lines.append(f"R vectors: {self.nR if self.nR else 'Not set'}")
        lines.append(f"Max G-vectors: {self.ngvecs}")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_rho_simple(save_path: str, 
                       wannier_path: str, 
                       seedname: str,
                       q_vec: np.ndarray,
                       bands_range: Optional[list] = None) -> Tuple:
    """
    Simple wrapper to compute rho matrices.
    
    Args:
        save_path: Path to Yambo SAVE directory
        wannier_path: Path to Wannier90 files  
        seedname: Wannier90 seedname
        q_vec: q-vector in crystal coordinates
        bands_range: Range of bands to use [min, max)
        
    Returns:
        interface: WannierYamboInterface object
        rho: Density matrices [nR, nwann, ngvecs]
    """
    # Initialize interface
    interface = WannierYamboInterface(save_path, wannier_path, seedname, bands_range)
    
    # Load U matrix
    interface.load_U_matrix_from_umat()

    # Load R vectors
    interface.load_R_vectors_from_hr()
    
    # Compute rho for all R
    rho = interface.compute_all_rho_q_R(q_vec, return_gspace=True)
    
    return interface, rho