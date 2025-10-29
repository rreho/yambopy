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
            filename=filename
        )
        print(f"Loaded screening for {self.em1s_db.nqpoints} q-points")
    
    def compute_bare_potential_V(self, rho_q_nR: np.ndarray, q_vec: np.ndarray) -> np.ndarray:
        """
        Compute bare potential V_q^nR(r) from density.
        
        V_q^nR(r) = ∫ d³r' v(r,r') ρ_q^nR(r')
        
        In G-space: V_q^nR(G) = v(q+G) * ρ_q^nR(G)
        
        Args:
            rho_q_nR: Density in G-space [nwann, ngvecs]
            q_vec: q-vector in crystal coordinates
            
        Returns:
            V_q_nR: Bare potential in G-space [nwann, ngvecs]
        """
        if not hasattr(self, 'em1s_db'):
            raise ValueError("Screening database not loaded. Call load_screening_db first.")
        
        # Find q-point in em1s database
        # (This is simplified - needs proper q-point matching)
        iq = 0  # TODO: Implement proper q-point finding
        
        # Get bare Coulomb potential sqrt(v)
        sqrt_v = self.em1s_db.sqrt_V[iq, :]  # [ngvecs]
        
        # V = v * rho, where v = sqrt_v^2
        v_q = sqrt_v ** 2
        
        # Apply to each Wannier function
        V_q_nR = rho_q_nR * v_q[None, :]
        
        return V_q_nR
    
    def __repr__(self):
        """String representation."""
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