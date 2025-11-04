#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: HPC, FP, RR
#
# This file is part of the yambopy project
#
import os
import numpy as np
from netCDF4 import Dataset

class YamboCollisionDB(object):
    """
    Class to handle COLLISION databases from Yambo.
    
    Reads collision matrix elements V_nm(k,k',q) or W_nm(k,k',q) 
    from ndb.COLLISIONS_HXC or ndb.COLLISIONS_COH databases.
    
    Database structure:
    - Header file: ndb.COLLISIONS_{type}_header
      Contains COLLISIONS_STATE array: [n, m, k_idx, spin] for each collision
      Contains X_X_band_range: band range used in response function
    
    - Main data file: ndb.COLLISIONS_{type}
      Contains COLLISIONS_v: collision matrix elements (n_collisions, n_kpts, 2)
      Contains N_COLLISIONS_STATES: number of k-points in IBZ
    
    Note: COLLISION databases do NOT have fragment files (unlike em1s databases).
          Data is stored for IBZ k-points only. Use expand_to_bz() to get full BZ.
    
    Attributes:
        collision_type: 'HXC' for bare Coulomb V, 'COH' for screened W
        collision_state: Array (4, n_collisions) with [n, m, k_idx, spin] for each collision
        collision_v: Array (n_collisions, n_kpts_ibz) of complex collision matrix elements [Hartree]
        n_collisions: Number of unique collision pairs (n,m) combinations
        n_kpts_ibz: Number of k-points in IBZ
        coll_band_range: [min_band, max_band] actual collision bands (1-based Fortran indexing)
        response_band_range: [min_band, max_band] from X_X_band_range (response function)
    """
    
    def __init__(self, path: str = '.', collision_type: str = 'HXC'):
        """
        Initialize COLLISION database reader.
        
        Args:
            path: Path to directory containing ndb.COLLISIONS_* files
            collision_type: 'HXC' for V_nm (bare), 'COH' for W_nm (screened)
        """
        self.path = path
        self.collision_type = collision_type.upper()
        
        if self.collision_type not in ['HXC', 'COH']:
            raise ValueError(f"collision_type must be 'HXC' or 'COH', got {collision_type}")
        
        # File names
        self.base_name = f'ndb.COLLISIONS_{self.collision_type}'
        header_file = f'{self.base_name}_header'
        main_file = self.base_name
        
        header_path = os.path.join(path, header_file)
        main_path = os.path.join(path, main_file)
        
        # Check files exist
        if not os.path.isfile(header_path):
            raise FileNotFoundError(f"Header file {header_path} not found")
        if not os.path.isfile(main_path):
            raise FileNotFoundError(f"Main file {main_path} not found")
        
        # Read header file
        print(f"Loading {header_file}...")
        with Dataset(header_path, 'r') as f:
            # Read collision states: (4, n_collisions) array
            # Each column is [n, m, k_idx, spin] for a collision
            self.collision_state = np.array(f.variables['COLLISIONS_STATE'][:])
            self.n_collisions = self.collision_state.shape[1]
            
            # Read response function band range (not necessarily collision band range!)
            if 'X_X_band_range' in f.variables:
                self.response_band_range = np.array(f.variables['X_X_band_range'][:])
            else:
                self.response_band_range = None
        
        # Determine actual collision band range from collision states
        n_bands = self.collision_state[0, :]  # First index band
        m_bands = self.collision_state[1, :]  # Second index band
        self.coll_band_range = np.array([
            min(n_bands.min(), m_bands.min()),
            max(n_bands.max(), m_bands.max())
        ])
        self.n_coll_bands = self.coll_band_range[1] - self.coll_band_range[0] + 1
        
        # Get unique k-points in IBZ
        k_indices = np.unique(self.collision_state[2, :])
        self.ibz_k_indices = k_indices
        self.n_kpts_ibz = len(k_indices)
        
        print(f"  Collision bands: {self.coll_band_range[0]} - {self.coll_band_range[1]} (Fortran indexing)")
        print(f"  Number of collision pairs: {self.n_collisions}")
        print(f"  K-points in IBZ: {self.n_kpts_ibz}")
        
        # Read main data file
        print(f"Loading {main_file}...")
        with Dataset(main_path, 'r') as f:
            # Read number of k-points (should match IBZ count)
            n_kpts_check = int(f.variables['N_COLLISIONS_STATES'][:])
            if n_kpts_check != self.n_kpts_ibz:
                print(f"  Warning: N_COLLISIONS_STATES ({n_kpts_check}) != n_kpts_ibz ({self.n_kpts_ibz})")
            
            # Read collision matrix elements
            # Shape: (n_collisions, n_kpts, 2) where last dim is [real, imag]
            coll_v_raw = np.array(f.variables['COLLISIONS_v'][:])
            
            # Convert to complex array: (n_collisions, n_kpts_ibz)
            self.collision_v_ibz = coll_v_raw[:, :, 0] + 1j * coll_v_raw[:, :, 1]
        
        # Initially no expansion to full BZ
        self.collision_v_bz = None
        self.lattice = None
        
        print(f"✓ COLLISION database loaded successfully")
    
    def expand_to_bz(self, lattice):
        """
        Expand collision matrix elements from IBZ to full BZ using YamboLatticeDB.
        
        Args:
            lattice: YamboLatticeDB object with kpoints_indexes defined
        """
        if not hasattr(lattice, 'kpoints_indexes'):
            raise ValueError("YamboLatticeDB must have kpoints_indexes. Call expand_kpts() first.")
        
        self.lattice = lattice
        n_kpts_bz = len(lattice.kpoints_indexes)
        
        print(f"Expanding collision data from IBZ ({self.n_kpts_ibz}) to full BZ ({n_kpts_bz})...")
        
        # Expand: collision_v_bz[icoll, ik_bz] = collision_v_ibz[icoll, kpoints_indexes[ik_bz]]
        # Note: kpoints_indexes maps ik_bz -> ik_ibz, and k-points in collision_state are 1-based
        # So we need to convert: collision_state k-index (1-based) -> Python index (0-based)
        
        # Build full BZ collision data
        self.collision_v_bz = np.zeros((self.n_collisions, n_kpts_bz), dtype=np.complex128)
        
        for icoll in range(self.n_collisions):
            # Get the IBZ k-index for this collision (1-based Fortran)
            k_ibz_fortran = self.collision_state[2, icoll]
            k_ibz_python = k_ibz_fortran - 1  # Convert to 0-based
            
            # This collision is defined at k_ibz_python in the IBZ
            # Expand it to all k-points in BZ that map to this IBZ point
            for ik_bz in range(n_kpts_bz):
                if lattice.kpoints_indexes[ik_bz] == k_ibz_python:
                    self.collision_v_bz[icoll, ik_bz] = self.collision_v_ibz[icoll, k_ibz_python]
        
        print(f"✓ Collision data expanded to full BZ")
    
    def get_collision_by_index(self, icoll: int, ik: int, use_bz: bool = False) -> complex:
        """
        Get collision matrix element by collision index and k-point index.
        
        Args:
            icoll: Collision pair index (0-based Python indexing)
            ik: k-point index (0-based Python indexing)
            use_bz: If True, use full BZ data; if False, use IBZ data
            
        Returns:
            Matrix element in Hartree units (complex)
        """
        if icoll < 0 or icoll >= self.n_collisions:
            raise ValueError(f"Invalid collision index {icoll}. Must be 0 <= icoll < {self.n_collisions}")
        
        if use_bz:
            if self.collision_v_bz is None:
                raise ValueError("BZ data not available. Call expand_to_bz() first.")
            if ik < 0 or ik >= self.collision_v_bz.shape[1]:
                raise ValueError(f"Invalid k-point index {ik}. Must be 0 <= ik < {self.collision_v_bz.shape[1]}")
            return self.collision_v_bz[icoll, ik]
        else:
            if ik < 0 or ik >= self.n_kpts_ibz:
                raise ValueError(f"Invalid k-point index {ik}. Must be 0 <= ik < {self.n_kpts_ibz}")
            return self.collision_v_ibz[icoll, ik]
    
    def get_collision(self, n: int, m: int, ik: int, spin: int = 1, use_bz: bool = False) -> complex:
        """
        Get collision matrix element V_nm or W_nm for given bands and k-point.
        
        Args:
            n, m: Band indices (1-based Fortran indexing, as in Yambo)
            ik: k-point index (1-based Fortran indexing, as in Yambo)
            spin: Spin index (1 or 2)
            use_bz: If True, search in full BZ; if False, search in IBZ
            
        Returns:
            Matrix element in Hartree units (complex)
            Returns 0 if collision not found in database
        """
        # Find collision index matching (n, m, k_idx, spin)
        for icoll in range(self.n_collisions):
            state = self.collision_state[:, icoll]
            if (state[0] == n and state[1] == m and 
                state[2] == ik and state[3] == spin):
                # Found matching collision
                # Return value at the k-point (convert ik from 1-based to 0-based)
                return self.get_collision_by_index(icoll, ik - 1, use_bz=use_bz)
        
        # Collision not found in database
        return 0.0 + 0.0j
    
    def get_collision_state(self, icoll: int) -> tuple:
        """
        Get the (n, m, k_idx, spin) tuple for a given collision index.
        
        Args:
            icoll: Collision pair index (0-based Python indexing)
            
        Returns:
            Tuple (n, m, k_idx, spin) with 1-based Fortran indices
        """
        if icoll < 0 or icoll >= self.n_collisions:
            raise ValueError(f"Invalid collision index {icoll}. Must be 0 <= icoll < {self.n_collisions}")
        
        state = self.collision_state[:, icoll]
        return tuple(int(x) for x in state)
    
    def get_collision_array(self, icoll: int, use_bz: bool = False) -> np.ndarray:
        """
        Get collision matrix elements at all k-points for a given collision.
        
        Args:
            icoll: Collision pair index (0-based Python indexing)
            use_bz: If True, return full BZ data; if False, return IBZ data
            
        Returns:
            Array (n_kpts,) of complex collision matrix elements
        """
        if icoll < 0 or icoll >= self.n_collisions:
            raise ValueError(f"Invalid collision index {icoll}. Must be 0 <= icoll < {self.n_collisions}")
        
        if use_bz:
            if self.collision_v_bz is None:
                raise ValueError("BZ data not available. Call expand_to_bz() first.")
            return self.collision_v_bz[icoll, :]
        else:
            return self.collision_v_ibz[icoll, :]
    
    def get_collision_matrix_at_k(self, ik: int, use_bz: bool = False, 
                                   band_offset: int = 0) -> np.ndarray:
        """
        Get full collision matrix V_nm(k) or W_nm(k) for all bands at a given k-point.
        
        Useful for integration with WannierYamboInterface where band indexing may differ.
        
        Args:
            ik: k-point index (0-based Python indexing)
            use_bz: If True, use full BZ data; if False, use IBZ data
            band_offset: Offset to apply to band indices for 0-based Python indexing
                        (e.g., if collision bands are 3-6 in Fortran, use band_offset=-3 
                         to get 0-based Python indices 0-3)
            
        Returns:
            V_nm or W_nm matrix (n_coll_bands, n_coll_bands) in Hartree units
        """
        # Initialize matrix
        V_nm = np.zeros((self.n_coll_bands, self.n_coll_bands), dtype=np.complex128)
        
        # Determine which k-point to use
        if use_bz:
            if self.collision_v_bz is None:
                raise ValueError("BZ data not available. Call expand_to_bz() first.")
            k_fortran = ik + 1  # Convert to 1-based
        else:
            k_fortran = ik + 1  # Convert to 1-based
        
        # Fill matrix
        for icoll in range(self.n_collisions):
            n, m, k_coll, spin = self.get_collision_state(icoll)
            
            # Check if this collision is for the requested k-point
            if k_coll == k_fortran:
                # Convert band indices
                n_idx = n - self.coll_band_range[0] + band_offset
                m_idx = m - self.coll_band_range[0] + band_offset
                
                if 0 <= n_idx < self.n_coll_bands and 0 <= m_idx < self.n_coll_bands:
                    V_nm[n_idx, m_idx] = self.get_collision_by_index(icoll, ik, use_bz=use_bz)
        
        return V_nm
    
    def __str__(self):
        lines = []
        lines.append("=" * 70)
        lines.append(f"Yambo COLLISION Database ({self.collision_type})")
        lines.append("=" * 70)
        lines.append(f"Path:                {self.path}")
        lines.append(f"Base file:           {self.base_name}")
        lines.append(f"Type:                {'Bare Coulomb V_nm' if self.collision_type == 'HXC' else 'Screened W_nm'}")
        lines.append(f"N collision pairs:   {self.n_collisions}")
        lines.append(f"Collision bands:     {self.coll_band_range[0]} - {self.coll_band_range[1]} ({self.n_coll_bands} bands)")
        if self.response_band_range is not None:
            lines.append(f"Response bands:      {self.response_band_range[0]} - {self.response_band_range[1]}")
        lines.append(f"K-points (IBZ):      {self.n_kpts_ibz}")
        if self.collision_v_bz is not None:
            lines.append(f"K-points (Full BZ):  {self.collision_v_bz.shape[1]}")
            lines.append(f"Expanded to BZ:      Yes")
        else:
            lines.append(f"Expanded to BZ:      No")
        lines.append("")
        lines.append("Collision states (first 5):")
        for i in range(min(5, self.n_collisions)):
            n, m, k_idx, spin = self.get_collision_state(i)
            value_ibz = self.collision_v_ibz[i, k_idx - 1]  # k_idx is 1-based
            lines.append(f"  [{i:3d}] bands ({n},{m}) k={k_idx} spin={spin} | V_ibz = {value_ibz:.6e}")
        if self.n_collisions > 5:
            lines.append(f"  ... and {self.n_collisions - 5} more")
        return "\n".join(lines)