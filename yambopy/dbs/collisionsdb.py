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
    -------------------
    Header file: ndb.COLLISIONS_{type}_header
      - COLLISIONS_STATE: (4, n_collisions) array with [n, m, k, spin] for each base collision
      - X_X_band_range: band range used in response function (not collision bands!)
    
    Main data file: ndb.COLLISIONS_{type}
      - COLLISIONS_v: (n_collisions, n_collision_states, 2) array
        * n_collisions: number of stored base collision pairs (upper triangle)
        * n_collision_states: dimension of primed state space = n_bands² × n_kpts × n_spin
        * 2: real and imaginary parts
      - N_COLLISIONS_STATES: total dimension of primed state space
    
    Physical meaning:
    -----------------
    COLLISIONS_v[icoll, istate] = scattering matrix element between:
      - Base collision icoll: (n, m, k, spin) from COLLISIONS_STATE[:,icoll]
      - Primed state istate: (n', m', k', spin') from the full collision state space
    
    The collision represents: |n,k,spin⟩⟨m,k,spin| → |n',k',spin'⟩⟨m',k',spin'|
    
    Storage:
    --------
    Only upper triangle of (n,m) pairs is stored: n >= m
    N_COLLISIONS_STATES = n_coll_bands × n_coll_bands × n_kpts_ibz × n_spin
    
    Attributes:
    -----------
    collision_type : str
        'HXC' for bare Coulomb V, 'COH' for screened W
    collision_state : ndarray (4, n_collisions)
        Base collision states [n, m, k_idx, spin] (Fortran 1-based indexing)
    collision_v : ndarray (n_collisions, n_collision_states)
        Collision matrix elements [Hartree], complex
    n_collisions : int
        Number of stored base collision pairs
    n_collision_states : int
        Dimension of primed state space (N_COLLISIONS_STATES)
    n_kpts_ibz : int
        Number of k-points in IBZ
    n_coll_bands : int
        Number of collision bands
    coll_band_range : ndarray [min_band, max_band]
        Actual collision bands (1-based Fortran indexing)
    cv_only : bool
        True if CV_only (conduction-valence only) mode is enabled
    n_valence_bands : int
        Number of valence bands (only for CV_only mode)
    n_conduction_bands : int
        Number of conduction bands (only for CV_only mode)
    valence_band_range : ndarray [min, max]
        Valence band range (only for CV_only mode)
    conduction_band_range : ndarray [min, max]
        Conduction band range (only for CV_only mode)
    primed_state_map : ndarray (n_collision_states, 4)
        Map from istate -> (n', m', k', spin') (to be built on demand)
    """
    
    def __init__(self, path: str = '.', collision_type: str = 'HXC'):
        """
        Initialize COLLISION database reader.
        
        Parameters
        ----------
        path : str
            Path to directory containing ndb.COLLISIONS_* files
        collision_type : str
            'HXC' for V_nm (bare Coulomb), 'COH' for W_nm (screened)
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
            # Each column is [n, m, k_idx, spin] for a base collision
            self.collision_state = np.array(f.variables['COLLISIONS_STATE'][:])
            self.n_collisions = self.collision_state.shape[1]
            
            # Read response function band range (NOT collision band range!)
            if 'X_X_band_range' in f.variables:
                self.response_band_range = np.array(f.variables['X_X_band_range'][:])
            else:
                self.response_band_range = None
            
            # Check for CV_only mode
            if 'CV_only_scattering' in f.variables:
                self.cv_only = bool(f.variables['CV_only_scattering'][:])
            else:
                self.cv_only = False
            
            # Read valence and conduction band info if CV_only
            if self.cv_only:
                # Try to read from database
                if 'GreenF_n_bands_valence' in f.variables:
                    n_val = int(f.variables['GreenF_n_bands_valence'][:])
                else:
                    n_val = None
                
                if 'GreenF_n_bands_conduction' in f.variables:
                    n_cond = int(f.variables['GreenF_n_bands_conduction'][:])
                else:
                    n_cond = None
                
                # Store for later inference if not found
                self._n_val_db = n_val
                self._n_cond_db = n_cond
        
        # Determine actual collision band range from collision states
        n_bands = self.collision_state[0, :]  # First band index
        m_bands = self.collision_state[1, :]  # Second band index
        self.coll_band_range = np.array([
            min(n_bands.min(), m_bands.min()),
            max(n_bands.max(), m_bands.max())
        ], dtype=int)
        self.n_coll_bands = int(self.coll_band_range[1] - self.coll_band_range[0] + 1)
        
        # Infer valence/conduction bands for CV_only mode
        if self.cv_only:
            # In CV_only: n bands are valence (lower), m bands are conduction (higher)
            # Infer from collision states if not in database
            unique_n = np.unique(n_bands)
            unique_m = np.unique(m_bands)
            
            self.valence_band_range = np.array([unique_n.min(), unique_n.max()], dtype=int)
            self.conduction_band_range = np.array([unique_m.min(), unique_m.max()], dtype=int)
            
            self.n_valence_bands = len(unique_n)
            self.n_conduction_bands = len(unique_m)
        else:
            self.valence_band_range = None
            self.conduction_band_range = None
            self.n_valence_bands = None
            self.n_conduction_bands = None
        
        # Get unique k-points in IBZ from collision states
        k_indices = np.unique(self.collision_state[2, :]).astype(int)
        self.ibz_k_indices = k_indices
        self.n_kpts_ibz = len(k_indices)
        
        # Get spin info
        spin_indices = np.unique(self.collision_state[3, :]).astype(int)
        self.n_spin = len(spin_indices)
        
        print(f"  Collision bands: {self.coll_band_range[0]} - {self.coll_band_range[1]} (Fortran indexing)")
        print(f"  Number of stored base collisions: {self.n_collisions}")
        print(f"  K-points in IBZ: {self.n_kpts_ibz}")
        print(f"  Spins: {self.n_spin}")
        print(f"  CV_only mode: {self.cv_only}")
        if self.cv_only:
            print(f"    Valence bands: {self.valence_band_range[0]} - {self.valence_band_range[1]} ({self.n_valence_bands} bands)")
            print(f"    Conduction bands: {self.conduction_band_range[0]} - {self.conduction_band_range[1]} ({self.n_conduction_bands} bands)")
        
        # Read main data file
        print(f"Loading {main_file}...")
        with Dataset(main_path, 'r') as f:
            # Read N_COLLISIONS_STATES: dimension of primed state space
            # Formula depends on CV_only mode:
            #   CV_only: Nbv × Nbc × (2×Nk) × n_spin  (2×Nk for time-reversal)
            #   General: Nb1 × Nb2 × Nk × n_spin
            self.n_collision_states = int(f.variables['N_COLLISIONS_STATES'][:])
            
            # Calculate expected dimension
            if self.cv_only:
                n_expected = self.n_valence_bands * self.n_conduction_bands * (2 * self.n_kpts_ibz) * self.n_spin
                print(f"  N_COLLISIONS_STATES (primed state dimension): {self.n_collision_states}")
                print(f"  Expected (CV_only): {self.n_valence_bands}×{self.n_conduction_bands}×(2×{self.n_kpts_ibz})×{self.n_spin} = {n_expected}")
            else:
                n_expected = self.n_coll_bands * self.n_coll_bands * self.n_kpts_ibz * self.n_spin
                print(f"  N_COLLISIONS_STATES (primed state dimension): {self.n_collision_states}")
                print(f"  Expected (General): {self.n_coll_bands}×{self.n_coll_bands}×{self.n_kpts_ibz}×{self.n_spin} = {n_expected}")
            
            # Read collision matrix elements
            # Shape: (n_collisions, n_collision_states, 2)
            coll_v_raw = np.array(f.variables['COLLISIONS_v'][:])
            
            print(f"  COLLISIONS_v shape: {coll_v_raw.shape}")
            
            # Convert to complex array: (n_collisions, n_collision_states)
            self.collision_v = coll_v_raw[:, :, 0] + 1j * coll_v_raw[:, :, 1]
        
        # Primed state map (built on demand)
        self.primed_state_map = None
        
        print(f"✓ COLLISION database loaded successfully")
    
    def build_primed_state_map(self):
        """
        Build the mapping from primed state index to (n', m', k', spin').
        
        The ordering depends on CV_only mode:
        
        General mode:
            istate = (n' - n_min) + (m' - n_min) * n_bands + (k' - 1) * n_bands² + (spin' - 1) * n_bands² * n_kpts
        
        CV_only mode:
            Loop order: spin → k (includes both k and -k, 2×Nk) → m_conduction → n_valence
            istate = (n' - n_val_min) + (m' - m_cond_min) * Nbv + ik * Nbv*Nbc + (spin' - 1) * Nbv*Nbc*2*Nk
            where ik goes from 0 to 2*Nk-1 (includes time-reversed -k points)
        
        This creates an array: primed_state_map[istate, :] = [n', m', k', spin']
        """
        print("Building primed state map...")
        
        self.primed_state_map = np.zeros((self.n_collision_states, 4), dtype=int)
        
        istate = 0
        
        if self.cv_only:
            # CV_only mode: valence × conduction × (2×k) × spin
            n_val_min = self.valence_band_range[0]
            m_cond_min = self.conduction_band_range[0]
            
            k_list = sorted(self.ibz_k_indices)
            
            for spin in range(1, self.n_spin + 1):
                # Loop over k and -k (time-reversed pairs)
                for ik in range(2 * self.n_kpts_ibz):
                    # Map to actual k-point index
                    # First Nk indices: regular k-points
                    # Next Nk indices: time-reversed -k points
                    k_idx = k_list[ik % self.n_kpts_ibz]
                    
                    # Loop over conduction bands (m')
                    for m in range(m_cond_min, m_cond_min + self.n_conduction_bands):
                        # Loop over valence bands (n')
                        for n in range(n_val_min, n_val_min + self.n_valence_bands):
                            if istate < self.n_collision_states:
                                self.primed_state_map[istate] = [n, m, k_idx, spin]
                                istate += 1
            
            print(f"  CV_only ordering: spin → k (2×{self.n_kpts_ibz}) → m_cond → n_val")
        else:
            # General mode: n' × m' × k × spin
            n_min = self.coll_band_range[0]
            
            for spin in range(1, self.n_spin + 1):
                for k in sorted(self.ibz_k_indices):
                    for m in range(n_min, n_min + self.n_coll_bands):
                        for n in range(n_min, n_min + self.n_coll_bands):
                            if istate < self.n_collision_states:
                                self.primed_state_map[istate] = [n, m, k, spin]
                                istate += 1
            
            print(f"  General ordering: spin → k → m → n")
        
        if istate != self.n_collision_states:
            print(f"Warning: Built {istate} states but expected {self.n_collision_states}")
        else:
            print(f"✓ Built primed state map with {self.n_collision_states} states")
    
    def get_collision_element(self, n: int, m: int, k: int, 
                              n_prime: int, m_prime: int, k_prime: int,
                              spin: int = 1) -> complex:
        """
        Get collision matrix element V_{(n,m,k),(n',m',k')} or W_{(n,m,k),(n',m',k')}.
        
        Parameters
        ----------
        n, m : int
            Base state band indices (Fortran 1-based)
        k : int
            Base state k-point index (Fortran 1-based, in IBZ)
        n_prime, m_prime : int
            Primed state band indices (Fortran 1-based)
        k_prime : int
            Primed state k-point index (Fortran 1-based, in IBZ)
        spin : int
            Spin index (1 or 2)
            
        Returns
        -------
        complex
            Collision matrix element in Hartree units
            Returns 0 if not found
        """
        # Find base collision index
        icoll = self._find_collision_index(n, m, k, spin)
        if icoll is None:
            return 0.0 + 0.0j
        
        # Find primed state index
        if self.primed_state_map is None:
            self.build_primed_state_map()
        
        istate = self._find_primed_state_index(n_prime, m_prime, k_prime, spin)
        if istate is None:
            return 0.0 + 0.0j
        
        return self.collision_v[icoll, istate]
    
    def get_collision_matrix_at_k(self, k: int, k_prime: int = None, spin: int = 1) -> np.ndarray:
        """
        Get collision matrix V_nm(k, k') or W_nm(k, k') at specific k-points.
        
        If k_prime is None, returns diagonal elements: V_nm(k, k).
        This is the typical case for Wannier transformations.
        
        Parameters
        ----------
        k : int
            Base state k-point index (Fortran 1-based)
        k_prime : int, optional
            Primed state k-point index (Fortran 1-based)
            If None, uses k_prime = k (diagonal)
        spin : int
            Spin index (1 or 2)
            
        Returns
        -------
        ndarray (n_coll_bands, n_coll_bands)
            Collision matrix V_nm or W_nm in Hartree units
        """
        if k_prime is None:
            k_prime = k
        
        # Build primed state map if needed
        if self.primed_state_map is None:
            self.build_primed_state_map()
        
        # Initialize matrix
        V_matrix = np.zeros((self.n_coll_bands, self.n_coll_bands), dtype=np.complex128)
        
        n_min = self.coll_band_range[0]
        
        # Fill matrix
        for n_band in range(n_min, n_min + self.n_coll_bands):
            for m_band in range(n_min, n_min + self.n_coll_bands):
                for n_prime_band in range(n_min, n_min + self.n_coll_bands):
                    for m_prime_band in range(n_min, n_min + self.n_coll_bands):
                        # Get collision element
                        V_elem = self.get_collision_element(n_band, m_band, k, 
                                                           n_prime_band, m_prime_band, k_prime, spin)
                        
                        # Store in matrix (convert to 0-based Python indexing)
                        n_idx = n_band - n_min
                        m_idx = m_band - n_min
                        n_prime_idx = n_prime_band - n_min
                        m_prime_idx = m_prime_band - n_min
                        
                        # For diagonal k=k', the matrix is V[n,m,n',m']
                        # Typically we want V[n,n'] * delta_{m,m'}
                        # or the full 4D tensor collapsed appropriately
                        if m_band == m_prime_band:  # Diagonal in m indices
                            V_matrix[n_idx, n_prime_idx] += V_elem
        
        return V_matrix
    
    def _find_collision_index(self, n: int, m: int, k: int, spin: int) -> int:
        """
        Find the collision index for given (n, m, k, spin).
        
        Returns None if not found.
        """
        for icoll in range(self.n_collisions):
            state = self.collision_state[:, icoll]
            if (state[0] == n and state[1] == m and 
                state[2] == k and state[3] == spin):
                return icoll
        
        # Try swapped (m, n) if not found (due to upper triangle storage)
        for icoll in range(self.n_collisions):
            state = self.collision_state[:, icoll]
            if (state[0] == m and state[1] == n and 
                state[2] == k and state[3] == spin):
                return icoll
        
        return None
    
    def _find_primed_state_index(self, n_prime: int, m_prime: int, k_prime: int, spin: int) -> int:
        """
        Find the primed state index for given (n', m', k', spin).
        
        Returns None if not found.
        """
        for istate in range(self.n_collision_states):
            state = self.primed_state_map[istate]
            if (state[0] == n_prime and state[1] == m_prime and 
                state[2] == k_prime and state[3] == spin):
                return istate
        
        return None
    
    def get_collision_state(self, icoll: int) -> tuple:
        """
        Get the (n, m, k, spin) tuple for a given collision index.
        
        Parameters
        ----------
        icoll : int
            Collision index (0-based Python indexing)
            
        Returns
        -------
        tuple (n, m, k, spin)
            Base collision state with Fortran 1-based indices
        """
        if icoll < 0 or icoll >= self.n_collisions:
            raise ValueError(f"Invalid collision index {icoll}")
        
        state = self.collision_state[:, icoll]
        return tuple(int(x) for x in state)
    
    def get_primed_state(self, istate: int) -> tuple:
        """
        Get the (n', m', k', spin') tuple for a given primed state index.
        
        Parameters
        ----------
        istate : int
            Primed state index (0-based Python indexing)
            
        Returns
        -------
        tuple (n', m', k', spin')
            Primed state with Fortran 1-based indices
        """
        if self.primed_state_map is None:
            self.build_primed_state_map()
        
        if istate < 0 or istate >= self.n_collision_states:
            raise ValueError(f"Invalid primed state index {istate}")
        
        state = self.primed_state_map[istate]
        return tuple(int(x) for x in state)
    
    def __str__(self):
        lines = []
        lines.append("=" * 70)
        lines.append(f"Yambo COLLISION Database ({self.collision_type})")
        lines.append("=" * 70)
        lines.append(f"Path:                    {self.path}")
        lines.append(f"Base file:               {self.base_name}")
        lines.append(f"Type:                    {'Bare Coulomb V_nm' if self.collision_type == 'HXC' else 'Screened W_nm'}")
        lines.append(f"")
        lines.append(f"Base collision pairs:    {self.n_collisions}")
        lines.append(f"Primed state space:      {self.n_collision_states}")
        lines.append(f"Collision bands:         {self.coll_band_range[0]} - {self.coll_band_range[1]} ({self.n_coll_bands} bands)")
        if self.response_band_range is not None:
            lines.append(f"Response bands:          {self.response_band_range[0]} - {self.response_band_range[1]}")
        lines.append(f"K-points (IBZ):          {self.n_kpts_ibz}")
        lines.append(f"Spin channels:           {self.n_spin}")
        lines.append(f"CV_only mode:            {self.cv_only}")
        if self.cv_only:
            lines.append(f"  Valence bands:         {self.valence_band_range[0]} - {self.valence_band_range[1]} ({self.n_valence_bands} bands)")
            lines.append(f"  Conduction bands:      {self.conduction_band_range[0]} - {self.conduction_band_range[1]} ({self.n_conduction_bands} bands)")
            lines.append(f"  Time-resolved k-pts:   2×{self.n_kpts_ibz} = {2*self.n_kpts_ibz} (includes -k)")
        lines.append("")
        lines.append("Base collision states (first 5):")
        for i in range(min(5, self.n_collisions)):
            n, m, k, spin = self.get_collision_state(i)
            lines.append(f"  [{i:3d}] (n={n}, m={m}, k={k}, spin={spin})")
        if self.n_collisions > 5:
            lines.append(f"  ... and {self.n_collisions - 5} more")
        
        if self.primed_state_map is not None:
            lines.append("")
            lines.append("Primed states (first 5):")
            for i in range(min(5, self.n_collision_states)):
                np, mp, kp, spinp = self.get_primed_state(i)
                lines.append(f"  [{i:3d}] (n'={np}, m'={mp}, k'={kp}, spin'={spinp})")
            if self.n_collision_states > 5:
                lines.append(f"  ... and {self.n_collision_states - 5} more")
        
        lines.append("=" * 70)
        return "\n".join(lines)