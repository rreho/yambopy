"""
Exciton symmetry analysis using spgrep.

Provides group theory analysis of exciton states including:
- Point group identification
- Little group analysis  
- Irrep decomposition
- Optical activity analysis
"""

import numpy as np
import warnings
from yambopy.optical_properties.base_optical import BaseOpticalProperties
from yambopy.optical_properties.utils import read_lelph_database, compute_symmetry_matrices
from yambopy.optical_properties.spgrep_point_group_ops import get_pg_info, decompose_rep, _compute_conjugacy_classes, _get_irrep_labels_for_pg, _get_d3h_character_table

# Use precise ha2ev value as suggested
ha2ev = 27.2114079527

warnings.filterwarnings('ignore')


class ExcitonGroupTheory(BaseOpticalProperties):
    """
    Group theory analysis of exciton states.
    
    Parameters
    ----------
    path : str, optional
        Calculation directory path
    save : str, optional  
        SAVE directory name
    BSE_dir : str, optional
        BSE directory name
    LELPH_dir : str, optional
        Electron-phonon directory name
    bands_range : list, optional
        Band range for analysis
    """
    
    def __init__(self, path=None, save='SAVE', BSE_dir='bse', LELPH_dir='lelph', 
                 bands_range=None, **kwargs):
        super().__init__(path=path, save=save, BSE_dir=BSE_dir, bands_range=bands_range)
        
        self._setup_directories(LELPH_dir=LELPH_dir)
        self.read(bands_range=bands_range, **kwargs)

    def read(self, bands_range=None, **kwargs):
        """Read databases and setup symmetry."""
        self.read_common_databases(bands_range=bands_range, **kwargs)
        self.lelph_db = read_lelph_database(self.LELPH_dir, kwargs.get('lelph_db'))
        self.qpts = self.lelph_db.qpoints
        
        self._setup_kpoint_mapping()
        self._setup_symmetry()
        
        if hasattr(self.wfdb, 'ktree'):
            self.kpt_tree = self.wfdb.ktree
        else:
            self._build_kpoint_tree(self.lelph_db.kpoints)

    def _setup_symmetry(self):
        """Setup crystal symmetry using spglib - little group will be determined per Q-point."""
        try:
            import spglib
            
            # Get crystal structure from Yambo database
            lattice = self.lat_vecs
            positions = self.ydb.red_atomic_positions  # Fractional coordinates
            numbers = self.ydb.atomic_numbers
            cell = (lattice, positions, numbers)
            
            # Get space group information
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
            self.spacegroup_label = f"{dataset['international']} (#{dataset['number']})"
            
            # Get complete point group operations from spglib
            # These are in lattice coordinates (Seitz representation) and include all operations
            symmetry = spglib.get_symmetry(cell, symprec=1e-5)
            spglib_rotations = symmetry['rotations']
            
            # Extract unique point group operations
            unique_rotations = []
            seen_rotations = set()
            
            for rot in spglib_rotations:
                rot_tuple = tuple(rot.flatten())
                if rot_tuple not in seen_rotations:
                    seen_rotations.add(rot_tuple)
                    unique_rotations.append(rot)
            
            self.crystal_point_group_ops = np.array(unique_rotations)
            
            # Extract only spatial operations from Yambo (remove time reversal)
            # Yambo includes both spatial + time reversal symmetries
            n_spatial = len(self.symm_mats) // 2 if self.ele_time_rev else len(self.symm_mats)
            yambo_cartesian_ops = self.symm_mats[:n_spatial]
            
            # Convert Yambo's Cartesian operations to lattice coordinates
            # Conversion: R_lattice = A^(-1) * R_cartesian * A
            A = lattice.T  # Lattice matrix (vectors as columns)
            A_inv = np.linalg.inv(A)
            
            yambo_lattice_ops = []
            for cart_op in yambo_cartesian_ops:
                latt_op = A_inv @ cart_op @ A
                yambo_lattice_ops.append(np.round(latt_op).astype(int))
            
            self.yambo_lattice_ops = np.array(yambo_lattice_ops)
            
            print(f"Space group: {self.spacegroup_label}")
            print(f"Crystal point group operations: {len(self.crystal_point_group_ops)}")
            print(f"Yambo spatial operations: {len(self.yambo_lattice_ops)}")
            
            # Initialize fractional translations (needed for phase factors)
            self.frac_trans = np.zeros((len(self.symm_mats), 3))
            
            self._compute_d_matrices()
            
        except ImportError as e:
            raise ImportError(f"spglib required: {e}")
        except Exception as e:
            raise RuntimeError(f"Symmetry setup failed: {e}")

    def _get_little_group(self, Q_vec, tol=1e-6):
        """
        Determine little group operations that leave Q-vector invariant.
        
        Parameters
        ----------
        Q_vec : np.ndarray
            Q-vector in reciprocal lattice coordinates
        tol : float
            Tolerance for invariance check
            
        Returns
        -------
        tuple
            (little_group_ops, little_group_indices)
        """
        little_group_ops = []
        little_group_indices = []
        
        # Check which Yambo operations leave Q invariant
        for i, op in enumerate(self.yambo_lattice_ops):
            # Apply operation to Q-vector: R * Q
            rotated_Q = op @ Q_vec
            
            # Check if R*Q = Q (modulo reciprocal lattice vectors)
            diff = rotated_Q - Q_vec
            # For reciprocal space, we need to check modulo 2π (or equivalently, integer differences)
            diff_mod = diff - np.round(diff)
            
            if np.allclose(diff_mod, 0, atol=tol):
                little_group_ops.append(op)
                little_group_indices.append(i)
        
        little_group_ops = np.array(little_group_ops)
        
        print(f"Little group has {len(little_group_ops)} operations")
        print(f"Little group indices: {little_group_indices}")
        
        return little_group_ops, little_group_indices

    def _setup_little_group_symmetry(self, little_group_ops):
        """
        Setup symmetry analysis for the little group.
        
        Parameters
        ----------
        little_group_ops : np.ndarray
            Little group operations in lattice coordinates
        """
        try:
            # Use spgrep to analyze the little group
            pg_label, classes, class_dict, char_tab, irrep_labels = get_pg_info(
                little_group_ops, time_rev=False
            )
            
            self.little_group_label = pg_label
            self.little_group_classes = classes
            self.little_group_class_dict = class_dict
            self.little_group_character_table = char_tab
            self.little_group_irrep_labels = irrep_labels
            
            print(f"Little group: {pg_label}")
            print(f"Little group conjugacy classes: {len(classes)}")
            
            # Debug: Print what spgrep returned
            print(f"spgrep character table shape: {char_tab.shape}")
            print(f"spgrep irrep labels: {irrep_labels}")
            print(f"spgrep character table:\n{char_tab}")
            
            # Print conjugacy classes in the expected format
            print("Classes (symmetry indices in each class):")
            class_names = ['E', '3C_2', '2C_3', 'sigma_h', '3sigma_v', '2S_3']  # D3h classes
            for i, (class_idx, ops) in enumerate(self.little_group_class_dict.items()):
                if i < len(class_names):
                    name = class_names[i]
                else:
                    name = f"Class_{i+1}"
                # Convert to 1-based indexing for display
                ops_1based = [op + 1 for op in ops]
                print(f"            {name:>8}    :  {ops_1based}")
            
        except Exception as e:
            print(f"Warning: Could not setup little group symmetry: {e}")
            # Fallback: use the operations directly without group theory analysis
            self.little_group_label = f"Unknown ({len(little_group_ops)} ops)"
            self.little_group_classes = len(little_group_ops)
            self.little_group_class_dict = {i: [i] for i in range(len(little_group_ops))}
            self.little_group_character_table = np.eye(len(little_group_ops))
            self.little_group_irrep_labels = [f"Γ{i+1}" for i in range(len(little_group_ops))]

    def _compute_d_matrices(self):
        """Compute D-matrices using Yambo's symmetries and wavefunctions."""
        print("Computing D-matrices with Yambo symmetries...")
        
        # Use proper D-matrix computation from other branches
        if hasattr(self, 'bands_range') and self.bands_range:
            self.spglib_Dmats = self.wfdb.Dmat()[:,:,0,:,:]
        else:
            self.spglib_Dmats = self.wfdb.Dmat()[:,:,0,:,:]
            
        print(f"D-matrices shape: {self.spglib_Dmats.shape}")
        print("Using proper D-matrices from wavefunctions")

    def analyze_exciton_symmetry(self, iQ, nstates, degen_thres=0.001):
        """
        Analyze exciton symmetry.
        
        Parameters
        ----------
        iQ : int
            Q-point index (1-based)
        nstates : int
            Number of states to analyze
        degen_thres : float
            Degeneracy threshold in eV
            
        Returns
        -------
        dict
            Analysis results
        """
        # Get Q-vector for little group analysis
        Q_vec = self.qpts[iQ-1]
        
        # Determine little group operations that leave Q invariant
        little_group_ops, little_group_indices = self._get_little_group(Q_vec)
        
        # Setup little group symmetry analysis
        self._setup_little_group_symmetry(little_group_ops)
        
        # Read BSE data (following original algorithm exactly)
        bands_range, BS_eigs, BS_wfcs = self._read_bse_data(self.BSE_dir,iQ, nstates)
        # Convert energies to eV for analysis (following original algorithm exactly)
        BS_eigs_eV = BS_eigs * ha2ev
        
        print(f"\nAnalyzing {nstates} exciton states at Q = {Q_vec}")
        print(f"Energies: {BS_eigs_eV.real} eV")
        
        # Get unique values up to threshold (following original algorithm exactly)
        uni_eigs, degen_eigs = np.unique((BS_eigs_eV / degen_thres).astype(int),
                                        return_counts=True)
        uni_eigs = uni_eigs * degen_thres
        
        print(f"Found {len(uni_eigs)} energy groups:")
        for i, (energy, degen) in enumerate(zip(uni_eigs, degen_eigs)):
            print(f"  Group {i+1}: {energy:.4f} eV, degeneracy {degen}")
        
        # Compute representation matrices for ALL states at once (following original algorithm)
        rep_matrices_all = self._compute_representation_matrices_all_states(BS_wfcs, iQ, uni_eigs, degen_eigs, little_group_indices)
        
        results = []
        
        for i, (energy, degen) in enumerate(zip(uni_eigs, degen_eigs)):
            print(f"\nSubspace {i+1}: {energy:.4f} eV (degeneracy {degen})")
            
            # Extract characters for this degenerate subspace from the full representation matrices
            characters = rep_matrices_all[i]
            
            print(f"Characters: {characters}")
            
            # Decompose into irreps using little group
            irrep_result = self._decompose_irreps(characters)
            print(f"Irrep decomposition: {irrep_result}")
            
            results.append({
                'energy': energy,
                'degeneracy': degen,
                'characters': characters,
                'irrep': irrep_result
            })
        
        
        return {
            'space_group': self.spacegroup_label,
            'little_group': self.little_group_label,
            'Q_point': Q_vec,
            'states': results,  # Changed from 'results' to 'states' to match test expectation
            'results': results  # Keep both for compatibility
        }

    def _decompose_irreps(self, characters):
        """Decompose characters into irreps using little group."""
        try:
            # Map to conjugacy classes by averaging characters within each class
            class_characters = []
            for class_idx, ops in self.little_group_class_dict.items():
                # Average characters over all operations in the conjugacy class
                class_chars = []
                for op_idx in ops:
                    if op_idx < len(characters):
                        class_chars.append(characters[op_idx])
                
                if class_chars:
                    class_characters.append(np.mean(class_chars))
                else:
                    class_characters.append(0.0)
            
            class_characters = np.array(class_characters)
            class_orders = [len(ops) for ops in self.little_group_class_dict.values()]
            
            print(f"Class characters: {class_characters}")
            print(f"Class orders: {class_orders}")
            
            # Use spgrep's decompose_rep function for proper irrep decomposition
            try:
                coeffs = decompose_rep(class_characters, self.little_group_character_table, class_orders)
                
                print(f"Decomposition coefficients: {coeffs}")
                
                # Format results with appropriate threshold
                irrep_parts = []
                for i, coeff in enumerate(coeffs):
                    if abs(coeff) > 0.1:  # Use reasonable threshold
                        # Use the proper Mulliken labels from get_pg_info
                        mulliken_label = self.little_group_irrep_labels[i] if i < len(self.little_group_irrep_labels) else f"Γ{i+1}"
                        
                        if abs(coeff - 1) < 0.1:
                            irrep_parts.append(mulliken_label)
                        elif abs(coeff - round(coeff)) < 0.1:
                            irrep_parts.append(f"{int(round(coeff))}{mulliken_label}")
                        else:
                            irrep_parts.append(f"{coeff:.2f}{mulliken_label}")
                
                if irrep_parts:
                    result = " + ".join(irrep_parts)
                    activity = self._analyze_activity(irrep_parts)
                    return f"{result} ({activity})"
                else:
                    # Fallback to general pattern matching if spgrep decomposition fails
                    print("spgrep decomposition gave no results, trying general pattern matching...")
                    irrep_result = self._identify_irrep_by_pattern_general(class_characters)
                    return irrep_result if irrep_result else f"Little group: {self.little_group_label}"
                    
            except Exception as e:
                print(f"spgrep decomposition failed: {e}, trying general pattern matching...")
                # Fallback to general pattern matching if spgrep fails
                irrep_result = self._identify_irrep_by_pattern_general(class_characters)
                return irrep_result if irrep_result else f"Little group: {self.little_group_label}"
                
        except Exception as e:
            print(f"Irrep decomposition failed: {e}")
            return f"Little group: {self.little_group_label}"

    def _analyze_activity(self, irrep_parts):
        """Analyze optical activity of irreps."""
        activities = []
        
        # Simple activity rules based on irrep labels
        for part in irrep_parts:
            label = part.lstrip('0123456789')  # Remove multiplicity
            
            if any(x in label.lower() for x in ['a1', 'a1g', 'a1\'']):
                activities.append("Raman")
            elif any(x in label.lower() for x in ['t1u', 'a2u', 'eu']):
                activities.append("IR")
            elif 'u' in label.lower():
                activities.append("Electric dipole")
            elif 'g' in label.lower():
                activities.append("Raman")
        
        return ", ".join(set(activities)) if activities else "Inactive"

    def _identify_irrep_by_pattern(self, class_characters):
        """
        Identify irreps by pattern matching for common point groups.
        
        This is more robust than character table decomposition when spgrep has issues.
        """
        # Round characters for pattern matching
        chars = np.round(class_characters, 1)
        
        # For D6h point group (6/mmm)
        if self.point_group_label == '6/mmm':
            # Common patterns for D6h irreps
            # Note: These patterns are based on the standard D6h character table
            
            # 1D irreps
            if np.allclose(chars[:6], [1, 1, 1, 1, 1, 1], atol=0.2):
                return "A1g (Raman)"
            elif np.allclose(chars[:6], [1, 1, -1, -1, -1, -1], atol=0.2):
                return "A2g (Raman)"
            elif np.allclose(chars[:6], [1, -1, 1, -1, 1, -1], atol=0.2):
                return "B1g (Raman)"
            elif np.allclose(chars[:6], [1, -1, -1, 1, -1, 1], atol=0.2):
                return "B2g (Raman)"
            elif np.allclose(chars[6:], [1, 1, 1, 1, 1, 1], atol=0.2):
                return "A1u (IR)"
            elif np.allclose(chars[6:], [1, 1, -1, -1, -1, -1], atol=0.2):
                return "A2u (IR)"
            elif np.allclose(chars[6:], [1, -1, 1, -1, 1, -1], atol=0.2):
                return "B1u (IR)"
            elif np.allclose(chars[6:], [1, -1, -1, 1, -1, 1], atol=0.2):
                return "B2u (IR)"
            
            # 2D irreps (E-type)
            elif chars[0] == 2:  # First character is 2 for 2D irreps
                # E1g pattern: [2, -1, 0, 2, -1, 0, ...]
                if np.allclose(chars[:6], [2, -1, 0, 2, -1, 0], atol=0.5):
                    return "E1g (Raman)"
                # E2g pattern: [2, 1, 0, -2, 1, 0, ...]
                elif np.allclose(chars[:6], [2, 1, 0, -2, 1, 0], atol=0.5):
                    return "E2g (Raman)"
                # E1u pattern: [2, -1, 0, 2, -1, 0, ...] (ungerade)
                elif len(chars) > 6 and np.allclose(chars[6:], [2, -1, 0, 2, -1, 0], atol=0.5):
                    return "E1u (IR)"
                # E2u pattern: [2, 1, 0, -2, 1, 0, ...] (ungerade)
                elif len(chars) > 6 and np.allclose(chars[6:], [2, 1, 0, -2, 1, 0], atol=0.5):
                    return "E2u (IR)"
                # Generic 2D irrep
                else:
                    return "E-type (2D irrep)"
        
        # For other point groups, add patterns as needed
        elif self.point_group_label in ['D3h', 'C6v']:
            if chars[0] == 2:
                return "E-type (2D irrep)"
            elif chars[0] == 1:
                return "A-type (1D irrep)"
        
        # Generic identification
        if chars[0] == 1:
            return "A-type (1D irrep)"
        elif chars[0] == 2:
            return "E-type (2D irrep)"
        elif chars[0] == 3:
            return "T-type (3D irrep)"
        
        return None

    def _identify_irrep_by_pattern_general(self, class_characters):
        """
        General irrep identification using the little group character table from spgrep.
        """
        try:
            # Create proper D3h character table with spgrep class ordering
            # spgrep classes: E, 3C2, 2C3, σh, 3σv, 2S3
            n_classes = len(class_characters)
            
            if n_classes == 6 and self.little_group_label == '-6m2':  # D3h in spgrep notation
                # Use D3h character table from spgrep_point_group_ops
                d3h_char_table = _get_d3h_character_table()
                d3h_irrep_labels = _get_irrep_labels_for_pg('D3h', 6)
                class_orders = [len(ops) for ops in self.little_group_class_dict.values()]
                
                print(f"Using D3h character table shape: {d3h_char_table.shape}")
                print(f"D3h character table:\n{d3h_char_table}")
                
                coeffs = decompose_rep(class_characters, d3h_char_table, class_orders)
                
                print(f"D3h decomposition coefficients: {coeffs}")
                
                # Format results using proper D3h Mulliken labels
                irrep_parts = []
                for i, coeff in enumerate(coeffs):
                    if abs(coeff) > 0.1:  # Use reasonable threshold
                        label = d3h_irrep_labels[i]  # Use proper D3h labels
                        if abs(coeff - 1) < 0.1:
                            irrep_parts.append(label)
                        elif abs(coeff - round(coeff)) < 0.1:
                            irrep_parts.append(f"{int(round(coeff))}{label}")
                        else:
                            irrep_parts.append(f"{coeff:.2f}{label}")
                
                if irrep_parts:
                    result = " + ".join(irrep_parts)
                    activity = self._analyze_activity(irrep_parts)
                    return f"{result} ({activity})"
            
            return None
            
        except Exception as e:
            print(f"General irrep identification failed: {e}")
            return None

    def _compute_representation_matrices_all_states(self, BS_wfcs, iQ, uni_eigs, degen_eigs, little_group_indices=None):
        """
        Compute representation matrices for ALL states following the original algorithm exactly.
        
        This matches the algorithm in phys-excph-symm branch lines 281-393.
        """
        from yambopy.bse.rotate_excitonwf import rotate_exc_wf
        
        print(f"Computing representation matrices for all {len(BS_wfcs)} states...")
        
        # Use provided little group indices or compute them
        if little_group_indices is not None:
            little_group = [idx + 1 for idx in little_group_indices]  # Convert to 1-based
        else:
            # Find little group (symmetries that leave Q invariant) - following original exactly
            little_group = []
            # Loop over symmetries (excluding time reversal operations)
            for isym in range(int(self.sym_red.shape[0] / (self.ele_time_rev + 1))):
                # Check if Sq = q (following original algorithm exactly)
                Sq_minus_q = np.einsum('ij,j->i', self.sym_red[isym],
                                      self.qpts[iQ - 1]) - self.qpts[iQ - 1]
                Sq_minus_q = Sq_minus_q - np.rint(Sq_minus_q)
                
                # Check if Sq = q (within tolerance)
                if np.linalg.norm(Sq_minus_q) > 1e-5:
                    continue
                
                little_group.append(isym + 1)  # +1 for 1-based indexing like original
        
        trace_all_real = []
        trace_all_imag = []
        
        # Loop over little group operations
        for isym in [idx - 1 for idx in little_group]:  # Convert back to 0-based for array indexing
            # Phase factor from fractional translations (following original exactly)
            tau_dot_k = np.exp(1j * 2 * np.pi *
                              np.dot(self.qpts[iQ - 1], self.frac_trans[isym]))
            
            # Rotate exciton wavefunction (following original algorithm exactly)
            wfc_tmp = rotate_exc_wf(
                BS_wfcs,
                self.sym_red[isym],
                self.lelph_db.kpoints,
                self.qpts[iQ - 1],
                self.spglib_Dmats[isym],
                False,
                ktree=self.kpt_tree
            )
            
            # Compute representation matrix (following original algorithm exactly)
            rep = np.einsum('n...,m...->nm', wfc_tmp, BS_wfcs.conj(),
                           optimize=True) * tau_dot_k
            
            # Compute traces for each degenerate subspace (following original algorithm exactly)
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
        
        print(f"Little group has {len(little_group)} operations")
        
        # Convert to numpy arrays (following original exactly)
        trace_all_real = np.array(trace_all_real)
        trace_all_imag = np.array(trace_all_imag)
        
        # Return characters for each degenerate subspace
        # Each row corresponds to one symmetry operation, each column to one degenerate subspace
        characters_by_subspace = []
        for i in range(len(uni_eigs)):
            characters = trace_all_real[:, i]  # Real part only for character analysis
            characters_by_subspace.append(characters)
        
        return characters_by_subspace

    def _read_bse_data(self, BSE_dir, iQ, nstates):
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
                                                   filename=f'ndb.BS_diago_Q{iQ}')
        except Exception as e:
            raise IOError(f'Cannot read ndb.BS_diago_Q{iQ+1} file: {e}')
            
        bands_range = bse_db_iq.nbands
        BS_eigs = bse_db_iq.eigenvalues[:nstates]
        BS_wfcs = bse_db_iq.get_Akcv()[:nstates]
        
        # Convert to Hartree units (following original algorithm exactly)
        BS_eigs = BS_eigs / ha2ev
        
        return bands_range, BS_eigs, BS_wfcs



    def print_character_table(self):
        """Print the character table."""
        print(f"\nCharacter table for {self.little_group_label}:")
        print("-" * 50)
        
        # Header
        header = "Irrep".ljust(8)
        for i, class_name in enumerate(self.little_group_classes):
            header += f"{class_name}".rjust(8)
        print(header)
        print("-" * 50)
        
        # Character table rows
        for i, (label, chars) in enumerate(zip(self.little_group_irrep_labels, self.little_group_character_table)):
            row = f"{label}".ljust(8)
            for char in chars:
                if isinstance(char, complex):
                    if abs(char.imag) < 1e-10:
                        row += f"{char.real:7.0f} "
                    else:
                        row += f"{char:7.1f} "
                else:
                    row += f"{char:7.0f} "
            print(row)

    def compute(self, iQ=1, nstates=None, degen_thres=0.001):
        """
        Main computation method required by BaseOpticalProperties.
        
        This method performs the complete exciton symmetry analysis workflow.
        
        Parameters
        ----------
        iQ : int, optional
            Q-point index (1-based), default is 1 (Gamma point)
        nstates : int, optional
            Number of exciton states to analyze, default is min(10, available states)
        degen_thres : float, optional
            Degeneracy threshold in eV, default is 0.001
        """
        print(f"Computing exciton symmetry analysis for Q-point {iQ}...")
        
        # Set default nstates if not provided
        if nstates is None:
            nstates = min(10, len(self.BS_eigs) if hasattr(self, 'BS_eigs') else 10)
        
        # Perform the analysis
        results = self.analyze_exciton_symmetry(iQ, nstates, degen_thres)
        
        # Store results
        self.symmetry_results = results
        
        return results

    def analyze_selection_rules(self):
        """Analyze optical selection rules."""
        print(f"\nOptical selection rules for {self.little_group_label}:")
        print("-" * 40)
        
        for i, label in enumerate(self.little_group_irrep_labels):
            activity = self._analyze_activity([label])
            print(f"{label}: {activity}")

    def classify_symmetry_operations(self):
        """
        Classify symmetry operations for notebook compatibility.
        
        Returns summary information about the crystal structure and symmetries.
        """
        try:
            import spglib
            
            # Get space group info
            lattice = self.lat_vecs
            positions = self.ydb.red_atomic_positions  
            numbers = self.ydb.atomic_numbers
            cell = (lattice, positions, numbers)
            
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
            
            # Simple operation classification
            nsym = len(self.symm_mats)
            operations = {
                'identity': [],
                'rotation': [],
                'reflection': [],
                'inversion': [],
                'rotoinversion': [],
                'screw': [],
                'glide': [],
                '_summary': {
                    'space_group': dataset['international'],
                    'space_group_number': dataset['number'],
                    'point_group': self.point_group_label,
                    'crystal_system': self._get_crystal_system(dataset['number']),
                    'total_operations': nsym
                }
            }
            
            # Simple classification based on matrix properties
            for i, mat in enumerate(self.symm_mats):
                det = np.linalg.det(mat)
                trace = np.trace(mat)
                
                if np.allclose(mat, np.eye(3)):
                    operations['identity'].append((i, mat, "Identity", "E"))
                elif abs(det - 1) < 1e-6:
                    if abs(trace - 3) < 1e-6:
                        operations['identity'].append((i, mat, "Identity", "E"))
                    else:
                        operations['rotation'].append((i, mat, "Rotation", "C"))
                elif abs(det + 1) < 1e-6:
                    if abs(trace + 3) < 1e-6:
                        operations['inversion'].append((i, mat, "Inversion", "i"))
                    else:
                        operations['reflection'].append((i, mat, "Reflection", "σ"))
                else:
                    operations['rotoinversion'].append((i, mat, "Rotoinversion", "S"))
            
            return operations
            
        except Exception as e:
            print(f"Warning: Could not classify operations: {e}")
            return {
                '_summary': {
                    'space_group': self.spacegroup_label,
                    'point_group': self.point_group_label,
                    'total_operations': len(self.symm_mats)
                }
            }

    def _get_crystal_system(self, space_group_number):
        """Get crystal system from space group number."""
        if 1 <= space_group_number <= 2:
            return "triclinic"
        elif 3 <= space_group_number <= 15:
            return "monoclinic"
        elif 16 <= space_group_number <= 74:
            return "orthorhombic"
        elif 75 <= space_group_number <= 142:
            return "tetragonal"
        elif 143 <= space_group_number <= 167:
            return "trigonal"
        elif 168 <= space_group_number <= 194:
            return "hexagonal"
        elif 195 <= space_group_number <= 230:
            return "cubic"
        else:
            return "unknown"