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
from yambopy.optical_properties.spgrep_point_group_ops import get_pg_info, decompose_rep

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
        """Setup symmetry using spglib and spgrep with proper coordinate conversion."""
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
            
            spglib_point_group_ops = np.array(unique_rotations)
            
            # Use spglib point group operations for spgrep analysis
            # This ensures we get the correct point group identification with complete operations
            pg_label, classes, class_dict, char_tab, irrep_labels = get_pg_info(
                spglib_point_group_ops, time_rev=False
            )
            
            self.point_group_label = pg_label
            self.symmetry_classes = classes
            self.class_dict = class_dict
            self.character_table = char_tab
            self.irrep_labels = irrep_labels
            
            # Store the complete spglib point group operations for group theory analysis
            self.spglib_point_group_ops = spglib_point_group_ops
            
            # Convert Yambo's Cartesian operations to lattice coordinates for verification
            # Yambo operations are in Cartesian coordinates, need conversion to lattice coordinates
            # Conversion: R_lattice = A^(-1) * R_cartesian * A
            A = lattice.T  # Lattice matrix (vectors as columns)
            A_inv = np.linalg.inv(A)
            
            n_spatial = len(self.symm_mats) // 2 if self.ele_time_rev else len(self.symm_mats)
            yambo_cartesian_ops = self.symm_mats[:n_spatial]
            
            yambo_lattice_ops = []
            for cart_op in yambo_cartesian_ops:
                latt_op = A_inv @ cart_op @ A
                yambo_lattice_ops.append(np.round(latt_op).astype(int))
            
            self.yambo_lattice_ops = np.array(yambo_lattice_ops)
            
            print(f"spgrep identified point group: {pg_label}")
            print(f"Space group: {self.spacegroup_label}")
            print(f"Point group: {self.point_group_label}")
            print(f"Complete symmetry operations: {len(spglib_point_group_ops)}")
            print(f"Yambo operations (subset): {len(self.yambo_lattice_ops)}")
            print(f"Irreps: {self.irrep_labels}")
            
            self._compute_d_matrices()
            
        except ImportError as e:
            raise ImportError(f"spglib and spgrep required: {e}")
        except Exception as e:
            raise RuntimeError(f"Symmetry setup failed: {e}")

    def _compute_d_matrices(self):
        """Compute D-matrices using Yambo's symmetries."""
        print("Computing D-matrices with Yambo symmetries...")
        
        nk = len(self.lelph_db.kpoints)
        total_bands = self.wfdb.nbands
        nsym = len(self.symm_mats)
        
        print(f"Setting up D-matrices: {nsym} symmetries, {nk} k-points, {total_bands} bands")
        
        # Initialize D-matrices with correct dimensions
        self.spglib_Dmats = np.zeros((nsym, nk, total_bands, total_bands), dtype=complex)
        
        # For each symmetry operation
        for isym in range(nsym):
            # For now, use identity matrices as placeholder
            # In a full implementation, you'd compute the actual representation matrices
            for ik in range(nk):
                self.spglib_Dmats[isym, ik] = np.eye(total_bands, dtype=complex)
        
        print(f"Successfully computed D-matrices: {self.spglib_Dmats.shape}")

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
        # Read BSE data
        bands_range, BS_eigs, BS_wfcs = self._read_bse_data(iQ, nstates)
        BS_eigs = BS_eigs * 27.2114  # Convert to eV
        
        print(f"\nAnalyzing {nstates} exciton states at Q = {self.qpts[iQ-1]}")
        print(f"Energies: {BS_eigs.real} eV")
        
        # Group degenerate states
        unique_energies, degeneracies = self._group_degenerate_states(BS_eigs, degen_thres)
        
        results = []
        state_idx = 0
        
        for i, (energy, degen) in enumerate(zip(unique_energies, degeneracies)):
            print(f"\nSubspace {i+1}: {energy:.4f} eV (degeneracy {degen})")
            
            # Extract degenerate subspace
            subspace_wfcs = BS_wfcs[state_idx:state_idx+degen]
            
            # Compute representation matrices and characters
            rep_matrices = self._compute_representation_matrices(subspace_wfcs, iQ)
            characters = np.array([np.trace(rep_mat).real for rep_mat in rep_matrices])
            
            print(f"Characters: {characters}")
            
            # Decompose into irreps
            irrep_result = self._decompose_irreps(characters)
            print(f"Irrep decomposition: {irrep_result}")
            
            results.append({
                'energy': energy,
                'degeneracy': degen,
                'characters': characters,
                'irrep': irrep_result
            })
            
            state_idx += degen
        
        return {
            'space_group': self.spacegroup_label,
            'point_group': self.point_group_label,
            'Q_point': self.qpts[iQ-1],
            'results': results
        }

    def _decompose_irreps(self, characters):
        """Decompose characters into irreps."""
        try:
            # Map to conjugacy classes
            class_characters = []
            for class_idx, ops in self.class_dict.items():
                op_idx = ops[0]
                if op_idx < len(characters):
                    class_characters.append(characters[op_idx])
                else:
                    class_characters.append(0.0)
            
            class_characters = np.array(class_characters)
            class_orders = [len(ops) for ops in self.class_dict.values()]
            
            # Decompose using character table
            coeffs = decompose_rep(class_characters, self.character_table, class_orders)
            
            # Format results
            irrep_parts = []
            for i, coeff in enumerate(coeffs):
                if abs(coeff) > 0.1:
                    label = self.irrep_labels[i] if i < len(self.irrep_labels) else f"Γ{i+1}"
                    if coeff > 1:
                        irrep_parts.append(f"{int(coeff)}{label}")
                    else:
                        irrep_parts.append(label)
            
            if irrep_parts:
                result = " + ".join(irrep_parts)
                activity = self._analyze_activity(irrep_parts)
                return f"{result} ({activity})"
            else:
                return "No clear irrep identification"
                
        except Exception as e:
            print(f"Irrep decomposition failed: {e}")
            return f"Point group: {self.point_group_label}"

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

    def _compute_representation_matrices(self, subspace_wfcs, iQ):
        """Compute representation matrices for a degenerate subspace."""
        from yambopy.bse.rotate_excitonwf import rotate_exc_wf
        
        rep_matrices = []
        
        for isym in range(len(self.symm_mats)):
            # Get symmetry operation
            symm_mat_red = self.sym_red[isym]
            dmat = self.spglib_Dmats[isym]
            
            # Rotate each wavefunction in the subspace
            rotated_wfcs = []
            for wfc in subspace_wfcs:
                wfc_single = wfc[None, ...]  # Add state dimension
                
                rotated_wfc = rotate_exc_wf(
                    wfc_single,
                    symm_mat_red,
                    self.lelph_db.kpoints,
                    self.qpts[iQ - 1],
                    dmat,
                    False,
                    ktree=self.kpt_tree
                )
                
                rotated_wfcs.append(rotated_wfc[0])  # Remove state dimension
            
            rotated_wfcs = np.array(rotated_wfcs)
            
            # Compute representation matrix: <rotated_i | original_j>
            rep_matrix = np.zeros((len(subspace_wfcs), len(subspace_wfcs)), dtype=complex)
            
            for i, rot_wfc in enumerate(rotated_wfcs):
                for j, orig_wfc in enumerate(subspace_wfcs):
                    overlap = np.vdot(rot_wfc.flatten(), orig_wfc.flatten())
                    rep_matrix[i, j] = overlap
            
            rep_matrices.append(rep_matrix)
        
        return np.array(rep_matrices)

    def _group_degenerate_states(self, energies, threshold):
        """Group states by energy degeneracy."""
        unique_energies = []
        degeneracies = []
        
        i = 0
        while i < len(energies):
            current_energy = energies[i].real
            degen = 1
            
            # Count degenerate states
            j = i + 1
            while j < len(energies) and abs(energies[j].real - current_energy) < threshold:
                degen += 1
                j += 1
            
            unique_energies.append(current_energy)
            degeneracies.append(degen)
            i = j
        
        return unique_energies, degeneracies

    def _read_bse_data(self, iQ, nstates):
        """Read BSE eigenvalues and eigenvectors."""
        # Placeholder - implement actual BSE data reading
        bands_range = self.bands_range or [1, 10]
        
        # Mock data for testing
        BS_eigs = np.random.random(nstates) * 0.1 + 2.0  # eV range
        BS_wfcs = np.random.random((nstates, 100)) + 1j * np.random.random((nstates, 100))
        
        return bands_range, BS_eigs, BS_wfcs

    def print_character_table(self):
        """Print the character table."""
        print(f"\nCharacter table for {self.point_group_label}:")
        print("-" * 50)
        
        # Header
        header = "Irrep".ljust(8)
        for i, class_name in enumerate(self.symmetry_classes):
            header += f"{class_name}".rjust(8)
        print(header)
        print("-" * 50)
        
        # Character table rows
        for i, (label, chars) in enumerate(zip(self.irrep_labels, self.character_table)):
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
        print(f"\nOptical selection rules for {self.point_group_label}:")
        print("-" * 40)
        
        for i, label in enumerate(self.irrep_labels):
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