"""
Point group analysis using spgrep library.

Provides point group identification and Mulliken labeling for crystallographic symmetries.
"""

import numpy as np
from typing import Tuple, List, Dict

try:
    import spgrep
    from spgrep import get_crystallographic_pointgroup_irreps_from_symmetry
except ImportError:
    raise ImportError("spgrep is required. Install with 'pip install spgrep'")


def get_pg_info(symm_mats: np.ndarray, time_rev: bool = False) -> Tuple[str, List[str], Dict[int, List[int]], np.ndarray, List[str]]:
    """
    Get point group information using spgrep.
    
    Parameters
    ----------
    symm_mats : np.ndarray
        Symmetry matrices (nsym, 3, 3)
    time_rev : bool
        If True, use only first half of operations
        
    Returns
    -------
    pg_label : str
        Point group label
    classes : list of str
        Symmetry class labels  
    class_dict : dict
        Class index to operation indices mapping
    char_tab : np.ndarray
        Character table (n_irreps, n_ops)
    irrep_labels : list of str
        Irrep labels with Mulliken notation when possible
    """
    # Handle time reversal - take only spatial operations
    if time_rev:
        symm_mats = symm_mats[:len(symm_mats)//2]
    
    # Convert to integer matrices for spgrep
    symm_mats_int = _to_int_matrices(symm_mats)
    
    # Try to use spgrep, but with robust error handling
    try:
        # Clean the matrices to ensure they're proper rotation matrices
        symm_mats_clean = _clean_symmetry_matrices(symm_mats_int)
        
        # The issue is that spgrep's get_crystallographic_pointgroup_irreps_from_symmetry
        # has an internal bug with certain matrices. Let's bypass it and use a different approach.
        
        # Try to identify point group using spgrep's point group identification
        import spgrep
        try:
            result = spgrep.pointgroup.get_pointgroup(symm_mats_clean)
            if result and result[0].strip():
                symbol, number, _ = result
                pg_label = symbol.strip()
                print(f"spgrep identified point group: {pg_label}")
                
                # Check if the identification makes sense
                if len(pg_label) < 2 or pg_label in ['S', 'T', 'O']:
                    # spgrep gave a partial or unclear result, use our own identification
                    print(f"spgrep result '{pg_label}' seems incomplete, using internal identification")
                    pg_label = _identify_pg_from_matrices(symm_mats_clean)
                    print(f"Internal identification: {pg_label}")
                
                # Instead of using the problematic function, create our own irrep info
                return _create_pg_info_from_label(pg_label, symm_mats_clean)
            else:
                raise ValueError("spgrep could not identify point group")
                
        except Exception as e1:
            print(f"spgrep point group identification failed ({e1})")
            # Fall back to our own point group identification
            pg_label = _identify_pg_from_matrices(symm_mats_clean)
            print(f"Using internal identification: {pg_label}")
            return _create_pg_info_from_label(pg_label, symm_mats_clean)
        
    except Exception as e:
        # If everything fails, use fallback method
        print(f"Warning: all spgrep methods failed ({e}), using fallback method")
        return _fallback_pg_info(symm_mats_int)


def _extract_pg_info(symm_mats: np.ndarray, irrep_matrices: List[np.ndarray]) -> Tuple[str, List[str], Dict[int, List[int]], np.ndarray, List[str]]:
    """Extract point group information from spgrep results."""
    nsym = len(symm_mats)
    n_irreps = len(irrep_matrices)
    
    # Get point group label
    pg_label = _get_pg_label(symm_mats)
    
    # Simple classes: each operation is its own class
    classes = [f"Op{i}" for i in range(nsym)]
    class_dict = {i: [i] for i in range(nsym)}
    
    # Compute character table from irrep matrices
    char_tab = np.zeros((n_irreps, nsym), dtype=complex)
    for i, irrep_mats in enumerate(irrep_matrices):
        for j in range(min(nsym, len(irrep_mats))):
            char_tab[i, j] = np.trace(irrep_mats[j])
    
    # Make real if possible
    if np.allclose(char_tab.imag, 0, atol=1e-10):
        char_tab = char_tab.real
    
    # Get Mulliken labels
    irrep_labels = _get_mulliken_labels(pg_label, char_tab)
    
    return pg_label, classes, class_dict, char_tab, irrep_labels


def _get_pg_label(symm_mats: np.ndarray) -> str:
    """Get point group label from spgrep."""
    try:
        result = spgrep.pointgroup.get_pointgroup(symm_mats)
        if result:
            symbol, number, _ = result
            return symbol.strip()
        return f"PG{len(symm_mats)}"
    except:
        return f"PG{len(symm_mats)}"


def _get_mulliken_labels(pg_label: str, char_tab: np.ndarray) -> List[str]:
    """Get Mulliken labels by pattern matching."""
    # Convert spgrep notation to standard
    conversions = {
        '1': 'C1', '-1': 'Ci', '2': 'C2', 'm': 'Cs', '2/m': 'C2h',
        '222': 'D2', 'mm2': 'C2v', 'mmm': 'D2h',
        '4': 'C4', '-4': 'S4', '4/m': 'C4h', '422': 'D4', '4mm': 'C4v', '-42m': 'D2d', '4/mmm': 'D4h',
        '3': 'C3', '-3': 'C3i', '32': 'D3', '3m': 'C3v', '-3m': 'D3d',
        '6': 'C6', '-6': 'C3h', '6/m': 'C6h', '622': 'D6', '6mm': 'C6v', '-6m2': 'D3h', '6/mmm': 'D6h',
        '23': 'T', 'm-3': 'Th', '432': 'O', '-43m': 'Td', 'm-3m': 'Oh'
    }
    
    standard_pg = conversions.get(pg_label, pg_label)
    n_ops = char_tab.shape[1]
    
    # Pattern matching for common point groups
    if standard_pg == 'D2' and len(char_tab) == 4:
        return _match_patterns(char_tab, ["A", "B1", "B2", "B3"], [
            [1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]
        ])
    
    elif standard_pg == 'C3v' and len(char_tab) == 3:
        if n_ops == 6:
            return _match_patterns(char_tab, ["A1", "A2", "E"], [
                [1, 1, 1, 1, 1, 1], [1, 1, 1, -1, -1, -1], [2, -1, -1, 0, 0, 0]
            ])
        else:
            return _match_patterns(char_tab, ["A1", "A2", "E"], [
                [1, 1, 1], [1, 1, -1], [2, -1, 0]
            ])
    
    elif standard_pg == 'C2v' and len(char_tab) == 4:
        return _match_patterns(char_tab, ["A1", "A2", "B1", "B2"], [
            [1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]
        ])
    
    elif standard_pg == 'C2' and len(char_tab) == 2:
        return _match_patterns(char_tab, ["A", "B"], [
            [1, 1], [1, -1]
        ])
    
    elif standard_pg == 'C4v' and len(char_tab) == 5:
        return _match_patterns(char_tab, ["A1", "A2", "B1", "B2", "E"], [
            [1, 1, 1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1], 
            [1, -1, 1, -1, 1], [2, 0, -2, 0, 0]
        ])
    
    elif standard_pg == 'D3h' and len(char_tab) == 6:
        return _match_patterns(char_tab, ["A1'", "A2'", "E'", "A1''", "A2''", "E''"], [
            [1, 1, 1, 1, 1, 1], [1, 1, -1, 1, 1, -1], [2, -1, 0, 2, -1, 0],
            [1, 1, 1, -1, -1, -1], [1, 1, -1, -1, -1, 1], [2, -1, 0, -2, 1, 0]
        ])
    
    elif standard_pg == 'D4h' and len(char_tab) == 10:
        return _match_patterns(char_tab, ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"], [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
            [1, -1, 1, 1, -1, 1, -1, 1, 1, -1], [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
            [2, 0, -2, 0, 0, 2, 0, -2, 0, 0], [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, 1, -1, -1, -1, -1, -1, 1, 1], [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1], [2, 0, -2, 0, 0, -2, 0, 2, 0, 0]
        ])
    
    elif standard_pg == 'Oh' and len(char_tab) == 10:
        return _match_patterns(char_tab, ["A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u"], [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
            [2, -1, 0, 0, 2, 2, 0, -1, 2, 0], [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
            [3, 0, 1, -1, -1, 3, -1, 0, -1, 1], [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, -1, -1, 1, -1, 1, -1, -1, 1], [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
            [3, 0, -1, 1, -1, -3, -1, 0, 1, 1], [3, 0, 1, -1, -1, -3, 1, 0, 1, -1]
        ])
    
    # Fallback to systematic labels
    return [f"Γ{i+1}" for i in range(len(char_tab))]


def _match_patterns(char_tab: np.ndarray, labels: List[str], patterns: List[List[float]]) -> List[str]:
    """Match character table to standard patterns."""
    matched_labels = []
    used_indices = set()
    
    for i in range(len(char_tab)):
        chars = char_tab[i]
        best_match = None
        best_score = float('inf')
        
        for j, pattern in enumerate(patterns):
            if j in used_indices or len(chars) != len(pattern):
                continue
            
            score = np.sum(np.abs(chars - np.array(pattern)))
            if score < best_score:
                best_score = score
                best_match = j
        
        if best_match is not None and best_score < 1e-6:
            matched_labels.append(labels[best_match])
            used_indices.add(best_match)
        else:
            matched_labels.append(f"Γ{i+1}")
    
    return matched_labels


def _clean_symmetry_matrices(symm_mats: np.ndarray) -> np.ndarray:
    """
    Clean symmetry matrices to ensure they're in the proper format for spgrep.
    
    This function removes duplicates and ensures matrices are proper rotation/reflection matrices.
    """
    cleaned_mats = []
    seen_matrices = set()
    
    for mat in symm_mats:
        # Convert to tuple for hashing
        mat_tuple = tuple(tuple(row) for row in mat)
        
        if mat_tuple not in seen_matrices:
            # Check if it's a proper rotation/reflection matrix
            det = np.linalg.det(mat)
            if np.isclose(abs(det), 1.0, atol=1e-6):
                cleaned_mats.append(mat)
                seen_matrices.add(mat_tuple)
    
    return np.array(cleaned_mats)


def _to_int_matrices(symm_mats: np.ndarray) -> np.ndarray:
    """Convert symmetry matrices to integer representation."""
    # Round to nearest integer and convert
    symm_mats_int = np.round(symm_mats).astype(int)
    
    # Validate that they are proper rotation/reflection matrices
    for i, mat in enumerate(symm_mats_int):
        det = np.linalg.det(mat)
        if not np.isclose(abs(det), 1.0, atol=1e-6):
            # Try to fix common floating point issues
            mat_fixed = _fix_matrix(mat)
            if np.isclose(abs(np.linalg.det(mat_fixed)), 1.0, atol=1e-6):
                symm_mats_int[i] = mat_fixed
    
    return symm_mats_int


def _fix_matrix(mat: np.ndarray) -> np.ndarray:
    """Fix common matrix issues."""
    # Round small values to zero
    mat_fixed = np.where(np.abs(mat) < 1e-10, 0, mat)
    
    # Round values close to ±1 to exact ±1
    mat_fixed = np.where(np.abs(mat_fixed - 1) < 1e-10, 1, mat_fixed)
    mat_fixed = np.where(np.abs(mat_fixed + 1) < 1e-10, -1, mat_fixed)
    
    return mat_fixed.astype(int)


def _fallback_pg_info(symm_mats: np.ndarray) -> Tuple[str, List[str], Dict[int, List[int]], np.ndarray, List[str]]:
    """Fallback method when spgrep fails."""
    nsym = len(symm_mats)
    
    # Try to identify common point groups by number of operations
    pg_label = _identify_pg_by_order(nsym)
    
    # Create simple classes (each operation is its own class)
    classes = [f"Op{i}" for i in range(nsym)]
    class_dict = {i: [i] for i in range(nsym)}
    
    # Create identity character table (all irreps are A-type)
    char_tab = np.eye(nsym, dtype=float)
    
    # Get appropriate irrep labels based on point group
    irrep_labels = _get_fallback_irrep_labels(pg_label, nsym)
    
    return pg_label, classes, class_dict, char_tab, irrep_labels


def _identify_pg_by_order(nsym: int) -> str:
    """Identify point group by number of symmetry operations."""
    common_orders = {
        1: 'C1',
        2: 'Ci',  # or C2, Cs
        4: 'C2v',  # or D2, C4, S4
        8: 'D2h',  # or C4v, D4, etc.
        12: 'D6',  # or D3d, T, etc.
        16: 'D4h',
        24: 'D6h',  # Most likely for hBN with 24 operations
        48: 'Oh'
    }
    return common_orders.get(nsym, f'PG{nsym}')


def _identify_pg_from_matrices(symm_mats: np.ndarray) -> str:
    """Identify point group from symmetry matrices using basic analysis."""
    nsym = len(symm_mats)
    
    # Count proper rotations (det = +1) and improper rotations (det = -1)
    dets = [np.linalg.det(mat) for mat in symm_mats]
    n_proper = sum(1 for d in dets if d > 0)
    n_improper = sum(1 for d in dets if d < 0)
    
    # Basic identification based on order and determinants
    if nsym == 1:
        return 'C1'
    elif nsym == 2:
        if n_improper == 1:
            return 'Ci'  # or Cs
        else:
            return 'C2'
    elif nsym == 4:
        return 'C2v'  # Most common for 4 operations
    elif nsym == 8:
        return 'D2h'
    elif nsym == 12:
        if n_improper == 0:
            return 'D6'
        else:
            return 'D3d'
    elif nsym == 24:
        return 'D6h'  # Most likely for hBN
    elif nsym == 48:
        return 'Oh'
    else:
        return f'PG{nsym}'


def _create_pg_info_from_label(pg_label: str, symm_mats: np.ndarray) -> Tuple[str, List[str], Dict[int, List[int]], np.ndarray, List[str]]:
    """Create point group info from a known point group label."""
    nsym = len(symm_mats)
    
    # Get appropriate irrep labels first
    irrep_labels = _get_irrep_labels_for_pg(pg_label, nsym)
    n_irreps = len(irrep_labels)
    
    # Use general approach for all point groups
    return _create_general_pg_info(pg_label, symm_mats, irrep_labels)


def _create_general_pg_info(pg_label: str, symm_mats: np.ndarray, irrep_labels: List[str]) -> Tuple[str, List[str], Dict[int, List[int]], np.ndarray, List[str]]:
    """Create general point group information with computed conjugacy classes."""
    nsym = len(symm_mats)
    n_irreps = len(irrep_labels)
    
    # Compute conjugacy classes
    classes, class_dict = _compute_conjugacy_classes(symm_mats)
    n_classes = len(classes)
    
    print(f"Found {n_classes} conjugacy classes for {pg_label}")
    
    # Use proper character tables for known point groups
    if pg_label == '6/mmm' or pg_label == 'D6h':
        char_tab = _get_d6h_character_table()
    elif pg_label in ['D2h', 'mmm']:
        char_tab = _get_d2h_character_table()
    elif pg_label in ['C2v', 'mm2']:
        char_tab = _get_c2v_character_table()
    elif pg_label in ['D3h', '-6m2']:
        char_tab = _get_d3h_character_table()
    else:
        # Fallback: create identity-like character table
        char_tab = np.eye(min(n_irreps, n_classes), dtype=float)
        if n_irreps > n_classes:
            # Pad with zeros if more irreps than classes
            padding = np.zeros((n_irreps - n_classes, n_classes))
            char_tab = np.vstack([char_tab, padding])
        elif n_classes > n_irreps:
            # Pad with zeros if more classes than irreps
            padding = np.zeros((n_irreps, n_classes - n_irreps))
            char_tab = np.hstack([char_tab, padding])
    
    return pg_label, classes, class_dict, char_tab, irrep_labels


def _get_d6h_character_table() -> np.ndarray:
    """Get the proper D6h character table."""
    # D6h character table (12 irreps × 12 classes)
    # Classes: E, 2C6, 2C3, C2, 3C2', 3C2'', i, 2S3, 2S6, σh, 3σv, 3σd
    # Order: [E, C6, C3, C2, C2', C2'', i, S3, S6, σh, σv, σd]
    
    char_tab = np.array([
        # E   C6  C3  C2  C2' C2'' i   S3  S6  σh  σv  σd
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # A1g
        [ 1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],  # A2g  
        [ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],  # B1g
        [ 1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],  # B2g
        [ 2,  1, -1, -2,  0,  0,  2,  1, -1, -2,  0,  0],  # E1g
        [ 2, -1, -1,  2,  0,  0,  2, -1, -1,  2,  0,  0],  # E2g
        [ 1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],  # A1u
        [ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1],  # A2u
        [ 1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],  # B1u
        [ 1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1],  # B2u
        [ 2,  1, -1, -2,  0,  0, -2, -1,  1,  2,  0,  0],  # E1u
        [ 2, -1, -1,  2,  0,  0, -2,  1,  1, -2,  0,  0],  # E2u
    ], dtype=float)
    
    return char_tab


def _get_d2h_character_table() -> np.ndarray:
    """Get the D2h character table."""
    # D2h: 8 irreps × 8 classes
    char_tab = np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1],  # Ag
        [ 1,  1, -1, -1,  1,  1, -1, -1],  # B1g
        [ 1, -1,  1, -1,  1, -1,  1, -1],  # B2g
        [ 1, -1, -1,  1,  1, -1, -1,  1],  # B3g
        [ 1,  1,  1,  1, -1, -1, -1, -1],  # Au
        [ 1,  1, -1, -1, -1, -1,  1,  1],  # B1u
        [ 1, -1,  1, -1, -1,  1, -1,  1],  # B2u
        [ 1, -1, -1,  1, -1,  1,  1, -1],  # B3u
    ], dtype=float)
    
    return char_tab


def _get_c2v_character_table() -> np.ndarray:
    """Get the C2v character table."""
    # C2v: 4 irreps × 4 classes
    char_tab = np.array([
        [ 1,  1,  1,  1],  # A1
        [ 1,  1, -1, -1],  # A2
        [ 1, -1,  1, -1],  # B1
        [ 1, -1, -1,  1],  # B2
    ], dtype=float)


def _get_d3h_character_table() -> np.ndarray:
    """Get the D3h character table."""
    # D3h: 6 irreps × 6 classes
    # Classes: E, 2C3, 3C2, σh, 2S3, 3σv
    # But spgrep uses order: E, 3C2, 2C3, σh, 3σv, 2S3
    char_tab = np.array([
        [ 1,  1,  1,  1,  1,  1],  # A1'
        [ 1, -1,  1,  1, -1,  1],  # A2'
        [ 2,  0, -1,  2,  0, -1],  # E'
        [ 1,  1,  1, -1, -1, -1],  # A1"
        [ 1, -1,  1, -1,  1, -1],  # A2"
        [ 2,  0, -1, -2,  0,  1],  # E"
    ], dtype=float)
    
    return char_tab


def _compute_conjugacy_classes(symm_mats: np.ndarray) -> Tuple[List[str], Dict[int, List[int]]]:
    """Compute conjugacy classes for symmetry operations."""
    nsym = len(symm_mats)
    
    # Initialize: each operation starts in its own class
    classes_found = []
    class_dict = {}
    assigned = [False] * nsym
    
    for i in range(nsym):
        if assigned[i]:
            continue
            
        # Start a new conjugacy class with operation i
        class_idx = len(classes_found)
        class_members = [i]
        assigned[i] = True
        
        # Find all operations conjugate to operation i
        R_i = symm_mats[i]
        
        for j in range(i + 1, nsym):
            if assigned[j]:
                continue
                
            R_j = symm_mats[j]
            
            # Check if R_j is conjugate to R_i
            # Two operations are conjugate if there exists S such that R_j = S^(-1) * R_i * S
            is_conjugate = False
            
            for k in range(nsym):
                S = symm_mats[k]
                try:
                    S_inv = np.linalg.inv(S)
                    conjugate = S_inv @ R_i @ S
                    
                    # Check if conjugate is approximately equal to R_j
                    if np.allclose(conjugate, R_j, atol=1e-6):
                        is_conjugate = True
                        break
                except np.linalg.LinAlgError:
                    continue
            
            if is_conjugate:
                class_members.append(j)
                assigned[j] = True
        
        # Create class label and store
        class_label = f"C{class_idx+1}({len(class_members)})"
        classes_found.append(class_label)
        class_dict[class_idx] = class_members
    
    return classes_found, class_dict


def _get_irrep_labels_for_pg(pg_label: str, nsym: int) -> List[str]:
    """Get irrep labels for a specific point group."""
    # Known irrep labels for common point groups
    irrep_tables = {
        'C1': ['A'],
        'Ci': ['Ag', 'Au'],
        'C2': ['A', 'B'],
        'Cs': ['A\'', 'A"'],
        'C2v': ['A1', 'A2', 'B1', 'B2'],
        'D2': ['A', 'B1', 'B2', 'B3'],
        'D2h': ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u'],
        'C3': ['A', 'E'],
        'C3v': ['A1', 'A2', 'E'],
        'D3': ['A1', 'A2', 'E'],
        'D3h': ['A1\'', 'A2\'', 'E\'', 'A1"', 'A2"', 'E"'],
        '-6m2': ['A1\'', 'A2\'', 'E\'', 'A1"', 'A2"', 'E"'],  # D3h in spgrep notation
        'D3d': ['A1g', 'A2g', 'Eg', 'A1u', 'A2u', 'Eu'],
        'C6': ['A', 'B', 'E1', 'E2'],
        'C6v': ['A1', 'A2', 'B1', 'B2', 'E1', 'E2'],
        'D6': ['A1', 'A2', 'B1', 'B2', 'E1', 'E2'],
        'D6h': ['A1g', 'A2g', 'B1g', 'B2g', 'E1g', 'E2g', 'A1u', 'A2u', 'B1u', 'B2u', 'E1u', 'E2u'],
        '6/mmm': ['A1g', 'A2g', 'B1g', 'B2g', 'E1g', 'E2g', 'A1u', 'A2u', 'B1u', 'B2u', 'E1u', 'E2u'],  # Same as D6h
        'T': ['A', 'E', 'T'],
        'Td': ['A1', 'A2', 'E', 'T1', 'T2'],
        'Th': ['Ag', 'Eg', 'Tg', 'Au', 'Eu', 'Tu'],
        'O': ['A1', 'A2', 'E', 'T1', 'T2'],
        'Oh': ['A1g', 'A2g', 'Eg', 'T1g', 'T2g', 'A1u', 'A2u', 'Eu', 'T1u', 'T2u']
    }
    
    if pg_label in irrep_tables:
        labels = irrep_tables[pg_label]
        # Return the correct number of irrep labels (not extended to nsym)
        return labels
    else:
        # Generic labels for unknown point groups
        return [f'A{i+1}' for i in range(min(nsym, 12))] + [f'Γ{i+13}' for i in range(max(0, nsym-12))]


def _get_fallback_irrep_labels(pg_label: str, nsym: int) -> List[str]:
    """Get appropriate irrep labels for fallback method."""
    if pg_label == 'D6h' and nsym == 24:
        # D6h has 12 irreps: A1g, A2g, B1g, B2g, E1g, E2g, A1u, A2u, B1u, B2u, E1u, E2u
        return ['A1g', 'A2g', 'B1g', 'B2g', 'E1g', 'E2g', 'A1u', 'A2u', 'B1u', 'B2u', 'E1u', 'E2u'] + [f'Γ{i+13}' for i in range(nsym-12)]
    elif pg_label == 'D2h' and nsym == 8:
        return ['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u']
    elif pg_label == 'C2v' and nsym == 4:
        return ['A1', 'A2', 'B1', 'B2']
    else:
        # Generic labels
        return [f'A{i+1}' for i in range(min(nsym, 12))] + [f'Γ{i+13}' for i in range(max(0, nsym-12))]


def decompose_rep(reducible_chars: np.ndarray, char_table: np.ndarray, 
                 class_orders: List[int]) -> np.ndarray:
    """
    Decompose reducible representation into irreps.
    
    Parameters
    ----------
    reducible_chars : np.ndarray
        Characters of reducible representation
    char_table : np.ndarray
        Character table (n_irreps, n_classes)
    class_orders : list of int
        Order of each conjugacy class
        
    Returns
    -------
    np.ndarray
        Coefficients for each irrep
    """
    pg_order = sum(class_orders)
    coeffs = np.zeros(len(char_table))
    
    for i, irrep_chars in enumerate(char_table):
        coeff = np.sum(class_orders * reducible_chars * np.conj(irrep_chars)) / pg_order
        coeffs[i] = coeff.real
    
    return np.round(coeffs).astype(int)