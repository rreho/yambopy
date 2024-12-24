import sisl
import spglib
from yambopy.lattice import red_car
import numpy as np

class KPointGenerator():
    def __init__(self):
        self.k = None
        self.nkpoints = None
        self.weights = None
    
    def generate(self):
        """Abstract method to generate k-points."""
        raise NotImplementedError("This method must be implemented in subclasses.")

    def validate(self):
        """Validate the generated k-points."""
        if self.k is None or self.weights is None:
            raise ValueError("k-points or weights have not been generated.")

    def export(self, filename):
        """Export k-points to a file."""
        np.savetxt(filename, self.k, header="k-points")

    def get_kmq_grid(self, qmpgrid):

        #qmpgrid is meant to be an nnkp object
        kmq_grid = np.zeros((self.nkpoints, qmpgrid.nkpoints, 3))
        kmq_grid_table = np.zeros((self.nkpoints, qmpgrid.nkpoints, 5),dtype= int)
        for ik, k in enumerate(self.k):
            for iq, q in enumerate(qmpgrid.k):
                tmp_kmq, tmp_Gvec = self.fold_into_bz_Gs(k-q)
                idxkmq = self.find_closest_kpoint(tmp_kmq)
                kmq_grid[ik,iq] = tmp_kmq
                kmq_grid_table[ik,iq] = [ik, idxkmq, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kmq_grid = kmq_grid
        self.kmq_grid_table = kmq_grid_table

    def fold_into_bz_Gs(self, k_points, bz_range=(-0.5, 0.5), reciprocal_vectors=None):
        """
        Fold k-points into the first Brillouin Zone and determine the reciprocal lattice vectors G needed.

        Parameters:
        - k_points: Array of k-points in k-space (shape: (..., 3)).
        - bz_range: Tuple indicating the range of the BZ, default is (-0.5, 0.5) for each direction.
        - reciprocal_vectors: A list or matrix of reciprocal lattice vectors defining the BZ.

        Returns:
        - folded_k_points: The folded k-points within the BZ (shape: same as k_points).
        - G_vectors: The reciprocal lattice vectors that fold the k-points into the BZ (shape: same as k_points).
        """
        k_points = np.array(k_points)  # Ensure input is an array

        # Determine the G-vector multipliers for folding
        G_multiplier = np.floor((k_points - bz_range[0]) / (bz_range[1] - bz_range[0]))
        
        # Calculate the G_vectors
        if reciprocal_vectors is not None:
            reciprocal_vectors = np.array(reciprocal_vectors)  # Ensure reciprocal_vectors is an array
            G_vectors = np.tensordot(G_multiplier, reciprocal_vectors, axes=([1], [0]))
        else:
            # Assume a cubic lattice with unit cell length of 1
            G_vectors = G_multiplier * (bz_range[1] - bz_range[0])

        # Fold the k-points into the BZ
        folded_k_points = k_points - G_vectors
        # Correct for points exactly on the upper bound of the BZ
        mask_upper_bound = (folded_k_points == bz_range[0]) & (G_vectors >= 1.0)
        folded_k_points[mask_upper_bound] += (bz_range[1] - bz_range[0])
        G_vectors[mask_upper_bound] -= (bz_range[1] - bz_range[0])
        # Negate G_vectors
        G_vectors = -G_vectors
        # Handle points at the upper bound of the BZ
        mask_at_upper_bound = (folded_k_points == bz_range[1])
        folded_k_points[mask_at_upper_bound] -= (bz_range[1] - bz_range[0])
        G_vectors[mask_at_upper_bound] = 0

        return folded_k_points, G_vectors


    def find_closest_kpoint(self, point):
        # Convert point to a numpy array
        point = np.array(point)
        # Calculate distances considering periodic boundary conditions
        distances = np.linalg.norm((self.k - point + 0.5) % 1 - 0.5, axis=1)
        closest_idx = np.argmin(distances)
        
        return int(closest_idx)

    def get_kq_tables(self,qmpgrid):
        kplusq_table = np.zeros((self.nkpoints,qmpgrid.nkpoints),dtype=int)
        kminusq_table = np.zeros((self.nkpoints,qmpgrid.nkpoints), dtype=int)
        
        kplusq = self.k[:, np.newaxis, :] + qmpgrid.k[np.newaxis, :, :]
        kminusq = self.k[:, np.newaxis, :] - qmpgrid.k[np.newaxis, :, :]

        # Fold all kplusq and kminusq into the Brillouin zone
        kplusq = self.fold_into_bz(kplusq)
        kminusq = self.fold_into_bz(kminusq)

        # Find closest k-points for all combinations
        idxkplusq = np.apply_along_axis(self.find_closest_kpoint, -1, kplusq)
        idxkminusq = np.apply_along_axis(self.find_closest_kpoint, -1, kminusq)

        # Assign to tables
        kplusq_table = idxkplusq
        kminusq_table = idxkminusq

        return kplusq_table, kminusq_table
    
    def get_kq_tables_yambo(self,electronsdb):
        kplusq_table = np.zeros((self.nkpoints,electronsdb.nkpoints_ibz),dtype=int)
        kminusq_table = np.zeros((self.nkpoints,electronsdb.nkpoints_ibz), dtype=int)
        
        kplusq = self.k[:, np.newaxis, :] + electronsdb.red_kpoints[np.newaxis, :, :]
        kminusq = self.k[:, np.newaxis, :] - electronsdb.red_kpoints[np.newaxis, :, :]

        # Fold all kplusq and kminusq into the Brillouin zone
        kplusq = self.fold_into_bz(kplusq)
        kminusq = self.fold_into_bz(kminusq)

        # Find closest k-points for all combinations
        idxkplusq = np.apply_along_axis(self.find_closest_kpoint, -1, kplusq)
        idxkminusq = np.apply_along_axis(self.find_closest_kpoint, -1, kminusq)

        # Assign to tables
        kplusq_table = idxkplusq
        kminusq_table = idxkminusq


        return kplusq_table, kminusq_table

    def get_kindices_fromq(self,qmpgrid):
        q_points = qmpgrid.k  # Extract q-points array

        # Apply the function self.find_closest_kpoint to all q-points in a vectorized manner
        kindices_fromq = np.apply_along_axis(self.find_closest_kpoint, -1, q_points)

        # Assign the result directly
        self.kindices_fromq = kindices_fromq

        return kindices_fromq

    def fold_into_bz(self,points):
        'Fold a point in the first BZ defined in the range [-0.5,0.5]'
        # Applying the modulo operation to shift points within the range [-0.5, 0.5]
        folded_points = np.mod(points + 0.5, 1.0) - 0.5
        # Correcting points where original points were exactly 0.5 to remain 0.5
        folded_points[(points == 0.5)] = 0.5
        return folded_points
    
    def get_plaquette(self, nx, ny, nz, dir = 2):
        if (dir == 2):
            dir1 = np.array([1,0,0])
            dq1 = 1/nx
            dir2 = np.array([0,1,0])
            dq2 = 1/ny
            nps = nx*ny
        if (dir == 0):
            dir1 = np.array([0,1,0])
            dq1 = 1/ny
            dir2 = np.array([0,0,1])
            dq2 = 1/nz
            nps = ny*nz
        if (dir == 1):
            dir1 = np.array([1,0,0])
            dq1 = 1/nx
            dir2 = np.array([0,0,1])            
            dq2 = 1/nz
            nps = nx*nz
        counter = 0
        #here I need to use the q grid and then apply -b/2
        qplaquette_grid = np.zeros((nps, 4), dtype=int)
        for iq, q in enumerate(self.k):     # needs improving with array casting
            if (q[dir]==0.0):
                tmp_qp1, tmp_qp1Gvec = self.fold_into_bz_Gs(q + dq1*dir1)
                tmp_qp1p2, tmp_qp1p2Gvec = self.fold_into_bz_Gs(q + dq1*dir1+ dq2*dir2)
                tmp_qp2, tmp_qp2Gvec = self.fold_into_bz_Gs(q + dq2*dir2)  
                idxqp1 = self.find_closest_kpoint(tmp_qp1)
                idxqp1p2 = self.find_closest_kpoint(tmp_qp1p2)
                idxqp2 = self.find_closest_kpoint(tmp_qp2)
                qplaquette_grid[counter] = [iq, idxqp1, idxqp1p2, idxqp2]
                counter +=1            

        self.qplaquette_grid = qplaquette_grid