import sisl
import spglib
from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_io import NNKP
class tb_Monkhorst_Pack(sisl.physics.MonkhorstPack):
    def __init__(self, *args, **kwargs):
        #remember to pass with trs=False
        super().__init__(*args, **kwargs)
        self.car_kpoints = self.tocartesian(self.k)
        self.nkpoints = len(self.k)

    
    def find_closest_kpoint(self, point):
        distances = np.linalg.norm(self.k-point,axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx

    def fold_into_bz(self,points):
        'Fold a point in the first BZ defined in the range [-0.5,0.5]'
        folded_points = np.mod(points+0.5, 1.0) - 0.5
        return folded_points

    def find_closest_kpoints_broadcasting(self, points):
        """
        Find the indices of the closest k-points in self.k for each point in points.
        :param points: An array of points.
        :return: An array of indices of the closest k-points in self.k.
        For now this routine is not used. May be useful in the future.
        """
        # Reshape points and self.k for broadcasting
        points = points[:, np.newaxis, :]  # Shape: (num_points, 1, 3)
        k = self.k[np.newaxis, :, :]  # Shape: (1, num_k_points, 3)

        # Calculate distances (broadcasting is used here)
        distances = np.linalg.norm(points - k, axis=2)

        # Find the index of the minimum distance for each point
        closest_indices = np.argmin(distances, axis=1)
        return closest_indices

    def fold_into_bz(self,points):
        'Fold a point in the first BZ defined in the range [-0.5,0.5]'
        folded_points = np.mod(points+0.5, 1.0) - 0.5
        return folded_points

    def get_kq_tables(self,qmpgrid):
        kplusq_table = np.zeros((self.nkpoints,qmpgrid.nkpoints),dtype=int)
        kminusq_table = np.zeros((self.nkpoints,qmpgrid.nkpoints), dtype=int)
        
        for ik, k in enumerate(self.k):
            for iq, q in enumerate(qmpgrid.k):
                kplusq = k+q
                kminusq = k-q
                kplusq = self.fold_into_bz(kplusq)
                kminusq = self.fold_into_bz(kminusq)
                idxkplusq = self.find_closest_kpoint(kplusq)
                idxkminusq = self.find_closest_kpoint(kminusq)
                kplusq_table[ik,iq] = idxkplusq
                kminusq_table[ik,iq] = idxkminusq

        return kplusq_table, kminusq_table
    
class NNKP_Grids(NNKP):
    def __init__(self, seedname):
        super().__init__(seedname)
       
    def get_kmq_grid(self, qmpgrid):
        #qmpgrid is meant to be an nnkp object
        kmq_grid = np.zeros((self.num_kpoints, qmpgrid.num_kpoints, 3))
        kmq_grid_table = np.zeros((self.num_kpoints, qmpgrid.num_kpoints, 5),dtype= int)
        for ik, k in enumerate(self.kpoints):
            for iq, q in enumerate(qmpgrid.kpoints):
                tmp_kmq, tmp_Gvec = self.fold_into_bz(k-q)
                kmq_grid[ik,iq] = tmp_kmq
                idxkmq = self.find_closest_kpoint(k-q)
                kmq_grid_table[ik,iq] = [ik, idxkmq, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kmq_grid = kmq_grid
        self.kmq_grid_table = kmq_grid_table



    def fold_into_bz(self, k_point, bz_range=(-0.5, 0.5), reciprocal_vectors=None):
        """
        Fold a k-point into the first Brillouin Zone and determine the reciprocal lattice vector G needed.
        
        Parameters:
        - k_point: A point in k-space.
        - bz_range: Tuple indicating the range of the BZ, default is (-0.5, 0.5) for each direction.
        - reciprocal_vectors: A list of reciprocal lattice vectors defining the BZ.
        
        Returns:
        - folded_k_point: The folded k-point within the BZ.
        - G_vector: The reciprocal lattice vector that folds the k-point into the BZ.
        """
        
        k_point = np.array(k_point)
        
        # Determine the G-vector multiplier for folding
        G_multiplier = np.floor((k_point - bz_range[0]) / (bz_range[1] - bz_range[0]))
        
        # Calculate the G_vector
        if reciprocal_vectors is not None:
            G_vector = np.dot(G_multiplier, reciprocal_vectors)
        else:
            # If no reciprocal lattice vectors are provided, assume a cubic lattice with unit cell length of 1
            G_vector = G_multiplier * (bz_range[1] - bz_range[0])

        # Fold the k_point into the BZ
        folded_k_point = k_point - G_vector
        
        return folded_k_point, G_vector

    def find_closest_kpoint(self, point):
        distances = np.linalg.norm(self.kpoints-point,axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx
    
class tb_symm_grid():
    def __init__(self, latdb, mesh, is_shift=[0.0,0.0,0.0]):
        self.latdb = latdb
        self.mesh = mesh
        self.cell = (latdb.lat, latdb.red_atomic_positions, latdb.atomic_numbers) 
        #remember to pass with trs=False
        self.mapping, self.grid = spglib.get_ir_reciprocal_mesh(mesh, self.cell, is_shift)
        self.car_kpoints = self.tocartesian(self.k)
        self.nkpoints = len(self.k)
        (self.ibz_kpoints, self.fullBZ_kpoints) = self._get_kpoints()
        self.car_kpoints = red_car(self.fullBZ_kpoints, latdb.rlat)
        self.ibz_car_kpoints = red_car(self.ibz_car_kpoints, latdb.rlat)

    @classmethod
    def _get_kpoints(cls):
        fullbzgrid = []
        for i, (ir_gp_id, gp) in enumerate(zip(cls.mapping, cls.grid)):
            fullbzgrid = np.append(fullbzgrid, gp.astype(float)/self.mesh)
        ibzgrid = cls.grid[np.unique(cls.mapping)]/np.array(cls.mesh,dtype=float)
        
        return ibzgrid, fullbzgrid