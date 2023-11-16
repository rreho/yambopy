import sisl
import spglib
from yambopy.lattice import red_car
import numpy as np

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