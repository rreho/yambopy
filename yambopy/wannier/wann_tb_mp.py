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
        for iq, q in enumerate(self.k):
            if (q[dir]==0.0):
                tmp_qp1 = self.fold_into_bz(q + dq1*dir1)
                tmp_qp1p2 = self.fold_into_bz(q + dq1*dir1+ dq2*dir2)
                tmp_qp2 = self.fold_into_bz(q + dq2*dir2)  
                idxqp1 = self.find_closest_kpoint(tmp_qp1)
                idxqp1p2 = self.find_closest_kpoint(tmp_qp1p2)
                idxqp2 = self.find_closest_kpoint(tmp_qp2)
                qplaquette_grid[counter] = [iq, idxqp1, idxqp1p2, idxqp2]
                counter +=1            
        self.qplaquette_grid = qplaquette_grid
class NNKP_Grids(NNKP):
    def __init__(self, seedname, latdb, yambo_grid=False):
        super().__init__(seedname)
        if(yambo_grid):
            self.k = np.array([self.fold_into_bz(k) for ik,k in enumerate(self.k)])
        self.latdb = latdb
        self.lat = latdb.lat
        self.rlat = latdb.rlat*2*np.pi
        self.car_kpoints = red_car(self.k, self.rlat)
        
    def get_kmq_grid(self, qmpgrid):

        #qmpgrid is meant to be an nnkp object
        kmq_grid = np.zeros((self.nkpoints, qmpgrid.nkpoints, 3))
        kmq_grid_table = np.zeros((self.nkpoints, qmpgrid.nkpoints, 5),dtype= int)
        for ik, k in enumerate(self.k):
            for iq, q in enumerate(qmpgrid.k):
                tmp_kmq, tmp_Gvec = self.fold_into_bz_Gs(k-q)
                kmq_grid[ik,iq] = tmp_kmq
                idxkmq = self.find_closest_kpoint(tmp_kmq)
                kmq_grid_table[ik,iq] = [ik, idxkmq, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kmq_grid = kmq_grid
        self.kmq_grid_table = kmq_grid_table

    def get_qpb_grid(self, qmpgrid: 'NNKP_Grids'):
        '''
        For each q belonging to the Qgrid return Q+B and a table with indices
        containing the q index the q+b folded into the BZ and the G-vectors
        '''
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')
        
        # here I should work only with qmpgrid and its qmpgrid.b_grid
        qpb_grid = np.zeros((qmpgrid.nkpoints, qmpgrid.nnkpts, 3))
        qpb_grid_table = np.zeros((qmpgrid.nkpoints, qmpgrid.nnkpts, 5), dtype = int)
        for iq, q in enumerate(qmpgrid.k):
            for ib, b in enumerate(qmpgrid.b_grid[qmpgrid.nnkpts*iq:qmpgrid.nnkpts*(iq+1)]):
                tmp_qpb, tmp_Gvec = qmpgrid.fold_into_bz_Gs(q+b)
                qpb_grid[iq, ib] = tmp_qpb
                idxqpb = self.find_closest_kpoint(tmp_qpb)
                # here it should be tmp_Gvec, but with yambo grid I have inconsistencies because points are at 0.75
                qpb_grid_table[iq,ib] = [iq, idxqpb, int(qmpgrid.iG[ib+qmpgrid.nnkpts*iq,0]), int(qmpgrid.iG[ib+qmpgrid.nnkpts*iq,1]), int(qmpgrid.iG[ib+qmpgrid.nnkpts*iq,2])]
        
        self.qpb_grid = qpb_grid
        self.qpb_grid_table = qpb_grid_table

    def get_kpbover2_grid(self, qmpgrid: 'NNKP_Grids'):

        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')  
        
        #qmpgrid is meant to be an nnkp object
        kpbover2_grid = np.zeros((self.nkpoints, qmpgrid.nnkpts, 3))
        kpbover2_grid_table = np.zeros((self.nkpoints, qmpgrid.nnkpts, 5),dtype= int)
        for ik, k in enumerate(self.k):
            for ib, b in enumerate(qmpgrid.b_grid[qmpgrid.nnkpts*ik:qmpgrid.nnkpts*(ik+1)]):
                tmp_kpbover2, tmp_Gvec = self.fold_into_bz_Gs(k+b/2)
                kpbover2_grid[ik,ib] = tmp_kpbover2
                idxkpbover2 = self.find_closest_kpoint(tmp_kpbover2)
                kpbover2_grid_table[ik,ib] = [ik, idxkpbover2, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kpbover2_grid = kpbover2_grid
        self.kpbover2_grid_table = kpbover2_grid_table

    def get_kmqmbover2_grid(self, qmpgrid: 'NNKP_Grids'):

        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids') 
        #here I need to use the k-q grid and then apply -b/2
        kmqmbover2_grid = np.zeros((self.nkpoints, qmpgrid.nkpoints, qmpgrid.nnkpts,3))
        kmqmbover2_grid_table = np.zeros((self.nkpoints, qmpgrid.nkpoints, qmpgrid.nnkpts,5),dtype=int)
        for ik, k in enumerate(self.k):
            for iq, q in enumerate(qmpgrid.k):
                for ib, b in enumerate(qmpgrid.b_grid[qmpgrid.nnkpts*iq:qmpgrid.nnkpts*(iq+1)]):
                    tmp_kmqmbover2, tmp_Gvec = self.fold_into_bz_Gs(k -q - b/2)
                    kmqmbover2_grid[ik, iq, ib] = tmp_kmqmbover2
                    idxkmqmbover2 = self.find_closest_kpoint(tmp_kmqmbover2)
                    kmqmbover2_grid_table[ik, iq, ib] = [ik, idxkmqmbover2, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kmqmbover2_grid = kmqmbover2_grid
        self.kmqmbover2_grid_table = kmqmbover2_grid_table



    def fold_into_bz_Gs(self, k_point, bz_range=(-0.5, 0.5), reciprocal_vectors=None):
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
        # Correct for the case when the point is exactly on the upper bound of the BZ
        for i in range(len(folded_k_point)):
            if folded_k_point[i] == bz_range[0] and G_vector[i] >= 1.0:
                folded_k_point[i] += (bz_range[1] - bz_range[0])
                G_vector[i] -= (bz_range[1] - bz_range[0])
            G_vector[i] = -G_vector[i]
            if folded_k_point[i] == bz_range[1]:
                folded_k_point[i] -= (bz_range[1]-bz_range[0])
                G_vector[i] = 0
        return folded_k_point, G_vector

    def find_closest_kpoint(self, point):
        distances = np.linalg.norm(self.k-point,axis=1)
        closest_idx = np.argmin(distances)
        return int(closest_idx)

    def get_kq_tables(self,qmpgrid):
        kplusq_table = np.zeros((self.nkpoints,self.nkpoints),dtype=int)
        kminusq_table = np.zeros((self.nkpoints,self.nkpoints), dtype=int)
        
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

    def fold_into_bz(self,points):
        'Fold a point in the first BZ defined in the range [-0.5,0.5]'
        folded_points = np.mod(points+0.5, 1.0) - 0.5
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
        for iq, q in enumerate(self.k):
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