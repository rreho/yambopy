import sisl
import spglib
from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_io import NNKP

class GridUtilities:
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

    def get_qpb_grid(self, qmpgrid):
        '''
        For each q belonging to the Qgrid return Q+B and a table with indices
        containing the q index the q+b folded into the BZ and the G-vectors
        '''
        # if not isinstance(qmpgrid, NNKP_Grids):
        #     raise TypeError('Argument must be an instance of NNKP_Grids')
        
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

    def get_kpbover2_grid(self, qmpgrid):

        # if not isinstance(qmpgrid, NNKP_Grids):
        #     raise TypeError('Argument must be an instance of NNKP_Grids')  
        
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

    def get_kmqmbover2_grid(self, qmpgrid):

        # if not isinstance(qmpgrid, NNKP_Grids):
        #     raise TypeError('Argument must be an instance of NNKP_Grids') 
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
            if folded_k_point[i] == bz_range[0]:
                folded_k_point[i] += (bz_range[1]-bz_range[0])
                G_vector[i] = 0
        return folded_k_point, G_vector

    def find_closest_kpoint(self, point):
        # Convert point to a numpy array
        point = np.array(point)
        
        # Calculate distances considering periodic boundary conditions
        distances = np.linalg.norm((self.k - point + 0.5) % 1 - 0.5, axis=1)
        
        # Find the index of the minimum distance
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
    
    def get_kq_tables_yambo(self,savedb):
        kplusq_table = np.zeros((self.nkpoints,savedb.nkpoints),dtype=int)
        kminusq_table = np.zeros((self.nkpoints,savedb.nkpoints), dtype=int)
        
        for ik, k in enumerate(self.k):
            for iq, q in enumerate(savedb.red_kpoints):
                kplusq = k+q
                kminusq = k-q
                kplusq = self.fold_into_bz(kplusq)
                kminusq = self.fold_into_bz(kminusq)
                idxkplusq = self.find_closest_kpoint(kplusq)
                idxkminusq = self.find_closest_kpoint(kminusq)
                kplusq_table[ik,iq] = idxkplusq
                kminusq_table[ik,iq] = idxkminusq

        return kplusq_table, kminusq_table

    def get_kindices_fromq(self,qmpgrid):
        kindices_fromq = np.zeros((qmpgrid.nkpoints),dtype=int) # get k index corresponding to q-point
         
        for iq, q in enumerate(qmpgrid.k):
            idxk = self.find_closest_kpoint(q)
            kindices_fromq[iq] = idxk

        self.kindices_fromq=kindices_fromq
        return kindices_fromq

    def fold_into_bz(self,points):
        'Fold a point in the first BZ defined in the range ]-0.5,0.5]'
        # Applying the modulo operation to shift points within the range [-0.5, 0.5]
        folded_points = points%1.0
        # Correcting points where original points were exactly 0.5 to remain 0.5
        folded_points[folded_points > 0.5] -= 1
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

class tb_Monkhorst_Pack(sisl.physics.MonkhorstPack, GridUtilities):
    def __init__(self, nx, ny, nz, latdb, *args, **kwargs):
        #remember to pass with trs=False
        super().__init__(*args, **kwargs)
        self.car_kpoints = self.tocartesian(self.k)
        self.nkpoints = len(self.k)
        self.nnkpts = 8
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.b_grid = self._find_neighbors()
        self.iG = np.zeros((self.nkpoints*8,3),dtype=int) # for now hard coded to zeros. Note also that when we
        # fold into bz we get some G vctors which we don't care for now
        self.latdb = latdb # note here we pass tmp_lat that I defined in the example notebook.
        # We might want in the future to define a general lattice class.
        self.car_kpoints = red_car(self.k, self.latdb.rec_lat)

    def _find_neighbors(self):
        # Initialize the b_grid array
        b_grid = np.zeros((self.nkpoints*8,3), dtype=np.float128)
        
        # Generate all possible shifts for neighbors
        shifts = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 1],
            [-1, -1, -1]
        ])
        
        # Iterate over all k-points to find their neighbors
        for i in range(self.nkpoints):
            ix, iy, iz = np.unravel_index(i, (self.nx, self.ny, self.nz))
            # Calculate neighbor indices with periodic boundary conditions
            for j, shift in enumerate(shifts):
                nix = (ix + shift[0]) % self.nx
                niy = (iy + shift[1]) % self.ny
                niz = (iz + shift[2]) % self.nz
                neighbor_index = np.ravel_multi_index((nix, niy, niz), (self.nx, self.ny, self.nz))
                b_grid[8*i+j] = self.fold_into_bz(self.k[neighbor_index]-self.k[i])
        
        return b_grid   
    
    # def find_closest_kpoint(self, point):
    #     # Convert point to a numpy array
    #     point = np.array(point)
        
    #     # Calculate distances considering periodic boundary conditions
    #     distances = np.linalg.norm((self.k - point + 0.5) % 1 - 0.5, axis=1)
        
    #     # Find the index of the minimum distance
    #     closest_idx = np.argmin(distances)
        
    #     return int(closest_idx)

    # def fold_into_bz(self,points):
    #     'Fold a point in the first BZ defined in the range [-0.5,0.5]'
    #     # Applying the modulo operation to shift points within the range [-0.5, 0.5]
    #     folded_points = np.mod(points + 0.5, 1.0) - 0.5
    #     # Correcting points where original points were exactly 0.5 to remain 0.5
    #     folded_points[(points == 0.5)] = 0.5
    #     return folded_points

    # def find_closest_kpoints_broadcasting(self, points):
    #     """
    #     Find the indices of the closest k-points in self.k for each point in points.
    #     :param points: An array of points.
    #     :return: An array of indices of the closest k-points in self.k.
    #     For now this routine is not used. May be useful in the future.
    #     """
    #     # Reshape points and self.k for broadcasting
    #     points = points[:, np.newaxis, :]  # Shape: (num_points, 1, 3)
    #     k = self.k[np.newaxis, :, :]  # Shape: (1, num_k_points, 3)

    #     # Calculate distances (broadcasting is used here)
    #     distances = np.linalg.norm(points - k, axis=2)

    #     # Find the index of the minimum distance for each point
    #     closest_indices = np.argmin(distances, axis=1)
    #     return closest_indices


    # def get_kq_tables(self,qmpgrid):
    #     kplusq_table = np.zeros((self.nkpoints,qmpgrid.nkpoints),dtype=int)
    #     kminusq_table = np.zeros((self.nkpoints,qmpgrid.nkpoints), dtype=int)
        
    #     for ik, k in enumerate(self.k):
    #         for iq, q in enumerate(qmpgrid.k):
    #             kplusq = k+q
    #             kminusq = k-q
    #             kplusq = self.fold_into_bz(kplusq)
    #             kminusq = self.fold_into_bz(kminusq)
    #             idxkplusq = self.find_closest_kpoint(kplusq)
    #             idxkminusq = self.find_closest_kpoint(kminusq)
    #             kplusq_table[ik,iq] = idxkplusq
    #             kminusq_table[ik,iq] = idxkminusq

    #     return kplusq_table, kminusq_table
    
    # def get_plaquette(self, nx, ny, nz, dir = 2):
    #     if (dir == 2):
    #         dir1 = np.array([1,0,0])
    #         dq1 = 1/nx
    #         dir2 = np.array([0,1,0])
    #         dq2 = 1/ny
    #         nps = nx*ny
    #     if (dir == 0):
    #         dir1 = np.array([0,1,0])
    #         dq1 = 1/ny
    #         dir2 = np.array([0,0,1])
    #         dq2 = 1/nz
    #         nps = ny*nz
    #     if (dir == 1):
    #         dir1 = np.array([1,0,0])
    #         dq1 = 1/nx
    #         dir2 = np.array([0,0,1])            
    #         dq2 = 1/nz
    #         nps = nx*nz
    #     counter = 0
    #     #here I need to use the q grid and then apply -b/2
    #     qplaquette_grid = np.zeros((nps, 4), dtype=int)
    #     for iq, q in enumerate(self.k):
    #         if (q[dir]==0.0):
    #             tmp_qp1 = self.fold_into_bz(q + dq1*dir1)
    #             tmp_qp1p2 = self.fold_into_bz(q + dq1*dir1+ dq2*dir2)
    #             tmp_qp2 = self.fold_into_bz(q + dq2*dir2)  
    #             idxqp1 = self.find_closest_kpoint(tmp_qp1)
    #             idxqp1p2 = self.find_closest_kpoint(tmp_qp1p2)
    #             idxqp2 = self.find_closest_kpoint(tmp_qp2)
    #             qplaquette_grid[counter] = [iq, idxqp1, idxqp1p2, idxqp2]
    #             counter +=1            
    #     self.qplaquette_grid = qplaquette_grid
class NNKP_Grids(NNKP, GridUtilities):
    def __init__(self, seedname, latdb, yambo_grid=False):
        super().__init__(seedname)
        if(yambo_grid):
            self.k = np.array([self.fold_into_bz(k) for ik,k in enumerate(self.k)])
        self.latdb = latdb
        self.lat = latdb.lat
        self.rlat = latdb.rlat*2*np.pi
        self.car_kpoints = red_car(self.k, self.rlat)
        self.red_kpoints = self.k
        

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
    
# Class below is not needed if we use sisl
    # The problem with mp_grid is that it does not center the grid at Gamma, while in sisl is properly implemented
# class mp_grid(GridUtilities):
#     def __init__(self, nx, ny, nz,latdb, nnkpts=8, shift = np.array([0.0,0.0,0.0])):
#         super().__init__()
#         self.nnkpts = 8 
#         self.nx = nx 
#         self.ny = ny 
#         self.nz = nz
#         self.k = self._generate_k_grid(shift)
#         self.nkpoints = len(self.k)
#         self.b_grid = self._find_neighbors()
#         self.iG = np.zeros((self.nkpoints*8,3),dtype=int) # for now hard coded to zeros. Note also that when we
#         # fold into bz we get some G vctors which we don't care for now
#         self.latdb = latdb # note here we pass tmp_lat that I defined in the example notebook.
#         # We might want in the future to define a general lattice class.
#         self.car_kpoints = red_car(self.k, self.latdb.rec_lat)

#     def _generate_k_grid(self, shift = np.array([0.0,0.0,0.0])):
#         # Generate the Monkhorst-Pack k-point grid
#         # ix, iy, iz are indices running from 0 to nx-1, ny-1, nz-1 respectively
#         ix, iy, iz = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing='ij')
#         # Calculate the grid point positions in reciprocal space
#         kx = (ix ) / self.nx  + shift[0]
#         ky = (iy ) / self.ny  + shift[1]
#         kz = (iz ) / self.nz  + shift[2]
        
#         # Stack the kx, ky, kz arrays to form the k-grid
#         k_grid = np.vstack((kx.ravel(), ky.ravel(), kz.ravel())).T
        
#         return k_grid  

#     def _find_neighbors(self):
#         # Initialize the b_grid array
#         b_grid = np.zeros((self.nkpoints*8,3), dtype=np.float128)
        
#         # Generate all possible shifts for neighbors
#         shifts = np.array([(dx, dy, dz) for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]])
        
#         # Iterate over all k-points to find their neighbors
#         for i in range(self.nkpoints):
#             ix, iy, iz = np.unravel_index(i, (self.nx, self.ny, self.nz))
#             # Calculate neighbor indices with periodic boundary conditions
#             for j, shift in enumerate(shifts):
#                 nix = (ix + shift[0]) % self.nx
#                 niy = (iy + shift[1]) % self.ny
#                 niz = (iz + shift[2]) % self.nz
#                 neighbor_index = np.ravel_multi_index((nix, niy, niz), (self.nx, self.ny, self.nz))
#                 b_grid[8*i+j] = self.k[neighbor_index]-self.k[i]
        
#         return b_grid   
    
class k_list(GridUtilities):
    def __init__(self, qlist, latdb, nnkpts = 8):
        self.nnkpts = 8
        self.k = qlist
        self.nkpoints = len(self.k)
        self.iG = np.zeros((self.nkpoints*8,3),dtype=int)
        self.latdb = latdb # note here we pass tmp_lat that I defined in the example notebook.
        self.car_kpoints = red_car(self.k, self.latdb.rec_lat)