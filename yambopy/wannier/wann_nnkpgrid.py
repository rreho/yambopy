from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_kpoints import KPointGenerator
from yambopy.wannier.wann_io import NNKP
from yambopy.units import ang2bohr
from scipy.spatial import cKDTree
class NNKP_Grids(KPointGenerator):
    def __init__(self, seedname):
        self.nnkp_grid = NNKP(seedname)
        self.generate()

    def __getattr__(self, name):
        # Delegate attribute access to self.nnkp_grid if the attribute doesn't exist in NNKP_Grids
        if hasattr(self.nnkp_grid, name):
            return getattr(self.nnkp_grid, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def generate(self):
        """Generate k-grid from NNKP file."""
        self.k = self.nnkp_grid.k
        self.red_kpoints = self.nnkp_grid.k
        self.nkpoints = len(self.k)
        self.weights = 1/self.nkpoints
        self.k_tree = cKDTree(self.k)

    def get_kmq_grid(self,qgrid):
        # if not isinstance(qmpgrid, NNKP_Grids):
        #     raise TypeError('Argument must be an instance of NNKP_Grids')
        #here I need to use the k-q grid and then apply -b/2
        # Prepare dimensions

        kmq_grid = self.k[:,None,:] - qgrid.k[None,:,:]

        nkpoints = self.nkpoints
        nqpoints = qgrid.nkpoints
        n_images=1
        shifts = np.array(np.meshgrid(
            *[np.arange(-n_images, n_images + 1)] * 3)).T.reshape(-1, 3)

        n_shifts = len(shifts)

        # Create all periodic images of the k-points
        images = (qgrid.k[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
        # Track which original index each image comes from
        origin_indices = np.repeat(np.arange(nkpoints), n_shifts)
        tree = cKDTree(images)

        dist, idx = tree.query(kmq_grid)
        # matched_images = images[idx]
        matched_indices = origin_indices[idx]
        kmq_grid = qgrid.k[matched_indices]

        Gvec = np.zeros(shape=(nkpoints, nqpoints, 3))        # temporarily

        # # Find closest k-points for all points
        # # Populate the grids
        # kmq_grid = kmq_folded  # Shape (nkpoints, nqpoints, nnkpts, 3)
        kmq_grid_table = np.stack(
            [
                np.arange(nkpoints)[:, None].repeat(nqpoints, axis=1),  # ik
                matched_indices,  # idkmq
                Gvec[..., 0].astype(int),  # Gx
                Gvec[..., 1].astype(int),  # Gy
                Gvec[..., 2].astype(int)   # Gz
            ],
            axis=-1
        ).astype(int)  # Shape (nkpoints, nqpoints, 5)
        # self.kmq_grid = kmq_grid
        self.kmq_grid_table = kmq_grid_table   
        self.kmq_grid = kmq_grid     
        
        return kmq_grid, kmq_grid_table

    def get_kpq_grid(self, qmpgrid):

        nkpoints = self.nkpoints
        nqpoints = qmpgrid.nkpoints
        
        # Broadcast k and q grids to shape (nkpoints, nqpoints, 3)
        k_grid = np.expand_dims(self.k, axis=1)  # Shape (nkpoints, 1, 3)
        q_grid = np.expand_dims(qmpgrid.k, axis=0)  # Shape (1, nqpoints, 3)
        kq_add = k_grid + q_grid  # Shape (nkpoints, nqpoints, 3)

        # Fold into the Brillouin Zone and get G-vectors
        kpq_folded, Gvec = self.fold_into_bz_Gs(kq_add.reshape(-1, 3),include_upper_bound=False)  # Flatten for batch processing
        kpq_folded = kpq_folded.reshape(nkpoints, nqpoints, 3)
        Gvec = Gvec.reshape(nkpoints, nqpoints, 3)

        # Find closest k-points for all points
        closest_indices = self.find_closest_kpoint(kpq_folded.reshape(-1, 3)).reshape(nkpoints, nqpoints)
        # Populate the grids
        kpq_grid = kpq_folded  # Shape (nkpoints, nqpoints, nnkpts, 3)
        kpq_grid_table = np.stack(
            [
                np.arange(nkpoints)[:, None].repeat(nqpoints, axis=1),  # ik
                closest_indices,  # idxkp
                Gvec[..., 0].astype(int),  # Gx
                Gvec[..., 1].astype(int),  # Gy
                Gvec[..., 2].astype(int)   # Gz
            ],
            axis=-1
        ).astype(int)  # Shape (nkpoints, nqpoints, 5)

        self.kpq_grid = kpq_grid
        self.kpq_grid_table = kpq_grid_table

        return kpq_grid, kpq_grid_table
      

    def sort_B_G(self, kpb_grid_table):
        bvec_sorting = np.array([1,7,6,4,0,2,3,5]) # don't ask
        argsorted  = np.argsort(bvec_sorting) # don't ask!
        k0 = self.k                     # shape (nk, 3)
        nk, nb = kpb_grid_table.shape[:2]

        neighbor_indices = kpb_grid_table[:, :, 1]     # shape (nk, nb)
        Gvecs = kpb_grid_table[:, :, 2:5]              # shape (nk, nb, 3)
        k_neighbors = self.k[neighbor_indices]              # shape (nk, nb, 3)

        Bvecs = k_neighbors + Gvecs - k0[:, None, :]               # shape (nk, nb, 3)
        Bvecs_rounded = np.round(Bvecs, decimals=6)

        # Now lexsort per k-point manually
        sort_idx = np.empty((nk, nb), dtype=int)
        for i in range(nk):
            sort_idx[i] = np.lexsort(Bvecs_rounded[i].T[::-1])     # (nb,)
        sort_idx = sort_idx[:,argsorted]
        # Apply sorting using advanced indexing (batch-style)
        batch_indices = np.arange(nk)[:, None]                     # shape (nk, 1)
        Bvecs_sorted = Bvecs[batch_indices, sort_idx]              # (nk, nb, 3)
        Gvecs_sorted = Gvecs[batch_indices, sort_idx]              # (nk, nb, 3)

        return Gvecs_sorted, Bvecs_sorted, sort_idx


    def get_kpb_grid(self, kmpgrid: 'NNKP_Grids'):
        '''
        For each k belonging to the Qgrid return Q+B and a table with indices
        containing the k index the k+b folded into the BZ and the G-vectors

        The nnkpgrid is exactly the same as the k+b grid. So this is used.
        '''
        if not isinstance(kmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')

        kpb_grid_table = kmpgrid.nnkp.copy()
        kpb_grid = self.k[kpb_grid_table[:,:,1]]  # Tested and True
        #b_list = self.k[qpb_grid_table[0][:,1]]
        Gvecs_sorted, Bvecs_sorted, sort_idx = self.sort_B_G(kpb_grid_table)
        self.sort_idx = sort_idx
        self.b_list = Bvecs_sorted
        self.kpb_grid = np.take_along_axis(kpb_grid, sort_idx[:,:,None], axis=1)
        self.kpb_grid_table = np.take_along_axis(kpb_grid_table, sort_idx[:,:,None], axis=1)

    def get_qpb_grid(self, qmpgrid: 'NNKP_Grids'):
        '''
        For each q belonging to the Qgrid return Q+B and a table with indices
        containing the q index the q+b folded into the BZ and the G-vectors

        The nnkpgrid is exactly the same as the k+b grid. So this is used.
        '''
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')

        qpb_grid_table = qmpgrid.nnkp.copy()
        qpb_grid = self.k[qpb_grid_table[:,:,1]]  # Tested and True
        #b_list = self.k[qpb_grid_table[0][:,1]]
        Gvecs_sorted, Bvecs_sorted, sort_idx = self.sort_B_G(qpb_grid_table)
        self.sort_idx = sort_idx
        self.b_list = Bvecs_sorted
        self.qpb_grid = np.take_along_axis(qpb_grid, sort_idx[:,:,None], axis=1)
        self.qpb_grid_table = np.take_along_axis(qpb_grid_table, sort_idx[:,:,None], axis=1)


    def get_wannier90toyambo(self, lat_k, yambo=False):
        '''
        Because in term 1 and term 2 we need to compute the neighbour kpoint, 
        we need the neighbour from wannier90 .nnkp and then convert it to a 
        point in the yambo grid. 
        '''
        k = self.nnkp_grid.k    # Wannier90 kgrid
        shifts = np.array(np.meshgrid(
            *[np.arange(-1, 1 + 1)] * 3)).T.reshape(-1, 3)
        
        images = (lat_k.red_kpoints[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
        # Track which original index each image comes from
        origin_indices = np.repeat(np.arange(self.nnkp_grid.nkpoints), len(shifts))
        tree = cKDTree(images)

        dist, idx = tree.query(k)   # where in the yambo grid is the wannier90 kpoint?
        # matched_images = images[idx]
        matched_indices = origin_indices[idx]   # index of the wannier90 kpoint in the yambo grid
        k = lat_k.red_kpoints[matched_indices]

        self.yambotowannier90_table = matched_indices   # this is the index of the yambo point given a wannier90 point
        self.wannier90toyambo_table = np.argsort(matched_indices)
        if yambo:
            self.set_yambo_grid()

    def set_yambo_grid(self):
        '''
        Change convention of everything to the original yambo kpoints.
        '''
        w2y = self.wannier90toyambo_table
        y2w = self.yambotowannier90_table

        self.nnkp_grid.nnkp[:,:,:2] = w2y[self.nnkp_grid.nnkp[:,:,:2]]
        self.k = self.k[w2y]
        self.nnkp_grid.k = self.k


    def get_kpbover2_grid(self, kmpgrid: 'NNKP_Grids'):
        '''
        See get_qpb_grid. Now we have small b = B/2. But in our kmpgrid from the .nnkp file the b is just the b.
        '''
        if not isinstance(kmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')
        self.kpbover2_grid_table = kmpgrid.nnkp.copy()
        self.kpbover2_grid = self.k[self.kpbover2_grid_table[:,:,1]]  # Tested and True


    def get_kmqmbover2_grid(self, qmpgrid: 'NNKP_Grids'):       # need to improve this one
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')
        #here I need to use the k-q grid and then apply -b/2
        #kmqmbover2_grid = np.zeros((self.nkpoints, qmpgrid.nkpoints, qmpgrid.nnkpts,3))
        #kmqmbover2_grid_table = np.zeros((self.nkpoints, qmpgrid.nkpoints, qmpgrid.nnkpts,5),dtype=int)
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')

        # Prepare dimensions
        nkpoints = self.nkpoints
        nqpoints = qmpgrid.nkpoints
        nnkpts = qmpgrid.nnkpts

        # Broadcast k and q grids to shape (nkpoints, nqpoints, 3)
        k_grid = np.expand_dims(self.k, axis=1)  # Shape (nkpoints, 1, 3)
        q_grid = np.expand_dims(qmpgrid.k, axis=0)  # Shape (1, nqpoints, 3)
        kq_diff = k_grid - q_grid  # Shape (nkpoints, nqpoints, 3)

        # Reshape b_grid to be specific to each k-point
        b_grid = self.b_grid.reshape(nkpoints, nnkpts, 3)  # Shape (nkpoints, nnkpts, 3)

        # Calculate k - q - b for all combinations
        kqmbover2 = (
            kq_diff[:, :, np.newaxis, :]  # Shape (nkpoints, nqpoints, 1, 3)
            - b_grid[:, np.newaxis, :, :]  # Shape (nkpoints, 1, nnkpts, 3)
        )  # Final Shape (nkpoints, nqpoints, nnkpts, 3)

        # Fold into the Brillouin Zone and get G-vectors
        kqmbover2_folded, Gvec = self.fold_into_bz_Gs(kqmbover2.reshape(-1, 3),include_upper_bound=True)  # Flatten for batch processing
        kqmbover2_folded = kqmbover2_folded.reshape(nkpoints, nqpoints, nnkpts, 3)
        Gvec = Gvec.reshape(nkpoints, nqpoints, nnkpts, 3)

        # Find closest k-points for all points
        closest_indices = self.find_closest_kpoint(kqmbover2_folded.reshape(-1, 3)).reshape(nkpoints, nqpoints, nnkpts)
        # Populate the grids
        self.kmqmbover2_grid = kqmbover2_folded  # Shape (nkpoints, nqpoints, nnkpts, 3)
        self.kmqmbover2_grid_table = np.stack(
            [
                np.arange(nkpoints)[:, None, None].repeat(nqpoints, axis=1).repeat(nnkpts, axis=2),  # ik
                closest_indices,  # idxkmqmbover2
                Gvec[..., 0].astype(int),  # Gx
                Gvec[..., 1].astype(int),  # Gy
                Gvec[..., 2].astype(int)   # Gz
            ],
            axis=-1
        ).astype(int)  # Shape (nkpoints, nqpoints, nnkpts, 5)

    def get_kq_tables_yambo(self,electronsdb):
        
        kplusq_table = np.zeros((self.nkpoints,electronsdb.nkpoints_ibz),dtype=int)
        kminusq_table = np.zeros((self.nkpoints,electronsdb.nkpoints_ibz), dtype=int)
        
        _,kplusq_table = self.get_kpq_grid_yambo(electronsdb.red_kpoints)
        _,kminusq_table = self.get_kmq_grid_yambo(electronsdb.red_kpoints)

        return kplusq_table, kminusq_table

    def get_kmq_grid_yambo(self,red_kpoints):

        nkpoints = self.nkpoints
        nqpoints = len(red_kpoints)
     
        # Broadcast k and q grids to shape (nkpoints, nqpoints, 3)
        k_grid = np.expand_dims(self.k, axis=1)  # Shape (nkpoints, 1, 3)
        q_grid = np.expand_dims(red_kpoints, axis=0)  # Shape (1, nqpoints, 3)
        kq_diff = k_grid - q_grid  # Shape (nkpoints, nqpoints, 3)

        # Fold into the Brillouin Zone and get G-vectors
        kmq_folded, Gvec = self.fold_into_bz_Gs(kq_diff.reshape(-1, 3))
        kmq_folded = kmq_folded.reshape(nkpoints, nqpoints, 3)
        Gvec = Gvec.reshape(nkpoints, nqpoints, 3)

        # Find closest k-points for all points
        closest_indices = self.find_closest_kpoint(kmq_folded.reshape(-1, 3)).reshape(nkpoints, nqpoints)
        # Populate the grids
        kmq_grid = kmq_folded  # Shape (nkpoints, nqpoints, nnkpts, 3)
        kmq_grid_table = np.stack(
            [
                np.arange(nkpoints)[:, None].repeat(nqpoints, axis=1),  # ik
                closest_indices,  # idkmq
                Gvec[..., 0].astype(int),  # Gx
                Gvec[..., 1].astype(int),  # Gy
                Gvec[..., 2].astype(int)   # Gz
            ],
            axis=-1
        ).astype(int)  # Shape (nkpoints, nqpoints, 5)
        self.kmq_grid = kmq_grid
        self.kmq_grid_table = kmq_grid_table        
        
        return kmq_grid, kmq_grid_table
    def get_kpq_grid_yambo(self, red_kpoints):

        nkpoints = self.nkpoints
        nqpoints = len(red_kpoints)
        
        # Broadcast k and q grids to shape (nkpoints, nqpoints, 3)
        k_grid = np.expand_dims(self.k, axis=1)  # Shape (nkpoints, 1, 3)
        q_grid = np.expand_dims(red_kpoints, axis=0)  # Shape (1, nqpoints, 3)
        kq_add = k_grid + q_grid  # Shape (nkpoints, nqpoints, 3)

        # Fold into the Brillouin Zone and get G-vectors
        kpq_folded, Gvec = self.fold_into_bz_Gs(kq_add.reshape(-1, 3))
        kpq_folded = kpq_folded.reshape(nkpoints, nqpoints, 3)
        Gvec = Gvec.reshape(nkpoints, nqpoints, 3)

        # Find closest k-points for all points
        closest_indices = self.find_closest_kpoint(kpq_folded.reshape(-1, 3)).reshape(nkpoints, nqpoints)
        # Populate the grids
        kpq_grid = kpq_folded  # Shape (nkpoints, nqpoints, nnkpts, 3)
        kpq_grid_table = np.stack(
            [
                np.arange(nkpoints)[:, None].repeat(nqpoints, axis=1),  # ik
                closest_indices,  # idxkp
                Gvec[..., 0].astype(int),  # Gx
                Gvec[..., 1].astype(int),  # Gy
                Gvec[..., 2].astype(int)   # Gz
            ],
            axis=-1
        ).astype(int)  # Shape (nkpoints, nqpoints, 5)

        self.kpq_grid = kpq_grid
        self.kpq_grid_table = kpq_grid_table

        return kpq_grid, kpq_grid_table

    def __str__(self):
        return "Instance of NNKP_Grids"    
