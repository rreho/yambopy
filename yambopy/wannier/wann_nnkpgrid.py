from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_kpoints import KPointGenerator
from yambopy.wannier.wann_io import NNKP
from yambopy.units import ang2bohr
from scipy.spatial import cKDTree
class NNKP_Grids(KPointGenerator):
    def __init__(self, seedname,latdb):
        self.nnkp_grid = NNKP(seedname)
        self.latdb = latdb
        self.generate()

    def __getattr__(self, name):
        # Delegate attribute access to self.nnkp_grid if the attribute doesn't exist in NNKP_Grids
        if hasattr(self.nnkp_grid, name):
            return getattr(self.nnkp_grid, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def generate(self):
        """Generate k-grid from NNKP file."""
        self.k = self.nnkp_grid.k
        self.lat = self.latdb.lat
        self.rlat = self.latdb.rlat*2*np.pi
        self.car_kpoints = red_car(self.k, self.rlat)*ang2bohr # result in Bohr
        self.red_kpoints = self.nnkp_grid.k
        self.nkpoints = len(self.k)
        self.weights = 1/self.nkpoints
        self.k_tree = cKDTree(self.k)

    def get_kmq_grid(self,qgrid, sign = "+"):
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
      
    def get_kpb_grid(self, kmpgrid: 'NNKP_Grids'):
        '''
        For each k belonging to the Qgrid return Q+B and a table with indices
        containing the k index the k+b folded into the BZ and the G-vectors

        The nnkpgrid is exactly the same as the k+b grid. So this is used.
        '''
        if not isinstance(kmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')

        self.kpb_grid_table = kmpgrid.nnkp.copy()
        self.kpb_grid = self.k[self.kpb_grid_table[:,:,1]]  # Tested and True

    def get_qpb_grid(self, qmpgrid: 'NNKP_Grids'):
        '''
        For each q belonging to the Qgrid return Q+B and a table with indices
        containing the q index the q+b folded into the BZ and the G-vectors

        The nnkpgrid is exactly the same as the k+b grid. So this is used.
        '''
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')

        self.qpb_grid_table = qmpgrid.nnkp.copy()
        self.qpb_grid = self.k[self.qpb_grid_table[:,:,1]]  # Tested and True

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
