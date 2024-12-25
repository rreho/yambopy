from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_kpoints import KPointGenerator
from yambopy.wannier.wann_io import NNKP
from yambopy.units import ang2bohr
from scipy.spatial import cKDTree
class NNKP_Grids(KPointGenerator):
    def __init__(self, seedname,latdb, yambo_grid=False):
        self.nnkp_grid = NNKP(seedname)
        self.latdb = latdb
        self.yambo_grid = yambo_grid

    def __getattr__(self, name):
        # Delegate attribute access to self.nnkp_grid if the attribute doesn't exist in NNKP_Grids
        if hasattr(self.nnkp_grid, name):
            return getattr(self.nnkp_grid, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def generate(self):
        """Generate k-grid from NNKP file."""
        if(self.yambo_grid):
            self.k = np.array([self.fold_into_bz(k) for ik,k in enumerate(self.nnkp_grid.k)])    
        else:
            self.k = self.nnkp_grid.k
        self.lat = self.latdb.lat
        self.rlat = self.latdb.rlat*2*np.pi
        self.car_kpoints = red_car(self.k, self.rlat)*ang2bohr # result in Bohr
        self.red_kpoints = self.nnkp_grid.k
        self.nkpoints = len(self.k)
        self.weights = 1/self.nkpoints
        self.k_tree = cKDTree(self.k)

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
                idxqpb = self.find_closest_kpoint(tmp_qpb)
                qpb_grid[iq, ib] = tmp_qpb
                # here it should be tmp_Gvec, but with yambo grid I have inconsistencies because points are at 0.75
                qpb_grid_table[iq,ib] = [iq, idxqpb, int(qmpgrid.iG[ib+qmpgrid.nnkpts*iq,0]), int(qmpgrid.iG[ib+qmpgrid.nnkpts*iq,1]), int(qmpgrid.iG[ib+qmpgrid.nnkpts*iq,2])]
        
        self.qpb_grid = qpb_grid
        self.qpb_grid_table = qpb_grid_table

    def get_kpbover2_grid(self, qmpgrid: 'NNKP_Grids'):
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')

        # Reshape b_grid for vectorized addition
        b_grid = self.b_grid.reshape(self.nkpoints, self.nnkpts, 3)  # Shape (nkpoints, nnkpts, 3)

        # Add k and b_grid with broadcasting
        k_expanded = self.k[:, np.newaxis, :]  # Shape (nkpoints, 1, 3)
        combined_kb = k_expanded + b_grid  # Shape (nkpoints, nnkpts, 3)
        # Fold into the BZ for all k + b combinations
        folded_kb, Gvec = self.fold_into_bz_Gs(combined_kb.reshape(-1, 3))  # Flatten first two dims
        folded_kb = folded_kb.reshape(self.nkpoints, self.nnkpts, 3)
        Gvec = Gvec.reshape(self.nkpoints, self.nnkpts, 3)
        # Find closest kpoints
        idxkpbover2 = self.find_closest_kpoint(folded_kb.reshape(-1, 3)).reshape(self.nkpoints, self.nnkpts)
        # # Construct results

        self.kpbover2_grid = folded_kb
        self.kpbover2_grid_table = np.stack([
            np.repeat(np.arange(self.nkpoints)[:, np.newaxis], self.nnkpts, axis=1),  # ik
            idxkpbover2,  # Closest kpoint indices
            Gvec[..., 0].astype(int),  # Gx
            Gvec[..., 1].astype(int),  # Gy
            Gvec[..., 2].astype(int)   # Gz
        ], axis=-1)  # Shape (nkpoints, nnkpts, 5)


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
        kqmbover2_folded, Gvec = self.fold_into_bz_Gs(kqmbover2.reshape(-1, 3))  # Flatten for batch processing
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