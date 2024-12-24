from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_kpoints import KPointGenerator
from yambopy.wannier.wann_io import NNKP
from yambopy.units import ang2bohr

class NNKP_Grids(KPointGenerator):
    def __init__(self, seedname,latdb, yambo_grid=False):
        self.nnkp_grid = NNKP(seedname)
        self.latdb = latdb
        self.yambo_grid = yambo_grid

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
        
        #qmpgrid is meant to be an nnkp object
        kpbover2_grid = np.zeros((self.nkpoints, qmpgrid.nnkpts, 3))
        kpbover2_grid_table = np.zeros((self.nkpoints, qmpgrid.nnkpts, 5),dtype= int)
        for ik, k in enumerate(self.k):
            for ib, b in enumerate(self.b_grid[self.nnkpts*ik:self.nnkpts*(ik+1)]):
                tmp_kpbover2, tmp_Gvec = self.fold_into_bz_Gs(k+b)
                idxkpbover2 = self.find_closest_kpoint(tmp_kpbover2)
                kpbover2_grid[ik,ib] = tmp_kpbover2
                kpbover2_grid_table[ik,ib] = [ik, idxkpbover2, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kpbover2_grid = kpbover2_grid
        self.kpbover2_grid_table = kpbover2_grid_table

    def get_kmqmbover2_grid(self, qmpgrid: 'NNKP_Grids'):       # need to improve this one
        if not isinstance(qmpgrid, NNKP_Grids):
            raise TypeError('Argument must be an instance of NNKP_Grids')
        #here I need to use the k-q grid and then apply -b/2
        kmqmbover2_grid = np.zeros((self.nkpoints, qmpgrid.nkpoints, qmpgrid.nnkpts,3))
        kmqmbover2_grid_table = np.zeros((self.nkpoints, qmpgrid.nkpoints, qmpgrid.nnkpts,5),dtype=int)
        for ik, k in enumerate(self.k):
            for iq, q in enumerate(qmpgrid.k):
                for ib, b in enumerate(self.b_grid[self.nnkpts*iq:self.nnkpts*(iq+1)]):
                    tmp_kmqmbover2, tmp_Gvec = self.fold_into_bz_Gs(k -q - b)
                    idxkmqmbover2 = self.find_closest_kpoint(tmp_kmqmbover2)
                    kmqmbover2_grid[ik, iq, ib] = tmp_kmqmbover2
                    kmqmbover2_grid_table[ik, iq, ib] = [ik, idxkmqmbover2, int(tmp_Gvec[0]), int(tmp_Gvec[1]), int(tmp_Gvec[2])]

        self.kmqmbover2_grid = kmqmbover2_grid
        self.kmqmbover2_grid_table = kmqmbover2_grid_table
