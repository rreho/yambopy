import sisl
import spglib
from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_kpoints import KPointGenerator
from yambopy.wannier.wann_io import NNKP

class NNKP_Grid(KPointGenerator):
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
        self.car_kpoins = red_car(self.k, self.rlat)
        self.red_kpoints = self.nnkp_grid.k
        self.nkpoints = len(self.k)
        self.weights = 1/self.nkpoints
