from yambopy.lattice import red_car
from yambopy.units import bohr2ang
import numpy as np
from yambopy.wannier.wann_io import NNKP
from yambopy.wannier.wann_kpoints import KPointGenerator
from yambopy.units import ang2bohr
import numpy as np
from scipy.spatial import cKDTree

class tb_Monkhorst_Pack(KPointGenerator):
    def __init__(self, grid_shape,latdb, shift=np.array([0.0,0.0,0.0])):
        super().__init__()
        self.grid_shape = grid_shape
        self.latdb = latdb
        self.rlat = self.latdb.rlat*2*np.pi*ang2bohr
        self.shift = shift
    def generate(self):
        # Create grid points for n1, n2, and n3
        NGX, NGY, NGZ = self.grid_shape        
        n1 = np.arange(NGX)
        n2 = np.arange(NGY)
        n3 = np.arange(NGZ)

        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])  
        e3 = np.array([0.0, 0.0, 1.0])  
        # Create a meshgrid for all combinations of n1, n2, and n3
        n1, n2, n3 = np.meshgrid(n1, n2, n3, indexing='ij')
        b1,b2,b3 = self.rlat
        # Calculate the k-points
        k_points = (n1[:, :, :, np.newaxis] / NGX) * b1 + (n2[:, :, :, np.newaxis] / NGY) * b2 + (n3[:, :, :, np.newaxis] / NGZ) * b3 + self.rlat@self.shift
        red_points = (n1[:, :, :, np.newaxis] / NGX) * e1 + (n2[:, :, :, np.newaxis] / NGY) * e2 + (n3[:, :, :, np.newaxis] / NGZ) * e3 + self.shift
        # Flatten the grid into a list of k-points
        self.k = red_points.reshape(-1, 3)
        self.red_kpoints = self.k
        self.car_kpoints = k_points.reshape(-1,3)
        self.nkpoints = len(self.k)
        self.k_tree = cKDTree(self.k)        