import sisl
import spglib
from yambopy.lattice import red_car
import numpy as np
from yambopy.wannier.wann_io import NNKP
from yambopy.wannier.wann_kpoints import KPointGenerator
from ase.dft.kpoints import monkhorst_pack

class tb_Monkhorst_Pack(KPointGenerator):
    def __init__(self, grid_shape):
        super().__init__()
        self.grid_shape = grid_shape

    def generate(self):
        # Use ASE's monkhorst_pack to generate the k-points
        self.k = monkhorst_pack(self.grid_shape)
        self.nkpoints = len(self.k)
        self.weights = 1/(self.nkpoints)
        print(f"Generated {self.nkpoints} k-points using ASE.")