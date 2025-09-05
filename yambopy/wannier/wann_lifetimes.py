import numpy as np
from yambopy.wannier.wann_dipoles import TB_dipoles
from yambopy.wannier.wann_utils import HA2EV, BOHR2ANG



class TB_lifetimes(TB_dipoles):
    ''' compute the lifetimes of excitons for 3D, 2D and 1D systems
    $$
    \tau_{3 D, \alpha, \beta}^n=\frac{3 c^2 \hbar^3 N_{\mathbf{k}}}{4 \chi\left(E_0^n\right)^3 F_{\alpha, \beta}^{n, B S E}}
    $$
    $$
    \tau_{2 D, \alpha, \beta}^n=\frac{\hbar A_{u c} N_{\mathbf{k}}}{8 \pi \chi E_0^n F_{\alpha, \beta}^{n, B S E}}
    $$
    $$
    \tau_{1 D, \alpha, \beta}^n=\frac{c \hbar^2 l_1 N_{\mathbf{k}}}{2 \pi \chi\left(E_0^n\right)^2 F_{\alpha, \beta}^{n, B S E}}
    $$
    '''
    def __init__(self, tb_dipoles, dim):
        self.dim = dim

        if hasattr(tb_dipoles, 'cpot') and hasattr(tb_dipoles.cpot, 'lattice'):
            self.latdb = tb_dipoles.cpot.lattice
        else:
            raise AttributeError("TB_dipoles does not have the expected 'cpot.lattice' attribute")

        if hasattr(tb_dipoles, 'h2peigvec'):
            self.h2peigvec = tb_dipoles.h2peigvec
        else:
            raise AttributeError('Before computing lifetimes you need the excitonic energies.')

        if self.dim == '3D':
            self.tau = self._get_tau3D(tb_dipoles)
        elif self.dim == '2D':
            self.tau = self._get_tau2D(tb_dipoles)
        elif self.dim == '1D':
            self.tau = self._get_tau1D(tb_dipoles)

    def _get_tau3D(self, tb_dipoles):
        tau = np.zeros((tb_dipoles.ntransitions, 3, 3))
        F_kcv = tb_dipoles.F_kcv
        h2peigvec = tb_dipoles.h2peigvec / HA2EV
        dipvec = np.einsum('tp,txy->txy',h2peigvec**3,F_kcv)
        tau = 3 * tb_dipoles.nkpoints / (4 * dipvec)
        return tau

    def _get_tau2D(self, tb_dipoles):
        tau = np.zeros((tb_dipoles.ntransitions, 3, 3))
        F_kcv = tb_dipoles.F_kcv
        vc = np.linalg.norm(np.cross(self.latdb.lat[0], self.latdb.lat[1])*BOHR2ANG**2)
        h2peigvec = tb_dipoles.h2peigvec / HA2EV
        dipvec = np.einsum('tp,txy->txy',h2peigvec,F_kcv)
        tau = vc * tb_dipoles.nkpoints / (8 * np.pi * dipvec)
        return tau

    def _get_tau1D(self, tb_dipoles):
        tau = np.zeros((tb_dipoles.ntransitions))
        F_kcv = tb_dipoles.F_kcv
        vc = np.linalg.norm(self.latdb.lat[0]*BOHR2ANG**2)
        h2peigvec = tb_dipoles.h2peigvec / HA2EV
        dipvec = np.einsum('tp,txy->txy',h2peigvec**2,F_kcv)
        tau = vc * tb_dipoles.nkpoints / (2 * np.pi * dipvec)
        return tau
