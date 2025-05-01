import numpy as np
from yambopy.wannier.wann_utils import ensure_shape

class ChernNumber():
    def __init__(self, h2p=None, h=None):
        if h2p is not None:
            self.h2p = h2p
            self.NX, self.NY, self.NZ = self.h2p.qmpgrid.grid_shape
            self.spacing_x = np.array([1/self.NX, 0.0, 0.0], dtype=np.float128)
            self.i_x = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_x)
            self.spacing_y = np.array([0.0, 1/self.NY, 0.0], dtype=np.float128)
            self.i_y = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_y)
            self.spacing_z = np.array([0.0, 0.0, 1/self.NZ], dtype=np.float128)
            self.i_z = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_z)
            self.qpdqx_grid = self.h2p.kplusq_table
            self.spacing_xy = np.array([1/self.NX, 1/self.NY, 0.0], dtype=np.float128)
            self.i_xy = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_xy)
            self.spacing_zx = np.array([1/self.NX, 0.0, 1/self.NZ], dtype=np.float128)
            self.i_zx = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_zx)
            self.spacing_yz = np.array([0.0, 1/self.NY, 1/self.NZ], dtype=np.float128)
            self.i_yz = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_yz)
            self.qpdx_plane = self.qpdqx_grid[:, self.i_x][:, 1]
            self.qx_plane = self.qpdqx_grid[:, self.i_x][:, 0]
            self.qpdy_plane = self.qpdqx_grid[:, self.i_y][:, 1]
            self.qy_plane = self.qpdqx_grid[:, self.i_y][:, 0]
            self.qpdz_plane = self.qpdqx_grid[:, self.i_z][:, 1]
            self.qz_plane = self.qpdqx_grid[:, self.i_z][:, 0]
            self.qpdxy_plane = self.qpdqx_grid[:, self.i_xy][:, 1]
            self.qpdyz_plane = self.qpdqx_grid[:, self.i_yz][:, 1]
            self.qpdzx_plane = self.qpdqx_grid[:, self.i_zx][:, 1]

        if h is not None:
            self.h = h

    def chern_exc_exc(self, integrand):
        """
        Compute the flux of $\frac{A^* \partial A}{\partial q}$ through the planes x=0, y=0, and z=0
        in the electron reference frame.

        Parameters
        ----------
        integrand : callable
            Function to evaluate the integrand. Takes q_grid as input.

        Returns
        -------
        dict
            Fluxes through planes: {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            This corresponds to the Chern number from exciton-exciton interaction
            if `integrand` are the BSE eigenvectors.
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double, self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk), dtype=np.complex128)
        integrand_x = integrand[self.qx_plane].conj() * (integrand[self.qpdx_plane] - integrand[self.qx_plane])
        integrand_y = integrand[self.qy_plane].conj() * (integrand[self.qpdy_plane] - integrand[self.qy_plane])
        integrand_z = integrand[self.qz_plane].conj() * (integrand[self.qpdz_plane] - integrand[self.qz_plane])
        flux_x = np.sum(integrand_z + integrand_y, axis=(0, 2, 3, 4))
        flux_y = np.sum(integrand_z + integrand_y, axis=(0, 2, 3, 4))
        flux_z = np.sum(integrand_x + integrand_y, axis=(0, 2, 3, 4))
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

    def chern_e_e(self, integrand):
        """
        Compute the flux of :math:`\frac{A* A }{\partial q}` through the planes x=0, y=0, and z=0.
        electron reference frame

        Parameters
        ----------
        integrand : callable
            Function to evaluate the integrand. Takes q_grid as input.

        Returns
        -------
        dict
            Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_e_e number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double, self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk), dtype=np.complex128)
        eigvec_ck = self.h2p.eigvec[self.qx_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        eigvec_ckdqx = self.h2p.eigvec[self.qpdx_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        eigvec_ckdqy = self.h2p.eigvec[self.qpdy_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        eigvec_ckdqz = self.h2p.eigvec[self.qpdz_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        dotcx = np.einsum('jkl,jkp->jlp', np.conjugate(eigvec_ckdqx - eigvec_ck), eigvec_ck)
        dotcy = np.einsum('jkl,jkp->jlp', np.conjugate(eigvec_ckdqy - eigvec_ck), eigvec_ck)
        dotcz = np.einsum('jkl,jkp->jlp', np.conjugate(eigvec_ckdqz - eigvec_ck), eigvec_ck)
        integrand_x = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand, dotcx) / self.h2p.nq_double
        integrand_y = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand, dotcy) / self.h2p.nq_double
        integrand_z = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand, dotcz) / self.h2p.nq_double
        flux_x = (integrand_z + integrand_y) / 2
        flux_y = (integrand_z + integrand_y) / 2
        flux_z = (integrand_x + integrand_y) / 2
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

    def chern_h_h(self, integrand):
        """
        Compute the flux of $\frac{A* A }{\partial q}$ through the planes x=0, y=0, and z=0
        in the electron reference frame.

        Parameters
        ----------
        integrand : callable
            Function to evaluate the integrand. Takes q_grid as input.

        Returns
        -------
        dict
            Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            This corresponds to the chern_h_h number if integrand are the bse eigenvectors.
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double, self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk), dtype=np.complex128)
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        ikminusq = self.h2p.kminusq_table[:, :, 1]
        ikmqpdqx = kmqdq_grid_table[0][:, :, 1]
        ikmqpdqy = kmqdq_grid_table[1][:, :, 1]
        ikmqpdqz = kmqdq_grid_table[2][:, :, 1]
        eigvec_vkmq = self.h2p.eigvec[ikminusq, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        eigvec_vkmqpdqx = self.h2p.eigvec[ikmqpdqx, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        eigvec_vkmqpdqy = self.h2p.eigvec[ikmqpdqy, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        eigvec_vkmqpdqz = self.h2p.eigvec[ikmqpdqz, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        dotvx = np.einsum('ijkl,ijkp->ijlp', np.conjugate(eigvec_vkmqpdqx - eigvec_vkmq), eigvec_vkmq)
        dotvy = np.einsum('ijkl,ijkp->ijlp', np.conjugate(eigvec_vkmqpdqy - eigvec_vkmq), eigvec_vkmq)
        dotvz = np.einsum('ijkl,ijkp->ijlp', np.conjugate(eigvec_vkmqpdqz - eigvec_vkmq), eigvec_vkmq)
        integrand_x = np.einsum('abcde,abhde, eahc -> b', integrand.conj(), integrand, dotvx) / self.h2p.nq_double
        integrand_y = np.einsum('abcde,abhde, eahc -> b', integrand.conj(), integrand, dotvy) / self.h2p.nq_double
        integrand_z = np.einsum('abcde,abhde, eahc -> b', integrand.conj(), integrand, dotvz) / self.h2p.nq_double
        flux_x = (integrand_z + integrand_y) / 2
        flux_y = (integrand_z + integrand_y) / 2
        flux_z = (integrand_x + integrand_y) / 2
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

    def chern_exc_e(self, integrand):
        """
        Compute the flux of $\frac{A* \partial A * w_cc'k}{\partial q} through the planes x=0, y=0, and z=0.
        electron reference frame

        Parameters
        ----------
        integrand : callable
            Function to evaluate the integrand. Takes q_grid as input.

        Returns
        -------
        dict
            Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_exc_e number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double, self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk), dtype=np.complex128)
        eigvec_ck = self.h2p.eigvec[self.qx_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        eigvec_ckdqx = self.h2p.eigvec[self.qpdx_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        eigvec_ckdqy = self.h2p.eigvec[self.qpdy_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        eigvec_ckdqz = self.h2p.eigvec[self.qpdz_plane, :, :][:, :, np.unique(self.h2p.BSE_table[:, 2])]
        dotcx = np.einsum('jkl,jkp->jlp', np.conjugate(eigvec_ckdqx - eigvec_ck), eigvec_ck)
        dotcy = np.einsum('jkl,jkp->jlp', np.conjugate(eigvec_ckdqy - eigvec_ck), eigvec_ck)
        dotcz = np.einsum('jkl,jkp->jlp', np.conjugate(eigvec_ckdqz - eigvec_ck), eigvec_ck)
        integrand_x = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand[self.qpdx_plane] - integrand[self.qx_plane], dotcx) / self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b', (integrand[self.qpdx_plane] - integrand[self.qx_plane]).conj(), integrand, dotcx) / self.h2p.nq_double
        integrand_y = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand[self.qpdy_plane] - integrand[self.qx_plane], dotcy) / self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b', (integrand[self.qpdy_plane] - integrand[self.qx_plane]).conj(), integrand, dotcy) / self.h2p.nq_double
        integrand_z = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand[self.qpdz_plane] - integrand[self.qx_plane], dotcz) / self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b', (integrand[self.qpdz_plane] - integrand[self.qx_plane]).conj(), integrand, dotcz) / self.h2p.nq_double
        flux_x = (integrand_z + integrand_y) / 2
        flux_y = (integrand_z + integrand_y) / 2
        flux_z = (integrand_x + integrand_y) / 2
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

    def chern_exc_h(self, integrand):
        """
        Compute the flux of :math:`\frac{A* \partial A * w_vvc'k}{\partial q}`
        through the planes x=0, y=0, and z=0.

        Parameters
        ----------
        integrand : callable
            Function to evaluate the integrand. Takes :code:`q_grid` as input.

        Returns
        -------
        dict
            Fluxes through planes :code:`{'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}`.
            It corresponds to the chern_exc_h number if :code:`integrand` are the BSE eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double, self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk), dtype=np.complex128)
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        ikminusq = self.h2p.kminusq_table[:, :, 1]
        ikmqpdqx = kmqdq_grid_table[0][:, :, 1]
        ikmqpdqy = kmqdq_grid_table[1][:, :, 1]
        ikmqpdqz = kmqdq_grid_table[2][:, :, 1]
        eigvec_vkmq = self.h2p.eigvec[ikminusq, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        eigvec_vkmqpdqx = self.h2p.eigvec[ikmqpdqx, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        eigvec_vkmqpdqy = self.h2p.eigvec[ikmqpdqy, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        eigvec_vkmqpdqz = self.h2p.eigvec[ikmqpdqz, :, :][:, :, :, np.unique(self.h2p.BSE_table[:, 1])]
        dotvx = np.einsum('ijkl,ijkp->ijlp', np.conjugate(eigvec_vkmqpdqx - eigvec_vkmq), eigvec_vkmq)
        dotvy = np.einsum('ijkl,ijkp->ijlp', np.conjugate(eigvec_vkmqpdqy - eigvec_vkmq), eigvec_vkmq)
        dotvz = np.einsum('ijkl,ijkp->ijlp', np.conjugate(eigvec_vkmqpdqz - eigvec_vkmq), eigvec_vkmq)
        integrand_x = np.einsum('abcde,abhde, eahc -> b', integrand.conj(), integrand[self.qpdx_plane] - integrand[self.qx_plane], dotvx) / self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc -> b', (integrand[self.qpdx_plane] - integrand[self.qx_plane]).conj(), integrand, dotvx) / self.h2p.nq_double
        integrand_y = np.einsum('abcde,abhde, eahc -> b', integrand.conj(), integrand[self.qpdy_plane] - integrand[self.qx_plane], dotvy) / self.h2p.nq_double \
                    + np.einsum('abcde,abhde, e
