import numpy as np
from yambopy.wannier.wann_asegrid import ase_Monkhorst_Pack
from yambopy.wannier.wann_utils import *
from yambopy.wannier.wann_dipoles import TB_dipoles
from yambopy.wannier.wann_occupations import TB_occupations
from yambopy.dbs.bsekerneldb import *
from yambopy.wannier.wann_io import AMN
from scipy.linalg.lapack import zheev
import scipy
import gc

class ChernNumber():
    def __init__(self, h2p = None, h = None):
        
        if (h2p is not None): self.h2p = h2p
        if (h is not None): self.h = h

    def chern_exc_exc(self, integrand):
        """
        Compute the flux of $\frac{A* \partial A}{\partial q} through the planes x=0, y=0, and z=0.
        electron reference frame
        Parameters:
            integrand (callable): Function to evaluate the integrand. Takes q_grid as input.
            
        Returns:
            dict: Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_exc_exc number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double,self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk),dtype=np.complex128)
        # Create masks for points lying on the planes      
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        spacing_x = np.array([1/NX, 0.0, 0.0], dtype=np.float128)
        i_x = self.h2p.qmpgrid.find_closest_kpoint(spacing_x)
        spacing_y = np.array([0.0, 1/NY, 0.0], dtype=np.float128)
        i_y = self.h2p.qmpgrid.find_closest_kpoint(spacing_y)
        spacing_z = np.array([0.0, 0.0, 1/NZ], dtype=np.float128)
        i_z = self.h2p.qmpgrid.find_closest_kpoint(spacing_z)        
        qpdqx_grid = self.h2p.kplusq_table 

        # Extract points on each plane
        qpdx_plane = qpdqx_grid[:,i_x][:,1]#q_grid[qpdqx_grid[:,i_x][1]]
        qx_plane   = qpdqx_grid[:,i_x][:,0]#q_grid[qpdqx_grid[:,i_x][1]]
        qpdy_plane = qpdqx_grid[:,i_y][:,1]
        qy_plane   = qpdqx_grid[:,i_y][:,0]
        qpdz_plane = qpdqx_grid[:,i_z][:,1]
        qz_plane   = qpdqx_grid[:,i_z][:,0]
        # Evaluate the integrand at the points on each plane
        integrand_x = integrand[qx_plane].conj()*(integrand[qpdx_plane] - integrand[qx_plane])
        integrand_y = integrand[qy_plane].conj()*(integrand[qpdy_plane] - integrand[qy_plane])
        integrand_z = integrand[qz_plane].conj()*(integrand[qpdz_plane] - integrand[qz_plane])

        flux_x = np.sum(integrand_z+integrand_y,axis=(0,2,3,4)) 
        flux_y = np.sum(integrand_z+integrand_y,axis=(0,2,3,4)) 
        flux_z = np.sum(integrand_x+integrand_y,axis=(0,2,3,4)) 

        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

    def chern_e_e(self, integrand):
        """
        Compute the flux of $\frac{A* A }{\partial q} through the planes x=0, y=0, and z=0.
        electron reference frame
        Parameters:
            integrand (callable): Function to evaluate the integrand. Takes q_grid as input.
            
        Returns:
            dict: Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_e_e number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double,self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk),dtype=np.complex128)
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        # get overlaps valence w_{vv'k-q}
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        spacing_x = np.array([1/NX, 0.0, 0.0], dtype=np.float128)
        i_x = self.h2p.kmpgrid.find_closest_kpoint(spacing_x)
        spacing_y = np.array([0.0, 1/NY, 0.0], dtype=np.float128)
        i_y = self.h2p.kmpgrid.find_closest_kpoint(spacing_y)
        spacing_z = np.array([0.0, 0.0, 1/NZ], dtype=np.float128)
        i_z = self.h2p.kmpgrid.find_closest_kpoint(spacing_z)        
        qpdqx_grid = self.h2p.kplusq_table     

        qpdx_plane = qpdqx_grid[:,i_x][:,1]#q_grid[qpdqx_grid[:,i_x][1]]
        qx_plane   = qpdqx_grid[:,i_x][:,0]#q_grid[qpdqx_grid[:,i_x][1]]
        qpdy_plane = qpdqx_grid[:,i_y][:,1]
        qpdz_plane = qpdqx_grid[:,i_z][:,1]
        
        eigvec_ck = self.h2p.eigvec[qx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckdqx = self.h2p.eigvec[qpdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckdqy = self.h2p.eigvec[qpdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckdqz = self.h2p.eigvec[qpdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        dotcx = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckdqx-eigvec_ck), eigvec_ck) #l index is conjugated       
        dotcy = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckdqy-eigvec_ck), eigvec_ck) #l index is conjugated       
        dotcz = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckdqz-eigvec_ck), eigvec_ck) #l index is conjugated       

        # Evaluate the integrand at the points on each plane
        integrand_x = np.einsum('abcde,abcie, adi -> b',integrand.conj(), integrand, dotcx)/self.h2p.nq_double
        integrand_y = np.einsum('abcde,abcie, adi -> b',integrand.conj(), integrand, dotcy)/self.h2p.nq_double
        integrand_z = np.einsum('abcde,abcie, adi -> b',integrand.conj(), integrand, dotcz)/self.h2p.nq_double
        flux_x = (integrand_z+integrand_y)/2 # divide by 2 becuase I sum z and y
        flux_y = (integrand_z+integrand_y)/2 
        flux_z = (integrand_x+integrand_y)/2 
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z} 
    
    def chern_h_h(self, integrand):
        """
        Compute the flux of $\frac{A* A }{\partial q} through the planes x=0, y=0, and z=0.
        electron reference frame
        Parameters:
            integrand (callable): Function to evaluate the integrand. Takes q_grid as input.
            
        Returns:
            dict: Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_h_h number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double,self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk),dtype=np.complex128)
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        # get overlaps valence w_{vv'k-q}
        ikminusq = self.h2p.kminusq_table[:, :, 1]
        ikmqpdqx = kmqdq_grid_table[0][:,:,1] #dx
        ikmqpdqy = kmqdq_grid_table[1][:,:,1] #dy
        ikmqpdqz = kmqdq_grid_table[2][:,:,1] #dz
        eigvec_vkmq = self.h2p.eigvec[ikminusq, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqpdqx = self.h2p.eigvec[ikmqpdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqpdqy = self.h2p.eigvec[ikmqpdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqpdqz = self.h2p.eigvec[ikmqpdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        dotvx = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqpdqx-eigvec_vkmq), eigvec_vkmq) #np.conjugate(eigvec_vkmqpdqx-eigvec_vkmq)l index is conjugated       
        dotvy = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqpdqy-eigvec_vkmq), eigvec_vkmq) #l index is conjugated       
        dotvz = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqpdqz-eigvec_vkmq), eigvec_vkmq) #l index is conjugated       
        integrand_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand, dotvx)/self.h2p.nq_double
        integrand_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand, dotvy)/self.h2p.nq_double
        integrand_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand, dotvz)/self.h2p.nq_double
        flux_x = (integrand_z+integrand_y)/2 
        flux_y = (integrand_z+integrand_y)/2 
        flux_z = (integrand_x+integrand_y)/2 
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}     

    def chern_exc_e(self, integrand):
        """
        Compute the flux of $\frac{A* \partial A * w_cc'k}{\partial q} through the planes x=0, y=0, and z=0.
        electron reference frame
        Parameters:
            integrand (callable): Function to evaluate the integrand. Takes q_grid as input.
            
        Returns:
            dict: Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_exc_e number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double,self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk),dtype=np.complex128)
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        # get overlaps valence w_{vv'k-q}
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        spacing_x = np.array([1/NX, 0.0, 0.0], dtype=np.float128)
        i_x = self.h2p.kmpgrid.find_closest_kpoint(spacing_x)
        spacing_y = np.array([0.0, 1/NY, 0.0], dtype=np.float128)
        i_y = self.h2p.kmpgrid.find_closest_kpoint(spacing_y)
        spacing_z = np.array([0.0, 0.0, 1/NZ], dtype=np.float128)
        i_z = self.h2p.kmpgrid.find_closest_kpoint(spacing_z)        
        qpdqx_grid = self.h2p.kplusq_table     

        qpdx_plane = qpdqx_grid[:,i_x][:,1]#q_grid[qpdqx_grid[:,i_x][1]]
        qx_plane   = qpdqx_grid[:,i_x][:,0]#q_grid[qpdqx_grid[:,i_x][1]]
        qpdy_plane = qpdqx_grid[:,i_y][:,1]
        qpdz_plane = qpdqx_grid[:,i_z][:,1]
        
        eigvec_ck = self.h2p.eigvec[qx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckdqx = self.h2p.eigvec[qpdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckdqy = self.h2p.eigvec[qpdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckdqz = self.h2p.eigvec[qpdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        dotcx = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckdqx-eigvec_ck), eigvec_ck) #l index is conjugated       
        dotcy = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckdqy-eigvec_ck), eigvec_ck) #l index is conjugated       
        dotcz = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckdqz-eigvec_ck), eigvec_ck) #l index is conjugated       

        # Evaluate the integrand at the points on each plane
        integrand_x = np.einsum('abcde,abcie, adi -> b',integrand.conj(), integrand[qpdx_plane] - integrand[qx_plane], dotcx)/self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b', (integrand[qpdx_plane] - integrand[qx_plane]).conj(), integrand, dotcx)/self.h2p.nq_double
        integrand_y = np.einsum('abcde,abcie, adi -> b',integrand.conj(), integrand[qpdy_plane] - integrand[qx_plane], dotcy)/self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b', (integrand[qpdy_plane] - integrand[qx_plane]).conj(), integrand, dotcy)/self.h2p.nq_double
        integrand_z = np.einsum('abcde,abcie, adi -> b',integrand.conj(), integrand[qpdz_plane] - integrand[qx_plane], dotcz)/self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b', (integrand[qpdz_plane] - integrand[qx_plane]).conj(), integrand, dotcz)/self.h2p.nq_double
        flux_x = (integrand_z+integrand_y)/2 # divide by 2 becuase I sum z and y
        flux_y = (integrand_z+integrand_y)/2 
        flux_z = (integrand_x+integrand_y)/2 
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z} 

    def chern_exc_h(self, integrand):
        """
        Compute the flux of $\frac{A* \partial A * w_vvc'k}{\partial q} through the planes x=0, y=0, and z=0.
        electron reference frame
        Parameters:
            integrand (callable): Function to evaluate the integrand. Takes q_grid as input.
            
        Returns:
            dict: Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_exc_h number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double,self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk),dtype=np.complex128)
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        # get overlaps valence w_{vv'k-q}
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        spacing_x = np.array([1/NX, 0.0, 0.0], dtype=np.float128)
        i_x = self.h2p.kmpgrid.find_closest_kpoint(spacing_x)
        spacing_y = np.array([0.0, 1/NY, 0.0], dtype=np.float128)
        i_y = self.h2p.kmpgrid.find_closest_kpoint(spacing_y)
        spacing_z = np.array([0.0, 0.0, 1/NZ], dtype=np.float128)
        i_z = self.h2p.kmpgrid.find_closest_kpoint(spacing_z)        
        qpdqx_grid = self.h2p.kplusq_table     

        qpdx_plane = qpdqx_grid[:,i_x][:,1]#q_grid[qpdqx_grid[:,i_x][1]]
        qx_plane   = qpdqx_grid[:,i_x][:,0]#q_grid[qpdqx_grid[:,i_x][1]]
        qpdy_plane = qpdqx_grid[:,i_y][:,1]
        qpdz_plane = qpdqx_grid[:,i_z][:,1]

        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        # get overlaps valence w_{vv'k-q}
        ikminusq = self.h2p.kminusq_table[:, :, 1]
        ikmqpdqx = kmqdq_grid_table[0][:,:,1] #dx
        ikmqpdqy = kmqdq_grid_table[1][:,:,1] #dy
        ikmqpdqz = kmqdq_grid_table[2][:,:,1] #dz

        eigvec_vkmq = self.h2p.eigvec[ikminusq, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqpdqx = self.h2p.eigvec[ikmqpdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqpdqy = self.h2p.eigvec[ikmqpdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqpdqz = self.h2p.eigvec[ikmqpdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        dotvx = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqpdqx-eigvec_vkmq), eigvec_vkmq) #np.conjugate(eigvec_vkmqpdqx-eigvec_vkmq)l index is conjugated       
        dotvy = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqpdqy-eigvec_vkmq), eigvec_vkmq) #l index is conjugated       
        dotvz = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqpdqz-eigvec_vkmq), eigvec_vkmq) #l index is conjugated       

        # Evaluate the integrand at the points on each plane
        integrand_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand[qpdx_plane] - integrand[qx_plane], dotvx)/self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc -> b', (integrand[qpdx_plane] - integrand[qx_plane]).conj(), integrand, dotvx)/self.h2p.nq_double
        integrand_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand[qpdy_plane] - integrand[qx_plane], dotvy)/self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc -> b', (integrand[qpdy_plane] - integrand[qx_plane]).conj(), integrand, dotvy)/self.h2p.nq_double
        integrand_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand[qpdz_plane] - integrand[qx_plane], dotvz)/self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc-> b', (integrand[qpdz_plane] - integrand[qx_plane]).conj(), integrand, dotvz)/self.h2p.nq_double
        flux_x = (integrand_z+integrand_y)/2 # divide by 2 becuase I sum z and y
        flux_y = (integrand_z+integrand_y)/2 
        flux_z = (integrand_x+integrand_y)/2 
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z} 

    def curvature_plaquette(self):
        (kmqdq_grid, kmqdq_grid_table) = self.h2p.kmpgrid.get_kmqpdq_grid(self.h2p.qmpgrid)
        # get overlaps valence w_{vv'k-q}
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        plaquettes = {
        'x' : np.zeros((NY)*(NZ),dtype=np.complex128),
        'y' : np.zeros((NZ)*(NX),dtype=np.complex128),
        'z' : np.zeros((NX)*(NY),dtype=np.complex128),
        }
        spacing_x = np.array([1/NX, 0.0, 0.0], dtype=np.float128)
        i_x = self.h2p.kmpgrid.find_closest_kpoint(spacing_x)
        spacing_y = np.array([0.0, 1/NY, 0.0], dtype=np.float128)
        i_y = self.h2p.kmpgrid.find_closest_kpoint(spacing_y)
        spacing_xy = np.array([1/NX, 1/NY, 0.0], dtype=np.float128)
        i_xy = self.h2p.kmpgrid.find_closest_kpoint(spacing_xy)   
        spacing_zx = np.array([1/NX, 0.0, 1/NZ], dtype=np.float128)
        i_zx = self.h2p.kmpgrid.find_closest_kpoint(spacing_zx)           
        spacing_yz = np.array([0.0, 1/NY, 1/NZ], dtype=np.float128)
        i_yz = self.h2p.kmpgrid.find_closest_kpoint(spacing_yz)                
        spacing_z = np.array([0.0, 0.0, 1/NZ], dtype=np.float128)
        i_z = self.h2p.kmpgrid.find_closest_kpoint(spacing_z)        
        qpdqx_grid = self.h2p.kplusq_table          
        
        qpdx_plane = qpdqx_grid[:,i_x][:,1]#q_grid[qpdqx_grid[:,i_x][1]]
        qx_plane   = qpdqx_grid[:,i_x][:,0]#q_grid[qpdqx_grid[:,i_x][1]]
        qpdy_plane = qpdqx_grid[:,i_y][:,1]
        qpdxy_plane = qpdqx_grid[:,i_xy][:,1]
        qpdyz_plane = qpdqx_grid[:,i_yz][:,1]
        qpdzx_plane = qpdqx_grid[:,i_zx][:,1]        
        qpdz_plane = qpdqx_grid[:,i_z][:,1]        

        plaquettes['x'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdy_plane].conj(),self.h2p.h2peigvec[qx_plane])  \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdyz_plane].conj(),self.h2p.h2peigvec[qpdy_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdz_plane].conj(),self.h2p.h2peigvec[qpdyz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qx_plane].conj(),self.h2p.h2peigvec[qpdz_plane])  
        plaquettes['y'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdz_plane].conj(),self.h2p.h2peigvec[qx_plane])  \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdzx_plane].conj(),self.h2p.h2peigvec[qpdz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdx_plane].conj(),self.h2p.h2peigvec[qpdzx_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qx_plane].conj(),self.h2p.h2peigvec[qpdx_plane])   
        plaquettes['z'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdx_plane].conj(),self.h2p.h2peigvec[qx_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdxy_plane].conj(),self.h2p.h2peigvec[qpdx_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qpdy_plane].conj(),self.h2p.h2peigvec[qpdxy_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[qx_plane].conj(),self.h2p.h2peigvec[qpdy_plane])                                      
        return plaquettes