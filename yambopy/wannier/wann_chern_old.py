import numpy as np
from yambopy.wannier.wann_utils import ensure_shape
class ChernNumber():
    def __init__(self, h2p = None, h = None, a=0.0, b=0.0, c=1.0, d=0.0):
        """
        Parameters:
            h2p : the H2P Hamiltonian with bse eigenvectors that will be used to compute the Chern number
            a, b, c,d  : coefficients for plane equation in momentum space. Default is the x-y plane
        """        
        if (h2p is not None): 
            self.h2p = h2p
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.qx_plane = self.h2p.qmpgrid.find_kpoints_inplane(1.0, 0.0, 0.0, 0.0)
            self.qy_plane = self.h2p.qmpgrid.find_kpoints_inplane(0.0, 1.0, 0.0, 0.0)
            self.qz_plane = self.h2p.qmpgrid.find_kpoints_inplane(0.0, 0.0 , 1.0, 0.0)
            self.q0_plane = self.h2p.qmpgrid.find_kpoints_inplane(self.a, self.b ,self.c, self.d)
            self.NX, self.NY, self.NZ = self.h2p.qmpgrid.grid_shape
            self.spacing_x = np.array([1/self.NX, 0.0, 0.0], dtype=np.float64)
            # Create masks for points lying on the planes      
            self.i_x = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_x)
            self.spacing_y = np.array([0.0, 1/self.NY, 0.0], dtype=np.float64)
            self.i_y = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_y)
            self.spacing_z = np.array([0.0, 0.0, 1/self.NZ], dtype=np.float64)
            self.i_z = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_z)        
            self.qpdqx_grid = self.h2p.kplusq_table     
            self.spacing_xy = np.array([1/self.NX, 1/self.NY, 0.0], dtype=np.float64)
            self.i_xy = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_xy)   
            self.spacing_zx = np.array([1/self.NX, 0.0, 1/self.NZ], dtype=np.float64)
            self.i_zx = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_zx)           
            self.spacing_yz = np.array([0.0, 1/self.NY, 1/self.NZ], dtype=np.float64)
            self.i_yz = self.h2p.kmpgrid.find_closest_kpoint(self.spacing_yz)               
            # Extract points on each plane
            self.qpdx_plane = self.qpdqx_grid[self.qx_plane,self.i_x][:,1]#q_grid[qpdqx_grid[:,i_x][1]]
            self.qx_plane   = self.qpdqx_grid[self.qx_plane,self.i_x][:,0]#q_grid[qpdqx_grid[:,i_x][1]]
            self.nx_plane = len(self.qx_plane)
            self.qpdy_plane = self.qpdqx_grid[self.qy_plane,self.i_y][:,1]
            self.qy_plane   = self.qpdqx_grid[self.qy_plane,self.i_y][:,0]
            self.ny_plane = len(self.qx_plane)
            self.qpdz_plane = self.qpdqx_grid[self.qz_plane,self.i_z][:,1]
            self.qz_plane   = self.qpdqx_grid[self.qz_plane,self.i_z][:,0]
            self.nz_plane = len(self.qz_plane)
            self.qpdxy_plane = self.qpdqx_grid[self.qz_plane,self.i_xy][:,1]
            self.qxy_plane = self.qpdqx_grid[self.qz_plane,self.i_xy][:,0]
            self.qpdyz_plane = self.qpdqx_grid[self.qx_plane,self.i_yz][:,1]
            self.qyz_plane = self.qpdqx_grid[self.qx_plane,self.i_yz][:,0]
            self.qpdzx_plane = self.qpdqx_grid[self.qy_plane,self.i_zx][:,1]
            self.qzx_plane = self.qpdqx_grid[self.qy_plane,self.i_zx][:,0]                  
            self.qxdy_plane = self.qpdqx_grid[self.qx_plane,self.i_y][:,1] # x = 0 + y       
            self.qxdydz_plane = self.qpdqx_grid[self.qx_plane, self.i_yz][:,1] # x = 0 + y + z
            self.qxdz_plane = self.qpdqx_grid[self.qx_plane, self.i_z][:,1] # x = 0 + z
            self.qydz_plane = self.qpdqx_grid[self.qy_plane,self.i_z][:,1] # x = 0 + y       
            self.qydxdz_plane = self.qpdqx_grid[self.qy_plane, self.i_zx][:,1] # x = 0 + y + z
            self.qydx_plane = self.qpdqx_grid[self.qy_plane, self.i_x][:,1] # x = 0 + z
            self.qzdx_plane = self.qpdqx_grid[self.qz_plane,self.i_x][:,1] # x = 0 + y       
            self.qzdxdy_plane = self.qpdqx_grid[self.qz_plane, self.i_xy][:,1] # x = 0 + y + z
            self.qzdy_plane = self.qpdqx_grid[self.qz_plane, self.i_y][:,1] # x = 0 + z                               

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

        # Evaluate the integrand at the points on each plane
        integrand_x = (integrand[self.qx_plane].conj()*(integrand[self.qpdx_plane] - integrand[self.qx_plane]))/self.nx_plane + \
                      (integrand[self.qx_plane].conj()*(integrand[self.qxdy_plane] - integrand[self.qx_plane]))/self.nx_plane + \
                      (integrand[self.qx_plane].conj()*(integrand[self.qxdz_plane] - integrand[self.qx_plane]))/self.nx_plane 
        integrand_y = (integrand[self.qy_plane].conj()*(integrand[self.qpdy_plane] - integrand[self.qy_plane]))/self.ny_plane + \
                      (integrand[self.qy_plane].conj()*(integrand[self.qydx_plane] - integrand[self.qy_plane]))/self.ny_plane + \
                      (integrand[self.qy_plane].conj()*(integrand[self.qydz_plane] - integrand[self.qy_plane]))/self.ny_plane
        integrand_z = (integrand[self.qz_plane].conj()*(integrand[self.qpdz_plane] - integrand[self.qz_plane]))/self.nz_plane + \
                      (integrand[self.qz_plane].conj()*(integrand[self.qzdx_plane] - integrand[self.qz_plane]))/self.nz_plane +\
                      (integrand[self.qz_plane].conj()*(integrand[self.qzdy_plane] - integrand[self.qz_plane]))/self.nz_plane
        
        flux_x = np.sum(integrand_z, axis=(0,2,3,4)) + np.sum(integrand_y,axis=(0,2,3,4)) 
        flux_y = np.sum(integrand_z, axis=(0,2,3,4)) + np.sum(integrand_y,axis=(0,2,3,4)) 
        flux_z = np.sum(integrand_x, axis=(0,2,3,4)) + np.sum(integrand_y,axis=(0,2,3,4))  

        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}

    def chern_e_e(self, integrand):
        """
        Compute the flux of $A^{\lambda q*}_{c^\prime v k}*A^{\lambda q}_{cv k} w_{cc^\primek}$ through the planes x=0, y=0, and z=0.
        electron reference frame
        Parameters:
            integrand (callable): Function to evaluate the integrand. Takes q_grid as input.
            
        Returns:
            dict: Fluxes through planes {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z}.
            it corresponds to the chern_e_e number if integrand are the bse eigenvectors
        """
        integrand = ensure_shape(integrand, (self.h2p.nq_double,self.h2p.dimbse, self.h2p.bse_nv, self.h2p.bse_nc, self.h2p.nk),dtype=np.complex128)
        #conduction bands tensors
        eigvec_ckx = self.h2p.eigvec[self.qx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckxdqx = self.h2p.eigvec[self.qpdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxdqy = self.h2p.eigvec[self.qxdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxdqz = self.h2p.eigvec[self.qxdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_cky = self.h2p.eigvec[self.qy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckydqx = self.h2p.eigvec[self.qydx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckydqy = self.h2p.eigvec[self.qpdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckydqz = self.h2p.eigvec[self.qydz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckz = self.h2p.eigvec[self.qz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckzdqx = self.h2p.eigvec[self.qzdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzdqy = self.h2p.eigvec[self.qzdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzdqz = self.h2p.eigvec[self.qpdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        # dot product q point belong on the yz-plane (x=0 or .qx_plane)
        dotcx_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqx-eigvec_ckx), eigvec_ckx)
        dotcx_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqy-eigvec_ckx), eigvec_ckx) 
        dotcx_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqz-eigvec_ckx), eigvec_ckx) 
        # dot product q point belong on the zx-plane (y=0 or .qy_plane)
        dotcy_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqx-eigvec_cky), eigvec_cky)
        dotcy_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqy-eigvec_cky), eigvec_cky) 
        dotcy_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqz-eigvec_cky), eigvec_cky) 
        # dot product q point belong on the xy-plane (z=0 or .qy_plane)
        dotcz_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqx-eigvec_ckz), eigvec_ckz)
        dotcz_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqy-eigvec_ckz), eigvec_ckz) 
        dotcz_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqz-eigvec_ckz), eigvec_ckz) 
        # Evaluate the integrand at the points on each plane
        integrandx_x = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qx_plane], integrand[self.qx_plane], dotcx_x)/self.nx_plane
        integrandx_y = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qx_plane], integrand[self.qx_plane], dotcx_y)/self.nx_plane
        integrandx_z = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qx_plane], integrand[self.qx_plane], dotcx_z)/self.nx_plane
        integrandy_x = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qy_plane], integrand[self.qy_plane], dotcy_x)/self.ny_plane
        integrandy_y = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qy_plane], integrand[self.qy_plane], dotcy_y)/self.ny_plane
        integrandy_z = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qy_plane], integrand[self.qy_plane], dotcy_z)/self.ny_plane
        integrandz_x = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qz_plane], integrand[self.qz_plane], dotcz_x)/self.nz_plane
        integrandz_y = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qz_plane], integrand[self.qz_plane], dotcz_y)/self.nz_plane
        integrandz_z = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.qz_plane], integrand[self.qz_plane], dotcz_z)/self.nz_plane                
        flux_x = (integrandx_x+integrandx_y+integrandx_z) # divide by 2 becuase I sum z and y (?)
        flux_y = (integrandy_x+integrandy_y+integrandy_z)
        flux_z = (integrandz_x+integrandz_y+integrandz_z)
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
        ikminusqx = self.h2p.kminusq_table[:, self.qx_plane, 1]
        ikminusqy = self.h2p.kminusq_table[:, self.qy_plane, 1]
        ikminusqz = self.h2p.kminusq_table[:, self.qz_plane, 1]
        # now I need to get the k-q-dx/dy/dz component for q being in the yz (qx_plane), zx (qy_plane) or xy (qz_plane) planes
        ikmqxpdqx = kmqdq_grid_table[0][:,self.qx_plane,1] #dx
        ikmqxpdqy = kmqdq_grid_table[1][:,self.qx_plane,1] #dy
        ikmqxpdqz = kmqdq_grid_table[2][:,self.qx_plane,1] #dz

        ikmqypdqx = kmqdq_grid_table[0][:,self.qy_plane,1] #dx
        ikmqypdqy = kmqdq_grid_table[1][:,self.qy_plane,1] #dy
        ikmqypdqz = kmqdq_grid_table[2][:,self.qy_plane,1] #dz

        ikmqzpdqx = kmqdq_grid_table[0][:,self.qz_plane,1] #dx
        ikmqzpdqy = kmqdq_grid_table[1][:,self.qz_plane,1] #dy
        ikmqzpdqz = kmqdq_grid_table[2][:,self.qz_plane,1] #dz

        eigvec_vkmqx = self.h2p.eigvec[ikminusqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqy = self.h2p.eigvec[ikminusqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqz = self.h2p.eigvec[ikminusqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqxpdqx = self.h2p.eigvec[ikmqxpdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxpdqy = self.h2p.eigvec[ikmqxpdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxpdqz = self.h2p.eigvec[ikmqxpdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqypdqx = self.h2p.eigvec[ikmqypdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqypdqy = self.h2p.eigvec[ikmqypdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqypdqz = self.h2p.eigvec[ikmqypdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzpdqx = self.h2p.eigvec[ikmqzpdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzpdqy = self.h2p.eigvec[ikmqzpdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzpdqz = self.h2p.eigvec[ikmqzpdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
       
        dotvx_x = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqxpdqx-eigvec_vkmqx), eigvec_vkmqx)
        dotvx_y = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqxpdqy-eigvec_vkmqy), eigvec_vkmqy)
        dotvx_z = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqxpdqz-eigvec_vkmqz), eigvec_vkmqz)             
        
        dotvy_x = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqypdqx-eigvec_vkmqx), eigvec_vkmqx) #l index is conjugated       
        dotvy_y = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqypdqy-eigvec_vkmqy), eigvec_vkmqy) #l index is conjugated       
        dotvy_z = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqypdqz-eigvec_vkmqz), eigvec_vkmqz) #l index is conjugated               
        
        dotvz_x = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqzpdqx-eigvec_vkmqx), eigvec_vkmqx) #l index is conjugated       
        dotvz_y = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqzpdqy-eigvec_vkmqy), eigvec_vkmqy) #l index is conjugated       
        dotvz_z = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqzpdqz-eigvec_vkmqz), eigvec_vkmqz) #l index is conjugated       

        integrandx_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qx_plane], integrand[self.qx_plane], dotvx_x)/self.h2p.nx_double
        integrandx_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qx_plane], integrand[self.qx_plane], dotvx_y)/self.h2p.nx_double
        integrandx_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qx_plane], integrand[self.qx_plane], dotvx_z)/self.h2p.nx_double       
        integrandy_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qy_plane], integrand[self.qy_plane], dotvy_x)/self.h2p.ny_double
        integrandy_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qy_plane], integrand[self.qy_plane], dotvy_y)/self.h2p.ny_double
        integrandy_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qy_plane], integrand[self.qy_plane], dotvy_z)/self.h2p.ny_double
        integrandz_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qz_plane], integrand[self.qz_plane], dotvz_x)/self.h2p.nz_double
        integrandz_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qz_plane], integrand[self.qz_plane], dotvz_y)/self.h2p.nz_double
        integrandz_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.qz_plane], integrand[self.qz_plane], dotvz_z)/self.h2p.nz_double

        flux_x = integrandx_x+integrandx_y+integrandx_z 
        flux_y = integrandy_x+integrandy_y+integrandy_z
        flux_z = integrandz_x+integrandz_y+integrandz_z 
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
        #x
        eigvec_ckx    = self.h2p.eigvec[self.qx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckxdqx = self.h2p.eigvec[self.qpdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxdqy = self.h2p.eigvec[self.qxdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxdqz = self.h2p.eigvec[self.qxdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        #y
        eigvec_cky    = self.h2p.eigvec[self.qy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckydqx = self.h2p.eigvec[self.qydx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckydqy = self.h2p.eigvec[self.qpdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckydqz = self.h2p.eigvec[self.qydz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        #z
        eigvec_ckz    = self.h2p.eigvec[self.qz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckzdqx = self.h2p.eigvec[self.qzdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzdqy = self.h2p.eigvec[self.qzdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzdqz = self.h2p.eigvec[self.qpdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        
        dotcx_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqx-eigvec_ckx), eigvec_ckx) #l index is conjugated  
        dotcx_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqy-eigvec_ckx), eigvec_ckx) #l index is conjugated  
        dotcx_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqz-eigvec_ckx), eigvec_ckx) #l index is conjugated       
        dotcy_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqx-eigvec_cky), eigvec_cky) #l index is conjugated  
        dotcy_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqy-eigvec_cky), eigvec_cky) #l index is conjugated  
        dotcy_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqz-eigvec_cky), eigvec_cky) #l index is conjugated      
        dotcz_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqx-eigvec_ckz), eigvec_ckz) #l index is conjugated  
        dotcz_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqy-eigvec_ckz), eigvec_ckz) #l index is conjugated  
        dotcz_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqz-eigvec_ckz), eigvec_ckz) #l index is conjugated      

        # prepare vector for 
        # Evaluate the integrand at the points on each plane
        integrand_x = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand[self.qpdx_plane] - integrand[self.qx_plane], dotcx)/self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b',(integrand[self.qpdx_plane] - integrand[self.qx_plane]).conj(), integrand, dotcx)/self.h2p.nq_double
        integrand_y = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand[self.qpdy_plane] - integrand[self.qx_plane], dotcy)/self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b',(integrand[self.qpdy_plane] - integrand[self.qx_plane]).conj(), integrand, dotcy)/self.h2p.nq_double
        integrand_z = np.einsum('abcde,abcie, adi -> b', integrand.conj(), integrand[self.qpdz_plane] - integrand[self.qx_plane], dotcz)/self.h2p.nq_double \
                    + np.einsum('abcde,abcie, adi -> b',(integrand[self.qpdz_plane] - integrand[self.qx_plane]).conj(), integrand, dotcz)/self.h2p.nq_double
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
        integrand_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand[self.qpdx_plane] - integrand[self.qx_plane], dotvx)/self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc -> b', (integrand[self.qpdx_plane] - integrand[self.qx_plane]).conj(), integrand, dotvx)/self.h2p.nq_double
        integrand_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand[self.qpdy_plane] - integrand[self.qx_plane], dotvy)/self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc -> b', (integrand[self.qpdy_plane] - integrand[self.qx_plane]).conj(), integrand, dotvy)/self.h2p.nq_double
        integrand_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj(), integrand[self.qpdz_plane] - integrand[self.qx_plane], dotvz)/self.h2p.nq_double \
                    + np.einsum('abcde,abhde, eahc-> b', (integrand[self.qpdz_plane] - integrand[self.qx_plane]).conj(), integrand, dotvz)/self.h2p.nq_double
        flux_x = (integrand_z+integrand_y)/2 # divide by 2 becuase I sum z and y
        flux_y = (integrand_z+integrand_y)/2 
        flux_z = (integrand_x+integrand_y)/2 
        return {'x=0': flux_x, 'y=0': flux_y, 'z=0': flux_z} 

    def curvature_plaquette(self):

        # get overlaps valence w_{vv'k-q}
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        plaquettes = {
        'x' : np.zeros((NY)*(NZ),dtype=np.complex128),
        'y' : np.zeros((NZ)*(NX),dtype=np.complex128),
        'z' : np.zeros((NX)*(NY),dtype=np.complex128),
        }    

        plaquettes['x'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qxdy_plane].conj(),self.h2p.h2peigvec[self.qx_plane])  \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qxdydz_plane].conj(),self.h2p.h2peigvec[self.qxdy_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qxdz_plane].conj(),self.h2p.h2peigvec[self.qxdydz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qx_plane].conj(),self.h2p.h2peigvec[self.qxdz_plane])  
        plaquettes['y'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qydz_plane].conj(),self.h2p.h2peigvec[self.qy_plane])  \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qydxdz_plane].conj(),self.h2p.h2peigvec[self.qydz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qydx_plane].conj(),self.h2p.h2peigvec[self.qydxdz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qy_plane].conj(),self.h2p.h2peigvec[self.qydx_plane])   
        plaquettes['z'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qzdx_plane].conj(),self.h2p.h2peigvec[self.qz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qzdxdy_plane].conj(),self.h2p.h2peigvec[self.qzdx_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qzdy_plane].conj(),self.h2p.h2peigvec[self.qzdxdy_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qz_plane].conj(),self.h2p.h2peigvec[self.qzdy_plane])                                      
        return plaquettes