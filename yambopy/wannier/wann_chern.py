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
            
            self.borders ={
                'yz': self.h2p.qmpgrid.find_border_kpoints_inplane(1.0,0.0,0.0,0.0),
                'zx': self.h2p.qmpgrid.find_border_kpoints_inplane(0.0,1.0,0.0,0.0),
                'xy': self.h2p.qmpgrid.find_border_kpoints_inplane(0.0,0.0,1.0,0.0),
            }
            self.nyz_border = len(self.borders['yz'])
            self.nzx_border = len(self.borders['zx'])
            self.nxy_border = len(self.borders['xy'])
            
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
            self.nx_plane = len(self.qx_plane)
            self.qpdy_plane = self.qpdqx_grid[self.qy_plane,self.i_y][:,1]
            self.ny_plane = len(self.qx_plane)
            self.qpdz_plane = self.qpdqx_grid[self.qz_plane,self.i_z][:,1]
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
            #extract points on the border
            #'yz'
            self.borderyz_x = self.qpdqx_grid[self.borders['yz'], self.i_x][:,1]
            self.borderyz_y = self.qpdqx_grid[self.borders['yz'], self.i_y][:,1]
            self.borderyz_z = self.qpdqx_grid[self.borders['yz'], self.i_z][:,1]
            #zx
            self.borderzx_x = self.qpdqx_grid[self.borders['zx'], self.i_x][:,1]
            self.borderzx_y = self.qpdqx_grid[self.borders['zx'], self.i_y][:,1]
            self.borderzx_z = self.qpdqx_grid[self.borders['zx'], self.i_z][:,1]
            #xy
            self.borderxy_x = self.qpdqx_grid[self.borders['xy'], self.i_x][:,1]
            self.borderxy_y = self.qpdqx_grid[self.borders['xy'], self.i_y][:,1]
            self.borderxy_z = self.qpdqx_grid[self.borders['xy'], self.i_z][:,1]                                                     

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
        integrand_yz = (integrand[self.borders['yz']].conj()*(integrand[self.borderyz_x] - integrand[self.borders['yz']]))/self.nyz_border + \
                       (integrand[self.borders['yz']].conj()*(integrand[self.borderyz_y] - integrand[self.borders['yz']]))/self.nyz_border + \
                       (integrand[self.borders['yz']].conj()*(integrand[self.borderyz_z] - integrand[self.borders['yz']]))/self.nyz_border 
        integrand_zx = (integrand[self.borders['zx']].conj()*(integrand[self.borderzx_x] - integrand[self.borders['zx']]))/self.nzx_border + \
                       (integrand[self.borders['zx']].conj()*(integrand[self.borderzx_y] - integrand[self.borders['zx']]))/self.nzx_border + \
                       (integrand[self.borders['zx']].conj()*(integrand[self.borderzx_z] - integrand[self.borders['zx']]))/self.nzx_border 
        integrand_xy = (integrand[self.borders['xy']].conj()*(integrand[self.borderxy_x] - integrand[self.borders['xy']]))/self.nxy_border + \
                       (integrand[self.borders['xy']].conj()*(integrand[self.borderxy_y] - integrand[self.borders['xy']]))/self.nxy_border + \
                       (integrand[self.borders['xy']].conj()*(integrand[self.borderxy_z] - integrand[self.borders['xy']]))/self.nxy_border 

        flux_yz = np.sum(integrand_yz, axis=(0,2,3,4))
        flux_zx = np.sum(integrand_zx, axis=(0,2,3,4)) 
        flux_xy = np.sum(integrand_xy, axis=(0,2,3,4))  

        return {'yz': flux_yz, 'zx': flux_zx, 'xy': flux_xy}

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
        eigvec_ckyz = self.h2p.eigvec[self.borders['yz'], :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckyzdqx = self.h2p.eigvec[self.borderyz_x, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckyzdqy = self.h2p.eigvec[self.borderyz_y, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckyzdqz = self.h2p.eigvec[self.borderyz_z, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzx = self.h2p.eigvec[self.borders['zx'], :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckzxdqx = self.h2p.eigvec[self.borderzx_x, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzxdqy = self.h2p.eigvec[self.borderzx_y, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzxdqz = self.h2p.eigvec[self.borderzx_z, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxy = self.h2p.eigvec[self.borders['xy'], :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckxydqx = self.h2p.eigvec[self.borderxy_x, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxydqy = self.h2p.eigvec[self.borderxy_y, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxydqz = self.h2p.eigvec[self.borderxy_z, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        # dot product q point belong on the yz-plane (x=0 or .qx_plane)
        dotcyz_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckyzdqx-eigvec_ckyz), eigvec_ckyz)
        dotcyz_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckyzdqy-eigvec_ckyz), eigvec_ckyz) 
        dotcyz_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckyzdqz-eigvec_ckyz), eigvec_ckyz) 
        # dot product q point belong on the zx-plane (y=0 or .qy_plane)
        dotczx_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzxdqx-eigvec_ckzx), eigvec_ckzx)
        dotczx_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzxdqy-eigvec_ckzx), eigvec_ckzx) 
        dotczx_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzxdqz-eigvec_ckzx), eigvec_ckzx) 
        # dot product q point belong on the xy-plane (z=0 or .qy_plane)
        dotcxy_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxydqx-eigvec_ckxy), eigvec_ckxy)
        dotcxy_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxydqy-eigvec_ckxy), eigvec_ckxy) 
        dotcxy_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxydqz-eigvec_ckxy), eigvec_ckxy) 
        # Evaluate the integrand at the points on each plane
        integrandyz_x = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['yz']], integrand[self.borders['yz']], dotcyz_x)/self.nyz_border
        integrandyz_y = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['yz']], integrand[self.borders['yz']], dotcyz_y)/self.nyz_border
        integrandyz_z = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['yz']], integrand[self.borders['yz']], dotcyz_z)/self.nyz_border
        integrandzx_x = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['zx']], integrand[self.borders['zx']], dotczx_x)/self.nzx_border
        integrandzx_y = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['zx']], integrand[self.borders['zx']], dotczx_y)/self.nzx_border
        integrandzx_z = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['zx']], integrand[self.borders['zx']], dotczx_z)/self.nzx_border
        integrandxy_x = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['xy']], integrand[self.borders['xy']], dotcxy_x)/self.nxy_border
        integrandxy_y = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['xy']], integrand[self.borders['xy']], dotcxy_y)/self.nxy_border
        integrandxy_z = np.einsum('abcde,abcie, adi -> b',integrand.conj()[self.borders['xy']], integrand[self.borders['xy']], dotcxy_z)/self.nxy_border                
        flux_yz = (integrandyz_x+integrandyz_y+integrandyz_z) # divide by 2 becuase I sum z and y (?)
        flux_zx = (integrandzx_x+integrandzx_y+integrandzx_z)
        flux_xy = (integrandxy_x+integrandxy_y+integrandxy_z)
        
        return {'yz': flux_yz, 'zx': flux_zx, 'xy': flux_xy}
    
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
        ikminusqyz = self.h2p.kminusq_table[:, self.borders['yz'], 1]
        ikminusqzx = self.h2p.kminusq_table[:, self.borders['zx'], 1]
        ikminusqxy = self.h2p.kminusq_table[:, self.borders['xy'], 1]
        # now I need to get the k-q-dx/dy/dz component for q being in the yz (qx_plane), zx (qy_plane) or xy (qz_plane) planes
        ikmqyzpdqx = kmqdq_grid_table[0][:,self.borders['yz'],1] #dx
        ikmqyzpdqy = kmqdq_grid_table[1][:,self.borders['yz'],1] #dy
        ikmqyzpdqz = kmqdq_grid_table[2][:,self.borders['yz'],1] #dz

        ikmqzxpdqx = kmqdq_grid_table[0][:,self.borders['zx'],1] #dx
        ikmqzxpdqy = kmqdq_grid_table[1][:,self.borders['zx'],1] #dy
        ikmqzxpdqz = kmqdq_grid_table[2][:,self.borders['zx'],1] #dz

        ikmqxypdqx = kmqdq_grid_table[0][:,self.borders['xy'],1] #dx
        ikmqxypdqy = kmqdq_grid_table[1][:,self.borders['xy'],1] #dy
        ikmqxypdqz = kmqdq_grid_table[2][:,self.borders['xy'],1] #dz

        eigvec_vkmqyz = self.h2p.eigvec[ikminusqyz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqzx = self.h2p.eigvec[ikminusqzx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqxy = self.h2p.eigvec[ikminusqxy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqyzpdqx = self.h2p.eigvec[ikmqyzpdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqyzpdqy = self.h2p.eigvec[ikmqyzpdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqyzpdqz = self.h2p.eigvec[ikmqyzpdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzxpdqx = self.h2p.eigvec[ikmqzxpdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzxpdqy = self.h2p.eigvec[ikmqzxpdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzxpdqz = self.h2p.eigvec[ikmqzxpdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxypdqx = self.h2p.eigvec[ikmqxypdqx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxypdqy = self.h2p.eigvec[ikmqxypdqy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxypdqz = self.h2p.eigvec[ikmqxypdqz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
       
        dotvyz_x = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqyzpdqx-eigvec_vkmqyz), eigvec_vkmqyz)
        dotvyz_y = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqyzpdqy-eigvec_vkmqyz), eigvec_vkmqyz)
        dotvyz_z = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqyzpdqz-eigvec_vkmqyz), eigvec_vkmqyz)             
        
        dotvzx_x = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqzxpdqx-eigvec_vkmqzx), eigvec_vkmqzx) #l index is conjugated       
        dotvzx_y = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqzxpdqy-eigvec_vkmqzx), eigvec_vkmqzx) #l index is conjugated       
        dotvzx_z = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqzxpdqz-eigvec_vkmqzx), eigvec_vkmqzx) #l index is conjugated               
        
        dotvxy_x = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqxypdqx-eigvec_vkmqxy), eigvec_vkmqxy) #l index is conjugated       
        dotvxy_y = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqxypdqy-eigvec_vkmqxy), eigvec_vkmqxy) #l index is conjugated       
        dotvxy_z = np.einsum('ijkl,ijkp->ijlp',np.conjugate(eigvec_vkmqxypdqz-eigvec_vkmqxy), eigvec_vkmqxy) #l index is conjugated       

        integrandyz_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['yz']], integrand[self.borders['yz']], dotvyz_x)/self.nyz_border
        integrandyz_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['yz']], integrand[self.borders['yz']], dotvyz_y)/self.nyz_border
        integrandyz_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['yz']], integrand[self.borders['yz']], dotvyz_z)/self.nyz_border       
        integrandzx_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['zx']], integrand[self.borders['zx']], dotvzx_x)/self.nzx_border
        integrandzx_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['zx']], integrand[self.borders['zx']], dotvzx_y)/self.nzx_border
        integrandzx_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['zx']], integrand[self.borders['zx']], dotvzx_z)/self.nzx_border
        integrandxy_x = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['xy']], integrand[self.borders['xy']], dotvxy_x)/self.nxy_border
        integrandxy_y = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['xy']], integrand[self.borders['xy']], dotvxy_y)/self.nxy_border
        integrandxy_z = np.einsum('abcde,abhde, eahc -> b',integrand.conj()[self.borders['xy']], integrand[self.borders['xy']], dotvxy_z)/self.nxy_border

        flux_yz = integrandyz_x+integrandyz_y+integrandyz_z 
        flux_zx = integrandzx_x+integrandzx_y+integrandzx_z
        flux_xy = integrandxy_x+integrandxy_y+integrandxy_z 

        return {'yz': flux_yz, 'zx': flux_zx, 'xy': flux_xy}         

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
        #yz plane
        eigvec_ckx    = self.h2p.eigvec[self.qx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckxdqx = self.h2p.eigvec[self.qpdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxdqy = self.h2p.eigvec[self.qxdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckxdqz = self.h2p.eigvec[self.qxdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        #zx plane
        eigvec_cky    = self.h2p.eigvec[self.qy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckydqx = self.h2p.eigvec[self.qydx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckydqy = self.h2p.eigvec[self.qpdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckydqz = self.h2p.eigvec[self.qydz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        #xy plane
        eigvec_ckz    = self.h2p.eigvec[self.qz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_ckzdqx = self.h2p.eigvec[self.qzdx_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzdqy = self.h2p.eigvec[self.qzdy_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_ckzdqz = self.h2p.eigvec[self.qpdz_plane, :, :][:,:,np.unique(self.h2p.BSE_table[:,2])]#[np.newaxis,:,:,:]  # Valence bands 
        
        dotcx_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqx-eigvec_ckx), eigvec_ckx) #l index is conjugated  
        dotcx_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqy-eigvec_ckx), eigvec_ckx) #l index is conjugated  
        dotcx_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckxdqz-eigvec_ckx), eigvec_ckx) #l index is conjugated       
        #prepare for cross products
        # wccx 
        wccx = np.array([dotcx_x, dotcx_y, dotcx_z])
        dotcy_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqx-eigvec_cky), eigvec_cky) #l index is conjugated  
        dotcy_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqy-eigvec_cky), eigvec_cky) #l index is conjugated  
        dotcy_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckydqz-eigvec_cky), eigvec_cky) #l index is conjugated      
        # wccy
        wccy = np.array([dotcy_x, dotcy_y, dotcy_z])
        dotcz_x = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqx-eigvec_ckz), eigvec_ckz) #l index is conjugated  
        dotcz_y = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqy-eigvec_ckz), eigvec_ckz) #l index is conjugated  
        dotcz_z = np.einsum('jkl,jkp->jlp',np.conjugate(eigvec_ckzdqz-eigvec_ckz), eigvec_ckz) #l index is conjugated      
        # wccz 
        wccz = np.array([dotcz_x, dotcz_y, dotcz_z])
        
        # Evaluate the integrand at the points on each plane
        integrand_x = np.array([np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qx_plane], integrand[self.qpdx_plane] - integrand[self.qx_plane]),\
                                np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qx_plane], integrand[self.qxdy_plane] - integrand[self.qx_plane]),\
                                np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qx_plane], integrand[self.qxdz_plane] - integrand[self.qx_plane]),    
        ])
        integrand_y = np.array([np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qy_plane], integrand[self.qydx_plane] - integrand[self.qy_plane]),\
                                np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qy_plane], integrand[self.qpdy_plane] - integrand[self.qy_plane]),\
                                np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qy_plane], integrand[self.qydz_plane] - integrand[self.qy_plane]),    
        ])
        integrand_z = np.array([np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qz_plane], integrand[self.qzdx_plane] - integrand[self.qz_plane]),\
                                np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qz_plane], integrand[self.qzdy_plane] - integrand[self.qz_plane]),\
                                np.einsum('abcde,abche -> abcdhe', integrand.conj()[self.qz_plane], integrand[self.qpdz_plane] - integrand[self.qz_plane]),    
        ])
        #cross products in plane
        cross_product_yz = integrand_x[1]*wccx[2][:, None, None,:,:, None]-integrand_x[2]*wccx[1][:, None, None,:,:, None]
        cross_product_zx = integrand_y[1]*wccy[2][:, None, None,:,:, None]-integrand_y[2]*wccy[1][:, None, None,:,:, None]
        cross_product_xy = integrand_z[1]*wccz[2][:, None, None,:,:, None]-integrand_z[2]*wccz[1][:, None, None,:,:, None]

        # sum over dimensions
        flux_yz = np.sum(cross_product_yz, axis=tuple((0,2,3,4,5))) # divide by 2 becuase I sum z and y
        flux_zx = np.sum(cross_product_zx, axis=tuple((0,2,3,4,5)))
        flux_xy = np.sum(cross_product_xy, axis=tuple((0,2,3,4,5))) 
        
        return {'yz': flux_yz, 'zx': flux_zx, 'xy': flux_xy}  
    
    def chern_exc_h(self, integrand):
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
        #k-q planes
        ikminusqyz = self.h2p.kminusq_table[:, self.qx_plane, 1]
        ikminusqzx = self.h2p.kminusq_table[:, self.qy_plane, 1]
        ikminusqxy = self.h2p.kminusq_table[:, self.qz_plane, 1]
        #k-q-dx planes
        ikmqdqyz_x = kmqdq_grid_table[0][:,self.qx_plane,1] #dx
        ikmqdqyz_y = kmqdq_grid_table[1][:,self.qx_plane,1] #dy
        ikmqdqyz_z = kmqdq_grid_table[2][:,self.qx_plane,1] #dz
        #zx
        ikmqdqzx_x = kmqdq_grid_table[0][:,self.qy_plane,1] #dx
        ikmqdqzx_y = kmqdq_grid_table[1][:,self.qy_plane,1] #dy
        ikmqdqzx_z = kmqdq_grid_table[2][:,self.qy_plane,1] #dz
        #xy
        ikmqdqxy_x = kmqdq_grid_table[0][:,self.qz_plane,1] #dx
        ikmqdqxy_y = kmqdq_grid_table[1][:,self.qz_plane,1] #dy
        ikmqdqxy_z = kmqdq_grid_table[2][:,self.qz_plane,1] #dz        
        
        #yz plane
        eigvec_vkmqyz   = self.h2p.eigvec[ikminusqyz, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqyz_x = self.h2p.eigvec[ikmqdqyz_x, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqyz_y = self.h2p.eigvec[ikmqdqyz_y, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqyz_z = self.h2p.eigvec[ikmqdqyz_z, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        #zx plane
        eigvec_vkmqzx   = self.h2p.eigvec[ikminusqzx, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqzx_x = self.h2p.eigvec[ikmqdqzx_x, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzx_y = self.h2p.eigvec[ikmqdqzx_y, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqzx_z = self.h2p.eigvec[ikmqdqzx_z, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands
        #xy plane
        eigvec_vkmqxy   = self.h2p.eigvec[ikminusqxy, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[:,np.newaxis,:,:]  # Valence bands
        eigvec_vkmqxy_x = self.h2p.eigvec[ikmqdqxy_x, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxy_y = self.h2p.eigvec[ikmqdqxy_y, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands 
        eigvec_vkmqxy_z = self.h2p.eigvec[ikmqdqxy_z, :, :][:,:,:,np.unique(self.h2p.BSE_table[:,1])]#[np.newaxis,:,:,:]  # Valence bands

        dotcyz_x = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqyz_x-eigvec_vkmqyz), eigvec_vkmqyz) #l index is conjugated  
        dotcyz_y = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqyz_y-eigvec_vkmqyz), eigvec_vkmqyz) #l index is conjugated  
        dotcyz_z = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqyz_z-eigvec_vkmqyz), eigvec_vkmqyz) #l index is conjugated       
        #prepare for cross products
        # wccyz 
        wccyz = np.array([dotcyz_x, dotcyz_y, dotcyz_z])
        dotczx_x = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqzx_x-eigvec_vkmqzx), eigvec_vkmqzx) #l index is conjugated  
        dotczx_y = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqzx_y-eigvec_vkmqzx), eigvec_vkmqzx) #l index is conjugated  
        dotczx_z = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqzx_z-eigvec_vkmqzx), eigvec_vkmqzx) #l index is conjugated       
        # wcczx
        wcczx = np.array([dotczx_x, dotczx_y, dotczx_z])
        dotcxy_x = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqxy_x-eigvec_vkmqxy), eigvec_vkmqxy) #l index is conjugated  
        dotcxy_y = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqxy_y-eigvec_vkmqxy), eigvec_vkmqxy) #l index is conjugated  
        dotcxy_z = np.einsum('ijkl,ijkp->jlpi',np.conjugate(eigvec_vkmqxy_z-eigvec_vkmqxy), eigvec_vkmqxy) #l index is conjugated       
        # wccxy 
        wccxy = np.array([dotcxy_x, dotcxy_y, dotcxy_z])
        
        # Evaluate the integrand at the points on each plane
        integrand_x = np.array([np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qx_plane], integrand[self.qpdx_plane] - integrand[self.qx_plane]),\
                                np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qx_plane], integrand[self.qxdy_plane] - integrand[self.qx_plane]),\
                                np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qx_plane], integrand[self.qxdz_plane] - integrand[self.qx_plane]),    
        ])
        integrand_y = np.array([np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qy_plane], integrand[self.qydx_plane] - integrand[self.qy_plane]),\
                                np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qy_plane], integrand[self.qpdy_plane] - integrand[self.qy_plane]),\
                                np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qy_plane], integrand[self.qydz_plane] - integrand[self.qy_plane]),    
        ])
        integrand_z = np.array([np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qz_plane], integrand[self.qzdx_plane] - integrand[self.qz_plane]),\
                                np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qz_plane], integrand[self.qzdy_plane] - integrand[self.qz_plane]),\
                                np.einsum('abcde,abhde -> abchde', integrand.conj()[self.qz_plane], integrand[self.qpdz_plane] - integrand[self.qz_plane]),    
        ])
        #cross products in plane
        cross_product_yz = integrand_x[1]*wccyz[2][:, None,:, :, None,:]-integrand_x[2]*wccyz[1][:, None,:, :, None,:]
        cross_product_zx = integrand_y[1]*wcczx[2][:, None,:, :, None,:]-integrand_y[2]*wcczx[1][:, None,:, :, None,:]
        cross_product_xy = integrand_z[1]*wccxy[2][:, None,:, :, None,:]-integrand_z[2]*wccxy[1][:, None,:, :, None,:]

        # sum over dimensions
        flux_yz = np.sum(cross_product_yz, axis=tuple((0,2,3,4,5))) # divide by 2 becuase I sum z and y
        flux_zx = np.sum(cross_product_zx, axis=tuple((0,2,3,4,5)))
        flux_xy = np.sum(cross_product_xy, axis=tuple((0,2,3,4,5))) 
        
        return {'yz': flux_yz, 'zx': flux_zx, 'xy': flux_xy}  

    def curvature_plaquette(self):

        # get overlaps valence w_{vv'k-q}
        NX, NY, NZ = self.h2p.qmpgrid.grid_shape
        plaquettes = {
        'yz' : np.zeros((NY)*(NZ),dtype=np.complex128),
        'zx' : np.zeros((NZ)*(NX),dtype=np.complex128),
        'xy' : np.zeros((NX)*(NY),dtype=np.complex128),
        }    

        plaquettes['yz'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qxdy_plane].conj(),self.h2p.h2peigvec[self.qx_plane])  \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qxdydz_plane].conj(),self.h2p.h2peigvec[self.qxdy_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qxdz_plane].conj(),self.h2p.h2peigvec[self.qxdydz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qx_plane].conj(),self.h2p.h2peigvec[self.qxdz_plane])  
        plaquettes['zx'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qydz_plane].conj(),self.h2p.h2peigvec[self.qy_plane])  \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qydxdz_plane].conj(),self.h2p.h2peigvec[self.qydz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qydx_plane].conj(),self.h2p.h2peigvec[self.qydxdz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qy_plane].conj(),self.h2p.h2peigvec[self.qydx_plane])   
        plaquettes['xy'] = np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qzdx_plane].conj(),self.h2p.h2peigvec[self.qz_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qzdxdy_plane].conj(),self.h2p.h2peigvec[self.qzdx_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qzdy_plane].conj(),self.h2p.h2peigvec[self.qzdxdy_plane]) \
                        * np.einsum('kts, kts -> ks' , self.h2p.h2peigvec[self.qz_plane].conj(),self.h2p.h2peigvec[self.qzdy_plane])                                      
        return plaquettes