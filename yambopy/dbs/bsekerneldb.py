# Author: Davide Romanin (revised: FP, RR)
#
# This file is part of the yambopy project
#
import os
from netCDF4 import Dataset
import numpy as np
from itertools import product
from yambopy import YamboLatticeDB
from yambopy.tools.string import marquee
from yambopy.units import I
from yambopy.lattice import car_red
from yambopy.wannier.wann_bse_wannier import BSEWannierTransformer

class YamboBSEKernelDB(object):
    """ Read the BSE Kernel database from yambo.
        It reads <t1| K |t2> where K is the kernel and t_i transition indices.
        
        Can only be used if yambo is run with parallel IO.
        
        Only supports "RESONANT" case for BSE calculation.
        TODO: support more cases
    """
    def __init__(self,lattice,kernel):
        if not isinstance(lattice,YamboLatticeDB):
            raise ValueError('Invalid type for lattice argument. It must be YamboLatticeDB')
        
        self.lattice  = lattice
        self.kernel   = kernel

    @classmethod
    def from_db_file(cls, lattice, Qpt=1, folder='.'):
        """Initialize this class from a ndb.BS_PAR_Q# file."""
        filename = 'ndb.BS_PAR_Q%d' % Qpt
        path_filename = os.path.join(folder, filename)
        if not os.path.isfile(path_filename):
            raise FileNotFoundError(f"File {path_filename} not found in YamboExcitonDB")

        with Dataset(path_filename) as database:
            if 'BSE_RESONANT' in database.variables:
                # Read as transposed since dimensions in netCDF are inverted
                reker, imker = database.variables['BSE_RESONANT'][:].T
                ker = reker + imker*I
                
                # Transform the triangular matrix to a square Hermitian matrix
                kernel = np.conjugate(np.transpose(np.triu(ker))) + np.triu(ker)
                kernel[np.diag_indices(len(kernel))] *= 0.5
                
                # Check if the kernel is Hermitian
                if not np.allclose(kernel, np.conjugate(kernel.T)):
                    raise ValueError("The constructed kernel matrix is not Hermitian")
            else:
                raise ValueError('Only BSE_RESONANT case supported so far')

        return cls(lattice, kernel)

    @property
    def ntransitions(self): return len(self.kernel)

    def consistency_BSE_BSK(self,excitons):
        """ Check that exciton and kernel dbs are consistent
        """
        if excitons.nexcitons != self.ntransitions:
            print('[WARNING] Exciton and transition spaces have different dimensions!')
        if excitons.ntransitions != self.ntransitions:
            print('[WARNING] Mismatch in ntransitions between ExcitonDB and BSEkernelDB!')        

    def get_kernel_exciton_basis(self,excitons):
        """ Switch from transition |tq>=|kc,k-qv> to excitonic |lq> basis. 
            In this basis the kernel is diagonal.
            
            <l|K|l> = sum_{t,t'}( <l|t><t|K|t'><t'|l> )
                    = sum_{t,t'}( (A^l_t)^* K_tt' A^l_t' ) 
       
            exciton: YamboExcitonDB object 
            Here t->kcv according to table from YamboExcitonDB database
        """
        kernel   = self.kernel
        Nstates  = self.ntransitions
        eivs     = excitons.eigenvectors
        self.consistency_BSE_BSK(excitons)

        # Basis transformation
        kernel_exc_basis  = np.einsum('ij,kj,ki->k', kernel, eivs, np.conj(eivs), optimize=True)
        #kernel_exc_basis = np.zeros(Nstates,dtype=complex)
        #for il in range(Nstates):
        #    kernel_exc_basis[il] = np.dot( np.conj(eivs[il]), np.dot(kernel,eivs[il]) )

        return kernel_exc_basis

    def get_kernel_value_bands(self,excitons,bands):
        """ Get value of kernel matrix elements 
            as a function of k in BZ for fixed c,v bands:
            
                K_cv(k,p) = <ck,vk-q|K|cp,vp-q>
                
            exciton: YamboExcitonDB object
            bands = [iv,ic] (NB: enumerated starting from one instead of zero) 
        """
        table  = excitons.table
        nk     = self.lattice.nkpoints
        kernel = self.kernel
        self.consistency_BSE_BSK(excitons)
        
        if bands[0] not in table[:,1] or bands[1] not in table[:,2]:
            raise ValueError('Band indices not matching available transitions')
             
        # Wcv defined on the full BZ (only a subset will be filled)
        Wcv = np.zeros((nk,nk),dtype=complex)
        # Find indices where selected valence band appears
        t_v = np.where(table[:,1]==bands[0])[0]
        # Find indices where selected conduction band appears
        t_c = np.where(table[:,2]==bands[1])[0]
        # Among those, find subset of indices where both appear together
        t_vc = [t for t in t_v if t in t_c ]

        # Iterate only on the subset
        for it1_subset, it2_subset in product(t_vc,repeat=2):
            ik = table[it1_subset][0]
            ip = table[it2_subset][0]
            Wcv[ik-1,ip-1] = kernel[it1_subset,it2_subset]
        return Wcv

    def get_kernel_value_bands_4D(self, excitons, bands_range):
        """
        Get value of kernel matrix elements as a function of (v, c, k, p)
        where v and c are valence and conduction bands, k and p are k-point indices.

        Args:
        excitons: YamboExcitonDB object containing exciton data and the lattice structure
        bands_range: Tuple (min_band, max_band) defining the range of band indices

        Returns:
        W: A 4D numpy array with dimensions corresponding to (v, c, k, p)
        """
        table = excitons.table
        nk = self.lattice.nkpoints
        kernel = self.kernel
        self.consistency_BSE_BSK(excitons)

        min_band, max_band = bands_range
        nbands = max_band

        # Initialize the 4D array for kernel values with dimensions covering the band range
        W = np.zeros((nbands, nbands, nk, nk), dtype=complex)

        # Iterate over all possible v, c bands within the specified range
        for iv_index, iv in enumerate(range(min_band, max_band + 1), start=0):
            for ic_index, ic in enumerate(range(min_band, max_band + 1), start=0):
                if iv not in table[:, 1] or ic not in table[:, 2]:
                    continue

                t_v = np.where(table[:, 1] == iv)[0]
                t_c = np.where(table[:, 2] == ic)[0]
                t_vc = [t for t in t_v if t in t_c]

                for it1_subset, it2_subset in product(t_vc, repeat=2):
                    ik = table[it1_subset][0]
                    ip = table[it2_subset][0]
                    # Store in 4D array, adjusting indices for zero-based Python indexing
                    W[iv-1, ic-1, ik - 1, ip - 1] = kernel[it1_subset, it2_subset]

        return W
    
    def get_string(self,mark="="):
        lines = []; app = lines.append
        app( marquee(self.__class__.__name__,mark=mark) )
        app( "kernel mode: RESONANT" )
        app( "number of transitions: %d"%self.ntransitions )
        return '\n'.join(lines)
    
    def get_kernel_indices_bands(self,excitons, bands, iq):
        ''' Given a pair of v and c indices get the t indices of the kernel
        '''
        table  = excitons.table
        nk     = self.lattice.nkpoints
        kernel = self.kernel
        self.consistency_BSE_BSK(excitons)
        
        if bands[0] not in table[:,1] or bands[1] not in table[:,2]:
            raise ValueError('Band indices not matching available transitions')
             
        # Find indices where selected valence band appears
        t_v = np.where(table[:,1]==bands[0])[0]
        # Find indices where selected conduction band appears
        t_c = np.where(table[:,2]==bands[1])[0]
        t_k = np.where(table[:,0]==iq)[0]
        # Among those, find subset of indices where both appear together
        t_vc = [t for t in t_v if (t in t_c and t in t_k) ]

        return t_vc[0]

    # ------------------------------------------------------------------
    # Export kernel to band basis tensor with explicit (k,k',v,c,v',c')
    # ------------------------------------------------------------------
    def as_band_kernel_6d(self, excitons):
        '''
        Build K(k,k',v,c,v',c') matching the convention |ck, v(k−q)>. Uses excitons.table.
        Returns (K6d, val_bands, cond_bands) where val_bands/cond_bands are sorted unique indices.
        '''
        table = excitons.table
        nk = self.lattice.nkpoints
        ker = self.kernel
        self.consistency_BSE_BSK(excitons)
        # Unique band sets (Yambo bands are 1-based)
        val_bands = np.array(sorted(np.unique(table[:,1]).tolist()), dtype=int)
        cond_bands = np.array(sorted(np.unique(table[:,2]).tolist()), dtype=int)
        Nv = len(val_bands); Nc = len(cond_bands)
        vmap = {b:i for i,b in enumerate(val_bands)}
        cmap = {b:i for i,b in enumerate(cond_bands)}
        K6 = np.zeros((nk, nk, Nv, Nc, Nv, Nc), dtype=np.complex128)
        Nt = ker.shape[0]
        # Fill via double loop over transitions
        for t1 in range(Nt):
            k1 = int(table[t1,0]) - 1
            v1 = vmap[int(table[t1,1])]
            c1 = cmap[int(table[t1,2])]
            for t2 in range(Nt):
                k2 = int(table[t2,0]) - 1
                v2 = vmap[int(table[t2,1])]
                c2 = cmap[int(table[t2,2])]
                K6[k1, k2, v1, c1, v2, c2] = ker[t1, t2]
        return K6, val_bands, cond_bands

    # ------------------------------------------------------------------
    # Build Wannier transformer from TB model and this kernel
    # ------------------------------------------------------------------
    def to_wannier_transformer(self, tbmodel, qmpgrid, iq, excitons, *, delta_R_h_list=None, delta_R_e_list=None, prune_tol=0.0, norm_factor=None):
        '''
        Construct BSEWannierTransformer using TBMODEL U matrices aligned to v(k−q), c(k),
        and the band kernel assembled from this database.
        '''
        # Kernel in (Nk,Nk,Nv,Nc,Nv,Nc)
        K6, val_bands, cond_bands = self.as_band_kernel_6d(excitons)
        # k-points (reduced) from Yambo lattice (reference ordering for kernel indices)
        kpts_red = np.asarray(self.lattice.red_kpoints, dtype=float)
        Nk = int(kpts_red.shape[0])
        # q (reduced) from provided q-mesh
        qvec = np.asarray(qmpgrid.k[iq], dtype=float)

        # U matrices aligned: val at k−q, cond at k; plus k−q mapping (in TB grid order)
        k_minus_q, U_val, U_cond = tbmodel.get_U_for_q(qmpgrid, iq)

        # Sanity checks on shapes
        assert U_val.shape[0] == tbmodel.nk, "TB U_val_aligned first axis must be nk of TB grid"
        assert U_cond.shape[0] == tbmodel.nk, "TB U_cond first axis must be nk of TB grid"
        assert K6.shape[0] == Nk and K6.shape[1] == Nk, "Kernel Nk must match lattice Nk"

        # # Build permutation between Yambo lattice k-order and TB mpgrid k-order
        # def _permute_from_to(src, dst, decimals=8):
        #     # Return P such that dst[P[i]] == src[i] (mod 1) after rounding
        #     src_mod = np.mod(np.asarray(src, float), 1.0)
        #     dst_mod = np.mod(np.asarray(dst, float), 1.0)
        #     key = lambda x: tuple(np.round(x, decimals=decimals))
        #     lut = {key(dst_mod[j]): j for j in range(dst_mod.shape[0])}
        #     P = np.empty(src_mod.shape[0], dtype=int)
        #     for i in range(src_mod.shape[0]):
        #         k = key(src_mod[i])
        #         if k not in lut:
        #             raise ValueError(
        #                 "Yambo↔TB k-grid mismatch: could not find a TB k matching lattice k[{}] under rounding.".format(i)
        #             )
        #         P[i] = lut[k]
        #     return P

        # try:
        #     P_y2tb = _permute_from_to(kpts_red, tbmodel.mpgrid.k, decimals=8)
        # except Exception as e:
        #     # Provide brief diagnostics on the worst offending point
        #     k_mod = np.mod(kpts_red, 1.0)
        #     tb_mod = np.mod(tbmodel.mpgrid.k, 1.0)
        #     # fallback nearest (not used for mapping, only for message)
        #     from numpy.linalg import norm
        #     diffs = np.min([norm(k_mod[:, None, :] - (tb_mod[None, :, :] + shift), axis=2)
        #                     for shift in (np.array([0, 0, 0])[None, None, :],)], axis=0)
        #     worst = int(np.argmax(np.min(diffs, axis=1)))
        #     raise ValueError(f"k-grid mismatch between lattice and TB grids. Example k[{worst}]={k_mod[worst]} not found in TB grid. Original error: {e}")

        # # Inverse permutation TB->Y
        # invP = np.empty_like(P_y2tb)
        # invP[P_y2tb] = np.arange(Nk)

        # # Reorder TB-provided data into Yambo lattice order expected by K6
        # U_val_aligned = U_val_aligned_tb[P_y2tb]
        # U_cond = U_cond_tb[P_y2tb]
        # # Remap k−q indices from TB order to Yambo order: for Y-index i, map TB j=P[i] -> TB j_mq -> Y invP[j_mq]
        # k_minus_q = invP[k_minus_q_tb[P_y2tb]]

        # # Diagnostics removed for simplicity as per user preference.

        # # Build k+q and (k,k')->q tables using TB mpgrid and provided qmpgrid, then map to Y-order
        # # kpq_table_tb: shape (nk, nq, 5) with indices [ik_tb, iq, 1] = idx of k+q in TB order
        kpq_grid_tb, kpq_table_tb = tbmodel.mpgrid.get_kpq_grid(qmpgrid)
        k_plus_q = kpq_table_tb[:,iq,1]
        if kpq_table_tb.shape[0] != tbmodel.nk or kpq_table_tb.shape[1] != qmpgrid.nkpoints:
            raise ValueError("kpq table shape mismatch with TB or q grids")
        if qmpgrid.nkpoints != Nk:
            # We assume q-grid equals k-grid size for our partial FT; enforce here
            raise ValueError("q-grid size must equal k-grid size (Nk)")
        # Map kpq table to Y-order: k_plus_q_y[i, iq] = ikp_y
        #k_plus_q_y = np.empty((Nk, Nk), dtype=int)
        # for i_y in range(Nk):
        #     i_tb = P_y2tb[i_y]
        #     for iq in range(Nk):
        #         ikp_tb = int(kpq_table_tb[i_tb, iq, 1])
        #         k_plus_q_y[i_y, iq] = int(invP[ikp_tb])
        # Invert to get pair_to_iq_y: for each (i, iq) -> ikp; set (i, ikp) -> iq
        # pair_to_iq_y = np.empty((Nk, Nk), dtype=int)
        # for i in range(Nk):
        #     for iq in range(Nk):
        #         ikp = k_plus_q_y[i, iq]
        #         pair_to_iq_y[i, ikp] = iq

        # Instantiate transformer with data in lattice order
        tr = BSEWannierTransformer(
            kpoints=kpts_red,
            qvec=qvec,
            U_val=U_val,
            U_cond=U_cond,
            K_band=K6,
            k_plus_q_indices=k_plus_q,
            k_minus_q_indices=k_minus_q,
            #pair_to_iq=pair_to_iq_y,
            k_plus_q=kpq_grid_tb,
            delta_R_h_list=delta_R_h_list,
            delta_R_e_list=delta_R_e_list,
            norm_factor_bra=norm_factor,
        )
        return tr

    def __str__(self):
        return self.get_string()
