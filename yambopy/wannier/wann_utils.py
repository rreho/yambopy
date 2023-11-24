
import numpy as np

HA2EV  = 27.211396132
BOHR2ANG = 0.52917720859
ANG2BOHR = 1./BOHR2ANG

def fermi_dirac_T(e, T, fermie):
    # atomic units
    kb = 3.1671e-06
    fermi = 1.0/(np.exp(e-fermie)/(kb*T))
    return fermi

def fermi_dirac(e, fermie):
    """ Vectorized Fermi-Dirac function. """
    # Create a boolean mask for conditions
    greater_than_fermie = e > fermie
    less_or_equal_minus_fermie = e <= -fermie

    # Initialize the result array with zeros (default case when e > fermie)
    result = np.zeros_like(e)

    # Apply conditions
    result[less_or_equal_minus_fermie] = 1

    return result

def sort_eig(eigv,eigvec=None):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eigv=np.array(eigv.real,dtype=float)
    # sort energies
    args=eigv.argsort()
    eigv=eigv[args]
    if not (eigvec is None):
        eigvec=eigvec[args]
        return (eigv,eigvec)
    return eigv

def find_kpoint_index(klist, kpoint):
    """
    Find the index of a kpoint in a list of kpoints.
    
    Parameters:
    - klist: A list or a NumPy array of kpoints.
    - kpoint: A single kpoint to find in the list.
    
    Returns:
    - Index of the kpoint in the list, or a message if the kpoint is not found.
    """
    # Convert klist to a NumPy array for efficient comparison
    klist_np = np.array(klist)
    kpoint_np = np.array(kpoint)

    # Find the index where kpoint matches in klist
    # We use np.all and np.where to compare each kpoint
    indices = np.where(np.all(klist_np == kpoint_np, axis=1))[0]

    if indices.size > 0:
        return indices[0]
    else:
        print('k-point not found')
        return None

class ChangeBasis():
    '''
    Handles change of basis from Wannier to Bloch and viceversa.
    Np = Nk (number of supercell = number of k-points)
    '''
    def __init__(self, model):
        self.irpos = model.irpos
        self.nrpos = model.nrpos
        self.nb = model.nb
        self.Uknm = model.Uknm
        self.eigvec = model.eigvec
        self.nk = model.nk
        self.kmpgrid = model.mpgrid
        self.k = self.kmpgrid.k
        self.car_kpoints = self.kmpgrid.car_kpoints

    #works don't know why. Probably becuase I am storing each component of the term withn the sum
    def _bloch_to_wann_factor(self, ik,  m, ire):
        k_dot_r = np.dot(self.mpgrid.k, self.irpos.T)
        phase_Uknm_nk = np.zeros((self.nb, self.nb, self.nk),dtype=np.complex128)
        for n in range(0,self.nb):
            phase_Uknm_nk[:,n,ik] =  np.exp(-1j*2*np.pi*k_dot_r[ik, ire])*self.Uknm[ik,n,m]*self.eigvec[ik,:,n]#
        return phase_Uknm_nk

    def bloch_to_wann(self):
        mRe = np.zeros((self.nb, self.nb, self.nrpos),dtype=np.complex128)
        for m in range(0,self.nb):
            for ip in range(0,self.nrpos):
                for n in range(0,self.nb):
                    for ik,k in enumerate(self.mpgrid.k):
                        mRe[:,m,ip] += self._bloch_to_wann_factor(self, ik, m, ip)[:,n,ik]
        return mRe

    def _wann_to_bloch_factor(self,mRe, ik,  n, ire):
        k_dot_r = np.dot(self.mpgrid.k, self.irpos.T)
        phase_Uknm_mRe = np.zeros((self.nb, self.nb, self.nrpos),dtype=np.complex128)
        for m in range(0,self.nb):
            phase_Uknm_mRe[:,m,ire] =  np.exp(1j*2*np.pi*k_dot_r[ik, ire]) *self.Uknm[ik,m,n]*mRe[:,m,ire]#
        return phase_Uknm_mRe

    def wann_to_bloch(self, mRe):
        nk_vec = np.zeros((self.nb, self.nb, self.nk),dtype=np.complex128)
        for ik, k in enumerate(self.mpgrid.k):
            for n in range(0,self.nb):
                for m in range(0, self.nb):
                    for p in range(0,self.nrpos):
                        nk_vec[:,n,ik] += 1/self.nk*self._wann_to_bloch_factor(self, mRe, ik, n, p)[:,m,p]
        nk_vec = nk_vec.transpose(1,0,2)
        return nk_vec