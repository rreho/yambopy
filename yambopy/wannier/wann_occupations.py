import numpy as np
from wannier.wann_utils import fermi_dirac_T, fermi_dirac

class TB_occupations():
    '''Class that handles occupations. The eigenvalues are assumed to be sorted'''
    def __init__(self, eigv, Tel, Tbos, elphg, fermie=0.0, method='FD'):
        self.eigv = eigv
        self.method = method
        self.fermie = fermie
        self.Tel = Tel
        self.Tbos = Tbos
        self.f_kn = self._get_fkn(method)
        #self.elphg = self.elphg
    @classmethod
    def _get_fkn(cls):
        nk = cls.eigv.shape[0]
        nb = cls.eigv.shape[1]
        f_kn = np.zeros((nk,nb), dtype=np.float128)
        if (cls.method =='FD'):
            if (cls.Tel==0):
                f_kn = fermi_dirac(cls.eigv, cls.fermi)
            else:
                f_kn = fermi_dirac_T(cls.eigv, cls.T, cls.fermie)
        if (cls.method == 'BE'):
            pass

