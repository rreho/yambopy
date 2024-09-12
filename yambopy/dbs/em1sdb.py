# Copyright (c) 2018, Henrique Miranda
# All rights reserved.
#
# This file is part of the yambopy project
#
from yambopy import *
from netCDF4 import Dataset
from yambopy.lattice import rec_lat, car_red, red_car

class YamboStaticScreeningDB(object):
    """
    Class to handle static screening databases from Yambo
    
    This reads the databases ``ndb.em1s*``
    There :math:`√v(q,g1) \chi_{g1,g2} (q,\omega=0) √v(q,g2)` is stored.
    
    To calculate epsilon (static dielectric function) we do:

    .. math::

        \epsilon^{-1}_{g1,g2}(q) = 1-v(q,g1)\chi_{g1,g2}
        
    """
    def __init__(self,save='.',em1s='.',filename='ndb.em1s',db1='ns.db1',do_not_read_cutoff=False):
        self.save = save
        self.em1s = em1s
        self.filename = filename
        self.no_cutoff = do_not_read_cutoff

        #read lattice parameters
        if os.path.isfile('%s/%s'%(self.save,db1)):
            try:
                database = Dataset("%s/%s"%(self.save,db1), 'r')
                self.alat = database.variables['LATTICE_PARAMETER'][:]
                self.lat  = database.variables['LATTICE_VECTORS'][:].T
                self.sym_car =  database.variables['SYMMETRY'][:]

                # gvectors_full = database.variables['G-VECTORS'][:].T
                # self.gvectors_full = np.array([ g/self.alat for g in gvectors_full ])
                self.volume = np.linalg.det(self.lat)
                self.rlat = rec_lat(self.lat)
            except:
                raise IOError("Error opening %s."%db1)
        else:
            raise FileNotFoundError("File %s not found."%db1)

        #read em1s database
        if os.path.isfile("%s/%s"%(self.em1s,self.filename)): 
            try:
                database = Dataset("%s/%s"%(self.em1s,self.filename), 'r')
            except:
                raise IOError("Error opening %s/%s in YamboStaticScreeningDB"%(self.save,self.filename))
        else:
            raise FileNotFoundError("File %s not found."%self.filename)

        #read some parameters
        size,nbands,eh = database.variables['X_PARS_1'][:3]
        self.size = int(size)
        self.nbands = int(nbands)
        self.eh = eh

        #read gvectors used for em1s
        gvectors          = np.array(database.variables['X_RL_vecs'][:].T)
        self.gvectors = gvectors
        self.gvectors_car = np.array([ g/self.alat for g in self.gvectors ])
        self.ngvectors    = len(self.gvectors)
        
        self.expanded = False

        #read q-points
        self.kpts_iku = database.variables['HEAD_QPT'][:].T
        self.car_kpoints = np.array([ iku/self.alat for iku in self.kpts_iku ])
        self.red_kpoints =  car_red(self.car_kpoints, self.rlat)
        np.array([ iku/self.alat for iku in self.kpts_iku ])
        self.nkpoints = len(self.car_kpoints)

        try:
            database.variables['CUTOFF'][:]
            self.cutoff = str(database.variables['CUTOFF'][:][0],'UTF-8').strip()
        except: IndexError
        
        #read fragments
        read_fragments=True
        for iQ in range(self.nkpoints):
            if not os.path.isfile("%s/%s_fragment_%d"%(self.em1s,self.filename,iQ+1)): read_fragments=False
        if read_fragments: self.readDBs() # get sqrt(v)*X*sqrt(v)

        #get square root of Coulomb potential v(q,G) 
        self.get_Coulomb()

    def readDBs(self):
        """
        Read the yambo databases
        """

        #create database to hold all the X data
        self.X = np.zeros([self.nkpoints,self.size,self.size],dtype=np.complex64)
        for nq in range(self.nkpoints):

            #open database for each k-point
            filename = "%s/%s_fragment_%d"%(self.em1s,self.filename,nq+1)
            try:
                database = Dataset(filename)
            except:
                print("warning: failed to read %s"%filename)


            # static screening means we have only one frequency
            # this try except is because the way this is stored has changed in yambo
            try:
                re, im = database.variables['X_Q_%d'%(nq+1)][0,:]
            except:
                re, im = database.variables['X_Q_%d'%(nq+1)][0,:].T

            self.X[nq] = re + 1j*im
         
            #close database
            database.close()
    def expand_kpts(self):
        """ Take a list of qpoints and symmetry operations and return the full brillouin zone
        with the corresponding index in the irreducible brillouin zone
        """

        #check if the kpoints were already exapnded
        if self.expanded == True: return self.kpoints_full, self.kpoints_indexes, self.symmetry_indexes

        kpoints_indexes  = []
        kpoints_full     = []
        symmetry_indexes = []

        #kpoints in the full brillouin zone organized per index
        kpoints_full_i = {}

        #expand using symmetries
        for nk,k in enumerate(self.car_kpoints):
            for ns,sym in enumerate(self.sym_car):
                new_k = np.dot(sym,k)

                #check if the point is inside the bounds
                k_red = car_red([new_k],self.rlat)[0]
                k_bz = (k_red+atol)%1

                #if the index in not in the dicitonary add a list
                if nk not in kpoints_full_i:
                    kpoints_full_i[nk] = []

                #if the vector is not in the list of this index add it
                if not vec_in_list(k_bz,kpoints_full_i[nk]):
                    kpoints_full_i[nk].append(k_bz)
                    kpoints_full.append(new_k)
                    kpoints_indexes.append(nk)
                    symmetry_indexes.append(ns)

        #calculate the weights of each of the kpoints in the irreducible brillouin zone
        self.full_nkpoints = len(kpoints_full)
        weights = np.zeros([self.nkpoints])
        for nk in kpoints_full_i:
            weights[nk] = float(len(kpoints_full_i[nk]))/self.full_nkpoints

        #set the variables
        self.expanded = True
        self.weights = np.array(weights)
        self.kpoints_full     = np.array(kpoints_full)
        self.kpoints_indexes  = np.array(kpoints_indexes)
        self.symmetry_indexes = np.array(symmetry_indexes)

        print("%d kpoints expanded to %d"%(len(self.car_kpoints),len(kpoints_full)))

        return self.kpoints_full, self.kpoints_indexes, self.symmetry_indexes

    def XtoSupercell(self, db_sc):
        """
        Works for 2D supercell
        """
        from scipy.spatial import cKDTree

        def gmod(v):
            u = red_car(np.array([v]), db_sc.rlat)*2*np.pi 
            return np.sqrt(np.dot(u[0],u[0]))

        def sort_by_gmod(arr):
            # Function to calculate gmod (magnitude) of a vector
            # Sort the array based on gmod
            threshold = 1.7164942
            arr = np.array(arr)
            gmods = np.array([gmod(v) for v in arr])
            mask = gmods <= threshold
            
            filtered_arr = arr[mask]
            filtered_gmods = gmods[mask]
            
            sorted_indices = np.argsort(filtered_gmods)
            sorted_filtered_arr = filtered_arr[sorted_indices]
            
            return sorted_filtered_arr


        def generate_g_vectors():
            g_vectors = []
            
            x_y_values = np.arange(-8,8,step=1)
            z_values = range(-17, 18)  # -9 to 9 inclusive
            
            for x in x_y_values:
                for y in x_y_values:
                    for z in z_values:
                        g_vectors.append([x, y, z])
            
            sorted_gvecs = sort_by_gmod(g_vectors)

            return np.array(sorted_gvecs)

        def create_g_vector_map(self, db_sc):

            g_vectors = generate_g_vectors()
            g_vectors_car = np.round(red_car(g_vectors, db_sc.rlat), 8)
            # g_vectors_car = np.unique(g_vectors_car, axis=0)
            kdtree = cKDTree(g_vectors_car)
            
            return kdtree, g_vectors_car

        def find_closest_g_vector(kdtree, query_vector, tolerance=1e-4):
            query_vector_rounded = np.round(query_vector, decimals=8)
            distance, index = kdtree.query(query_vector_rounded)
            if distance <= tolerance:
                return index
            return None

        def find_g_Q(Qpoint, qpoint, g_vectors):
            g_Q = qpoint - Qpoint
            if np.any(np.all(np.abs(car_red(np.array(g_Q- g_vectors), db_sc.rlat) ) < 0.0001, axis=1)):
                return True

            else: return False

        self.expand_kpts()
        db_sc.expand_kpts()
        kdtree, g_vectors = create_g_vector_map(self, db_sc)
        x_sc = np.zeros((len(db_sc.car_kpoints), len(g_vectors), len(g_vectors)), dtype=np.complex64)
        qpoints_full = db_sc.kpoints_full
        qpoints_folded_indexes = db_sc.kpoints_indexes
        Qpoints_full = self.kpoints_full
        Qpoints_folded_indexes = self.kpoints_indexes 
            
        for Qi, Qpoint in enumerate(Qpoints_full):
            print(f"\nProcessing Qpoint: {Qpoint}")
            for iG1, Gvec1 in enumerate(self.gvectors_car):
                for iG2, Gvec2 in enumerate(self.gvectors_car):
                    for qi, qpoint in enumerate(qpoints_full):
                        # print(f"Qpoint: {Qpoint}, qpoint: {qpoint}")
                        g_Q_flag = find_g_Q(Qpoint, qpoint, g_vectors)
                        if g_Q_flag:
                            g_Q = qpoint - Qpoint
                            g1 = find_closest_g_vector(kdtree, g_Q + Gvec1)
                            g2 = find_closest_g_vector(kdtree, g_Q + Gvec2)
                            if g1 and g2:
                                qpointindex = qpoints_folded_indexes[qi]
                                Qpointindex = Qpoints_folded_indexes[Qi]

                                if x_sc[qpointindex, g1, g2] != 0.0:
                                    if x_sc[qpointindex, g1, g2] !=self.X[Qpointindex, iG1, iG2] :
                                        print("Overwriting: ", x_sc[qpointindex, g1, g2], self.X[Qpointindex, iG1, iG2])
                                        print(qpointindex, g1, g2, Qpointindex, iG1, iG2)
                                        # break
                                x_sc[qpointindex, g1, g2] = self.X[Qpointindex, iG1, iG2]

                    # print(f"Matches found: {len(qg_map)}")

        return x_sc
        
    def UnfoldxDBS(self,db_sc, path):
        """
        Save the database
        """
        import os
        import shutil
        # if os.path.isdir(path): shutil.rmtree(path)
        # os.mkdir(path)

        #copy all the files
        oldpath = self.em1s
        filename = self.filename
        shutil.copyfile("%s/%s"%(oldpath,filename),"%s/%s"%(path,filename))
        for nq in range(self.nkpoints):
            fname = "%s_fragment_%d"%(filename,nq+1)
            shutil.copyfile("%s/%s"%(oldpath,fname),"%s/%s"%(path,fname))
        supercellsize = int(db_sc.alat[0]/self.alat[0])
        ngvectors= len(self.gvectors)*supercellsize
        #edit with the new wfs
        X = self.XtoSupercell(db_sc)
        fname = "%s"%(filename)
        database = Dataset("%s/%s"%(path,fname),'r+') 
        database.variables['X_RL_vecs'] = db_sc.gvectors[:ngvectors]
        database.variables['HEAD_QPT'] = db_sc.kpts_iku


        for nq in range(db_sc.nkpoints):
            fname = "%s_fragment_%d"%(filename,nq+1)
            database = Dataset("%s/%s"%(path,fname),'r+')
            database.variables['X_Q_%d'%(nq+1)] = np.zeros((ngvectors,ngvectors,2))
            database.variables['X_Q_%d'%(nq+1)][:,:,0] = X[nq].real
            database.variables['X_Q_%d'%(nq+1)][:,:,1] = X[nq].imag
            database.close()
    def saveDBS(self,path):
        """
        Save the database
        """
        if os.path.isdir(path): shutil.rmtree(path)
        os.mkdir(path)

        #copy all the files
        oldpath = self.em1s
        filename = self.filename
        shutil.copyfile("%s/%s"%(oldpath,filename),"%s/%s"%(path,filename))
        for nq in range(self.nkpoints):
            fname = "%s_fragment_%d"%(filename,nq+1)
            shutil.copyfile("%s/%s"%(oldpath,fname),"%s/%s"%(path,fname))

        #edit with the new wfs
        X = self.X
        for nq in range(self.nkpoints):
            fname = "%s_fragment_%d"%(filename,nq+1)
            database = Dataset("%s/%s"%(path,fname),'r+')
            database.variables['X_Q_%d'%(nq+1)][0,0,:] = X[nq].real
            database.variables['X_Q_%d'%(nq+1)][0,1,:] = X[nq].imag
            database.close()

    def writeeps(self,filename='em1s.dat',ng1=0,ng2=0,volume=False):
        """
        Write epsilon_{g1=0,g2=0} (q) as a function of |q| on a text file
        volume -> multiply by the volume
        """
        x,y = self._getepsq(volume=volume)
        np.savetxt(filename,np.array([x,y]).T)
    
    def get_g_index(self,g):
        """
        get the index of the gvectors.
        If the gvector is not present return None
        """
        for ng,gvec in enumerate(self.gvectors_car):
            if np.isclose(g,gvec).all():
                return ng
        return None
    
    

    def get_Coulomb(self):
        """
        By Matteo Zanfrognini

        If cutoff is present, look for ndb.cutoff and parse it.
        Otherwise, construct bare 3D potential.

        Returns sqrt_V[Nq,Ng]
        """  
  
        if self.cutoff!='none' and not self.no_cutoff:

            if os.path.isfile('%s/ndb.cutoff'%self.em1s):
                try:
                    database = Dataset("%s/ndb.cutoff"%self.em1s, 'r')
                    q_p_G_RE = np.array(database.variables["CUT_BARE_QPG"][:,:,0].T)
                    q_p_G_IM = np.array(database.variables["CUT_BARE_QPG"][:,:,1].T)
                    q_p_G = q_p_G_RE + 1j*q_p_G_IM
                    self.sqrt_V = np.sqrt(4.0*np.pi)/q_p_G
                    database.close()
                except:
                    raise IOError("Error opening ndb.cutoff.")
            else:
                print("[WARNING] Cutoff %s was used but ndb.cutoff not found in %s. Make sure this is fine for what you want!"%(self.cutoff,self.em1s))

        else:

            sqrt_V = np.zeros([self.nkpoints,self.ngvectors])
            nrm = np.linalg.norm
            for iq in range(self.nkpoints):
                for ig in range(self.ngvectors):
                        Q = 2.*np.pi*self.car_kpoints[iq]
                        G = 2.*np.pi*self.gvectors[ig]
                        QPG = nrm(Q+G)
                        if QPG==0.: QPG=1.e-5
                        sqrt_V[iq,ig] = np.sqrt(4.0*np.pi)/QPG        
            self.sqrt_V = sqrt_V

    def _getepsq(self,volume=False,use_trueX=False): 
        """
        Get epsilon_{0,0} = [1/(1+vX)]_{0,0} as a function of |q|
        vX is a matrix with size equal to the number of local fields components
 
        In the database we find √vX√v(\omega=0) where:
        v -> coulomb interaction (truncated or not)
        X -> electronic response function

        Arguments:
            ng1, ng2  -> Choose local field components
            volume    -> Normalize with the volume of the cell
            use_trueX -> Use desymmetrised vX [testing]
        """
        if not use_trueX: 
            X = self.X
        if use_trueX:  
            _,_ = self.getem1s()
            X = self.trueX

        x = [np.linalg.norm(q) for q in self.car_kpoints]
        y = [np.linalg.inv(np.eye(self.ngvectors)+xq)[0,0] for xq in X ]
      
        #order according to the distance
        x, y = list(zip(*sorted(zip(x, y))))
        y = np.array(y)

        #scale by volume?
        if volume: y *= self.volume 

        return x,y
    
    def _getvq(self,ng1=0):
        """
        Get Coulomb potential v_ng1 as a function of |q|

        v -> coulomb interaction (truncated or not)

        The quantity obtained is : v(q,g1)

        Arguments:
            ng1 -> Choose local field component
        """
        x = [np.linalg.norm(q) for q in self.car_kpoints]
        y = [vq[ng1]**2. for vq in self.sqrt_V]

        #order according to the distance
        x, y = list(zip(*sorted(zip(x, y))))
        y = np.array(y)

        return x,y

    def _getvxq(self,ng1=0,ng2=0,volume=False): 
        """
        Get vX_{ng1,ng2} as a function of |q|
        vX is a matrix with size equal to the number of local fields components
 
        In the database we find √vX√v(\omega=0) where:
        v -> coulomb interaction (truncated or not)
        X -> electronic response function

        The quantity obtained is: √v(q,g1) X_{g1,g2}(q) √v(q,g2)

        Arguments:
            ng1, ng2 -> Choose local field components
            volume   -> Normalize with the volume of the cell
        """
        x = [np.linalg.norm(q) for q in self.car_kpoints]
        y = [xq[ng2,ng1] for xq in self.X ]
      
        #order according to the distance
        x, y = list(zip(*sorted(zip(x, y))))
        y = np.array(y)

        #scale by volume?
        if volume: y *= self.volume 

        return x,y
 
    def _getem1s(self,ng1=0,ng2=0,volume=False):
        """
        Get eps^-1_{ng1,ng2} a function of |q|

        In the database we find √vX√v(\omega=0) where:
        v -> coulomb interaction (truncated or not)
        X -> electronic response function

        We need to explicitly use √v in order to obtain:

              eps^-1+{g1,g2} = 1+v(q,g1) X_{g1,g2}(q)
                             = 1 + √v_g1 √v_g1 X_g1g2 √v_g2/√v_g2

        This works for 
            - 3D bare √v
            - 2D cutoff √v positive definite (i.e., like slab z)

        Arguments:
            ng1, ng2 -> Choose local field components
            volume   -> Normalize with the volume of the cell
            symm     -> True:  √v(q,g1) X_{g1,g2}(q) √v(q,g2)
                        False: v(q,g1) X_{g1,g2}(q) TO BE IMPLEMENTED
        """
        trueX = np.zeros([self.nkpoints,self.size,self.size],dtype=np.complex64)

        for ig1 in range(self.ngvectors):
            for ig2 in range(self.ngvectors):
                trueX[:,ig1,ig2] = self.sqrt_V[:,ig1]*self.X[:,ig1,ig2]/self.sqrt_V[:,ig2]

        self.trueX = trueX # Store trueX as attribute

        x = [np.linalg.norm(q) for q in self.car_kpoints]
        y = [xq[ng2,ng1] for xq in self.trueX ]

        #order according to the distance
        x, y = list(zip(*sorted(zip(x, y))))
        y = np.array(y)

        #scale by volume?
        if volume: y *= self.volume

        return x,y
   
    def plot_epsm1(self,ax,ng1=0,ng2=0,volume=False,symm=False,**kwargs):
        """
        Plot epsilon^-1_{ng1,ng2} as a function of |q|
        
        Arguments
        ax   -> Instance of the matplotlib axes or some other object with the plot method
        symm -> True:  plot symmetrized version 1 + √vX√v
        symm -> False: plot true em1s as 1+vX [Default]
        """

        #get √vX√v_{ng1,ng2}
        if symm==True:  x,vX = self._getvxq(ng1=ng1,ng2=ng2,volume=volume)
        #get vX_{ng1,ng2}
        if symm==False: x,vX = self._getem1s(ng1=ng1,ng2=ng2,volume=volume)   

        ax.plot(x,(1+vX).real,**kwargs)
        ax.set_xlabel('$|q|$')
        ax.set_ylabel('$\epsilon^{-1}_{%d%d}(\omega=0)$'%(ng1,ng2))

     
    def plot_eps(self,ax,ng1=0,ng2=0,volume=False,use_trueX=False,**kwargs):
        """
        Get epsilon_{0,0} = [1/(1+vX)]_{0,0} as a function of |q|
        """
        x,y = self._getepsq(volume=volume)
        ax.plot(x,y.real,**kwargs)
        ax.set_xlabel('$|q|$')
        ax.set_ylabel('$\epsilon_{%d%d}(\omega=0)$'%(ng1,ng2))

    def plot_v(self,ax,ng1=0,**kwargs):
        """
        Get v_{ng1} (truncated or not) as a function of |q|
        """
        x,y = self._getvq(ng1=ng1)
        ax.plot(x,y.real,**kwargs)
        ax.set_xlabel('$|q|$')
        ax.set_ylabel('$v_{%d}$'%ng1)

    def __str__(self):

        lines = []; app=lines.append
        app(marquee(self.__class__.__name__))

        app('nkpoints (ibz):   %d'%self.nkpoints)
        app('X size (G-space): %d'%self.size) 
        app('cutoff:           %s'%self.cutoff) 

        return "\n".join(lines)


if __name__ == "__main__":

    ys = YamboStaticScreeningDB()
    print(ys)
  
    #plot static screening 
    ax = plt.gca()
    ys.plot_epsm1(ax)
    plt.show()
