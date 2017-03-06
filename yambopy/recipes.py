# Copyright (C) 2015 Henrique Pereira Coutada Miranda, Alejandro Molina Sanchez, Alexandre Morlet, Fulvio Paleari
# All rights reserved.
#
# This file is part of yambopy
#
#
from yambopy import *
import os

#
# by Henrique Miranda
#
def pack_files_in_folder(folder,save_folder=None,mask='',verbose=True):
    """
    Pack the output files in a folder to json files
    """
    if not save_folder: save_folder = folder
    #pack the files in .json files
    for dirpath,dirnames,filenames in os.walk(folder):
        #check if the folder fits the mask
        if mask in dirpath:
            #check if there are some output files in the folder
            if ([ f for f in filenames if 'o-' in f ]):
                print dirpath
                y = YamboOut(dirpath,save_folder=save_folder)
                y.pack()

#
# by Alejandro Molina-Sanchez
#
def breaking_symmetries(efield1,efield2=[0,0,0],folder='.',RmTimeRev=True):
    """
    Breaks the symmetries for a given field.
    Second field used in circular polarized pump configuration
    RmTimeRev : Remove time symmetry is set True by default
    """
    os.system('mkdir -p %s'%folder)
    os.system('cp -r database/SAVE %s'%folder)
    os.system('cd %s; yambo'%folder)
    ypp = YamboIn('ypp_ph -y -V all',folder=folder,filename='ypp.in')
    ypp['Efield1'] = efield1 # Field in the X-direction
    ypp['Efield2'] = efield2 # Field in the X-direction
    if RmTimeRev:
        ypp.arguments.append('RmTimeRev')   # Remove Time Symmetry
    ypp.write('%s/ypp.in'%folder)
    os.system('cd %s ; ypp_ph -F ypp.in'%folder )
    os.system('cd %s ; cd FixSymm; yambo '%folder )
    os.system('rm -r %s/SAVE'%folder)
    os.system('mv %s/FixSymm/SAVE %s/'%(folder,folder))
    os.system('rm -r %s/FixSymm'%folder)


#
# by Alexandre Morlet
#

def analyse_gw(folder,var,bandc,kpointc,bandv,kpointv,pack,text,draw):
    """
    Study the convergence of GW calculations by looking at the change in band-gap value.

    The script reads from <folder> all results from <variable> calculations and display them.

    Use the band and k-point options (or change default values) according to the size of your k-grid and
    the location of the band extrema.
    """

    print 'Valence band: ',bandv,'conduction band: ',bandc
    print 'K-point VB: ',kpointv, ' k-point CB: ',kpointc

    # Packing results (o-* files) from the calculations into yambopy-friendly .json files
    if pack:
        print 'Packing ...'
        pack_files_in_folder(folder,mask=var)
        pack_files_in_folder(folder,mask='reference')

    # importing data from .json files in <folder>
    print 'Importing data...'
    data = YamboAnalyser(folder)

    # extract data according to relevant variable
    outvars = data.get_data(var)
    invars = data.get_inputfiles_tag(var)
    tags = data.get_tags(var)

    # Get only files related to the convergence study of the variable,
    # ordered to have a smooth plot
    keys=[]
    sorted_invars = sorted(invars.items(), key=operator.itemgetter(1))

    for i in range(0,len(sorted_invars)):
        key=sorted_invars[i][0]
        if key.startswith(var) or key=='reference.json':
            keys.append(key)
    print 'Files detected: ',keys

    print 'Computing values...'
    ### Output

    # Unit of the variable :
    unit = invars[keys[0]]['variables'][var][1]

    # The following variables are used to make the script compatible with both short and extended output
    kpindex = tags[keys[0]].tolist().index('K-point')
    bdindex = tags[keys[0]].tolist().index('Band')
    e0index = tags[keys[0]].tolist().index('Eo')
    gwindex = tags[keys[0]].tolist().index('E-Eo')


    array = np.zeros((len(keys),2))

    for i,key in enumerate(keys):
        # input value
        # GbndRnge and BndsRnX_ are special cases
        if var.startswith('GbndRng') or var.startswith('BndsRnX'):
            # format : [1, nband, ...]
            array[i][0] = invars[key]['variables'][var][0][1]
        else:
            array[i][0] = invars[key]['variables'][var][0]

        # Output value (gap energy)
        # First the relevant lines are identified
        valence=[]
        conduction=[]
        for j in range(len(outvars[key]+1)):
            if outvars[key][j][kpindex]==kpointc and outvars[key][j][bdindex]==bandc:
                    conduction=outvars[key][j]
            elif outvars[key][j][kpindex]==kpointv and outvars[key][j][bdindex]==bandv:
                    valence = outvars[key][j]
        # Then the gap can be calculated
        array[i][1] = conduction[e0index]+conduction[gwindex]-(valence[e0index]+valence[gwindex])

    if text:
        os.system('mkdir -p analyse_%s'%folder)
        outname = './analyse_%s/%s_%s.dat'%(folder,folder,var)
        header = var+' ('+str(unit)+'), gap'
        np.savetxt(outname,array,delimiter='\t',header=header)
        print 'Data saved to ',outname

    if draw:
        plt.plot(array[:,0],array[:,1],'o-')
        plt.xlabel(var+' ('+unit+')')
        plt.ylabel('E_gw = E_lda + \Delta E')
        plt.show()

#
# by Alexandre Morlet
#
def analyse_bse(folder,var,numbexc,intexc,degenexc,maxexc,pack,text,draw):
    """
    Using ypp, you can study the convergence of BSE calculations in 2 ways:
      Create a .png of all absorption spectra relevant to the variable you study
      Look at the eigenvalues of the first n "bright" excitons (given a threshold intensity)

    The script reads from <folder> all results from <variable> calculations for processing.
    The resulting pictures and data files are saved in the ./analyse_<folder>/ folder.

    Arguments:
        folder   -> Folder containing SAVE and convergence runs.
        var      -> Variable tested (e.g. FFTGvecs)
        numbexc  -> Number of excitons to read beyond threshold (default=2)
        intexc   -> Minimum intensity for excitons to be considered bright (default=0.05)
        degenexc -> Energy threshold under which different peaks are merged (eV) (default=0.01)
        maxexc   -> Energy threshold after which excitons are not read anymore (eV) (default=8.0)
        pack     -> Skips packing o- files into .json files (default: True)
        text     -> Skips writing the .dat file (default: True)
        draw     -> Skips drawing (plotting) the abs spectra (default: True)
    """

    # Packing results (o-* files) from the calculations into yambopy-friendly .json files
    if pack: # True by default, False if -np used
        print 'Packing ...'
        pack_files_in_folder(folder,mask=var)
        pack_files_in_folder(folder,mask='reference')

    # importing data from .json files in <folder>
    print 'Importing data...'
    data = YamboAnalyser(folder)

    # extract data according to relevant var
    invars = data.get_inputfiles_tag(var)

    # Get only files related to the convergence study of the variable,
    # ordered to have a smooth plot
    keys=[]
    sorted_invars = sorted(invars.items(), key=operator.itemgetter(1))

    for i in range(0,len(sorted_invars)):
        key=sorted_invars[i][0]
        if key.startswith(var) or key=='reference.json':
            keys.append(key)
    print 'Files detected: ',keys

    # unit of the input value
    unit = invars[keys[0]]['variables'][var][1]

    ######################
    # Output-file filename
    ######################
    os.system('mkdir -p analyse_%s'%folder)
    outname = './analyse_%s/%s_%s'%(folder,folder,var)

    # Array that will contain the output
    excitons = []

    # Loop over all calculations
    for key in keys:
        jobname=key.replace('.json','')
        print jobname

        # input value
        # BndsRn__ is a special case
        if var.startswith('BndsRnX'):
        # format : [1, nband, ...]
            inp = invars[key]['variables'][var][0][1]
        else:
            inp = invars[key]['variables'][var][0]

        print 'Preparing JSON file. Calling ypp if necessary.'
        ### Creating the 'absorptionspectra.json' file
        # It will contain the exciton energies
        y = YamboOut(folder=folder,save_folder=folder)
        # Args : name of job, SAVE folder path, folder where job was run path
        a = YamboBSEAbsorptionSpectra(jobname,path=folder)
        # Get excitons values (runs ypp once)
        a.get_excitons(min_intensity=intexc,max_energy=maxexc,Degen_Step=degenexc)
        # Write .json file with spectra and eigenenergies
        a.write_json(filename=outname)

        ### Loading data from .json file
        f = open(outname+'.json')
        data = json.load(f)
        f.close()

        ### Plotting the absorption spectra
        # BSE spectra
        plt.plot(data['E/ev[1]'], data['EPS-Im[2]'],label=jobname,lw=2)
    #   # Axes : lines for exciton energies (disabled, would make a mess)
    #   for n,exciton in enumerate(data['excitons']):
    #       plt.axvline(exciton['energy'])

        ### Creating array with exciton values (according to settings)
        l = [inp]
        for n,exciton in enumerate(data['excitons']):
            if n <= numbexc-1:
                l.append(exciton['energy'])

        excitons.append(l)

    if text:
        header = 'Columns : '+var+' (in '+unit+') and "bright" excitons eigenenergies in order.'
        np.savetxt(outname+'.dat',excitons,header=header)
        print 'Data saved to ',outname+'.dat'

    if draw:
        plt.xlabel('$\omega$ (eV)')
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.legend()
        #plt.draw()
        #plt.show()
        plt.savefig(outname+'.png', bbox_inches='tight')
        print outname+'.png'

    print 'Done.'

#
# by Fulvio Paleari & Henrique Miranda
#
def merge_qp(output,files,verbose=False):
    #read all the files and display main info in each of them
    print "=========input========="
    filenames = [ f.name for f in files]
    datasets  = [ Dataset(filename) for filename in filenames]
    QP_table, QP_kpts, QP_E_E0_Z = [], [], []
    for d,filename in zip(datasets,filenames):
        _, nkpoints, nqps, _, nstrings = map(int,d['PARS'][:])
        print "filename:    ", filename
        if verbose:
            print "description:"
            for i in xrange(1,nstrings+1):
                print ''.join(d['DESC_strings_%05d'%i][0])
        else:
            print "description:", ''.join(d['DESC_strings_%05d'%(nstrings)][0])
        print
        QP_table.append( d['QP_table'][:].T )
        QP_kpts.append( d['QP_kpts'][:].T )
        QP_E_E0_Z.append( d['QP_E_Eo_Z'][:] )

    # create the QP_table
    QP_table_save = np.vstack(QP_table)

    # create the kpoints table
    #create a list with the bigger size of QP_table
    nkpoints = int(max(QP_table_save[:,2]))
    QP_kpts_save = np.zeros([nkpoints,3])
    #iterate over the QP's and store the corresponding kpoint
    for qp_file,kpts in zip(QP_table,QP_kpts):
        #iterate over the kpoints and save the coordinates on the list
        for qp in qp_file:
            n1,n2,nk = map(int,qp)
            QP_kpts_save[nk-1] = kpts[nk-1]

    # create the QPs energies table
    QP_E_E0_Z_save = np.concatenate(QP_E_E0_Z,axis=1)

    #create reference file from one of the files
    fin  = datasets[0]
    fout = Dataset(output,'w')

    variables_update = ['QP_table', 'QP_kpts', 'QP_E_Eo_Z']
    variables_save   = [QP_table_save.T, QP_kpts_save.T, QP_E_E0_Z_save]
    variables_dict   = dict(zip(variables_update,variables_save))
    PARS_save = fin['PARS'][:]
    PARS_save[1:3] = nkpoints,len(QP_table_save)

    #create the description string
    kmin,kmax = np.amin(QP_table_save[:,2]),np.amax(QP_table_save[:,2])
    bmin,bmax = np.amin(QP_table_save[:,1]),np.amax(QP_table_save[:,1])
    description = "QP @ K %03d - %03d : b %03d - %03d"%(kmin,kmax,bmin,bmax)
    description_save = np.array([i for i in " %s"%description])

    #output data
    print "========output========="
    print "filename:    ", output
    print "description: ", description

    #copy dimensions
    for dname, the_dim in fin.dimensions.iteritems():
        fout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)

    #get dimensions
    def dimensions(array):
        return tuple([ 'D_%010d'%d for d in array.shape ])

    #create missing dimensions
    for v in variables_save:
        for dname,d in zip( dimensions(v),v.shape ):
            if dname not in fout.dimensions.keys():
                fout.createDimension(dname, d)

    #copy variables
    for v_name, varin in fin.variables.iteritems():
        if v_name in variables_update:
            #get the variable
            merged = variables_dict[v_name]
            # create the variable
            outVar = fout.createVariable(v_name, varin.datatype, dimensions(merged))
            # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            #save outvar
            outVar[:] = merged

        else:
            # create the variable
            outVar = fout.createVariable(v_name, varin.datatype, varin.dimensions)
            # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            if v_name=='PARS':
                outVar[:] = PARS_save[:]
            elif v_name=='DESC_strings_%05d'%(nstrings):
                outVar[:] = varin[:]
                outVar[:,:len(description_save)] = description_save.T
            else:
                outVar[:] = varin[:]

    fout.close()
