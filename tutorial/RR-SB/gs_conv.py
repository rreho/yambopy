#
# Author: Riccardo Reho
# Run a TMD WSe2 convergence calculation using QE
from __future__ import print_function
import os
from pydoc import doc
import sys
from qepy import *
from schedulerpy import *
import argparse
import numpy as np
import sisl
import re

pw='/sw/arch/Centos8/EB_production/2021/software/QuantumESPRESSO/6.7-foss-2021a/bin/pw.x'

geomxsf='PbTe.xsf' 
scf_kpoints = [4,4,4]
nscf_kpoints = [12,12,12]
db_kpoints = [24,24,1]
energies = [60,70,80,90,100,110,120]
vacuum = np.arange(7,21,1)
filename='PbTe'
prefix = 'PbTe'
work_path='/gpfs/work2/0/prjs0229/rireho/PbTe/qe/convergencestudy'
npoints = 30
p = Path([ [[  0.37500,  0.37500,  0.75000],'K'],
              [[  0.5000,  0.5000,  0.5000],'L'],
              [[0.0,0.0,0.0],'$\Gamma$'],
              [[0.5000,0.0,0.5000],'X'],
              [[0.500,0.2500,0.7500],'W'],
              [[0.0,0.0,0.0],'$\Gamma$'],
              [[0.37500,0.37500,0.7500],'K']], [int(npoints)] )
#
# Create input files
#
# scheduler
scheduler = Scheduler.factory

# create the input files
def get_inputfile():
    """ Define a Quantum espresso input file for your system MaterialsCloud coordinates
    """
    if (not args.xsf):
        qe = PwIn()
        qe.system['ibrav'] = 0
        qe.atypes = {'Pb': [207.2, "Pb.upf"],
                    'Te': [127.6,"Te.upf"],}
        qe.control['prefix'] = "'%s'"%prefix
        qe.control['verbosity'] = "'high'"
        qe.control['wf_collect'] = '.true.'
        qe.control['pseudo_dir'] = "'/home/rireho/pseudos/SO'"
        qe.control['restart_mode']="'from_scratch'"
        qe.control['tprnfor']='.true.'
        qe.control['tstress']='.true.'
        qe.control['forc_conv_thr']= '1e-03'
        qe.control['etot_conv_thr']='1e-09'
        qe.control['max_seconds']='82800'
        qe.system['force_symmorphic'] = '.true.'
        qe.system['ecutwfc'] = 60
        qe.system['occupations'] = "'smearing'"
        qe.system['nat'] = newgeom.na
        qe.system['ntyp'] = 2
        qe.system['lspinorb']='.true.'
        qe.system['noncolin']='.true.'
        qe.kpoints = [9, 9, 1]
        qe.electrons['conv_thr'] = 1e-12
        qe.electrons['electron_maxstep'] = 1000
        qe.set_atoms([['Pb',[0.0000000000,        0.0000000000,       0.000000000]],
                    ['Te',[0.5000000000,        0.500000000,        0.500000000    ]]])
    else:
        qe = PwIn()
        qe.system['ibrav'] = 0
        qe.atypes = {'Pb': [207.2, "Pb.upf"],
                    'Te': [127.6,"Te.upf"],}
        qe.control['prefix'] = "'%s'"%prefix
        qe.control['verbosity'] = "'high'"
        qe.control['wf_collect'] = '.true.'
        qe.control['pseudo_dir'] = "'/home/rireho/pseudos/SO'"
        qe.control['restart_mode']="'from_scratch'"
        qe.control['tprnfor']='.true.'
        qe.control['tstress']='.true.'
        qe.control['forc_conv_thr']= '1e-06'
        qe.control['etot_conv_thr']='1e-09'
        qe.control['max_seconds']='82800'
        qe.system['force_symmorphic'] = '.true.'
        qe.system['ecutwfc'] = 60
        qe.system['occupations'] = "'smearing'"
        qe.system['degauss']=1e-03
        qe.system['nat'] = newgeom.na
        qe.system['ntyp'] = 2
        qe.system['lspinorb']='.true.'
        qe.system['noncolin']='.true.'
        qe.kpoints = [9, 9, 1]
        qe.electrons['conv_thr'] = 1e-12
        qe.electrons['electron_maxstep'] = 1000
        qe.set_atoms([[newgeom.atoms[i].symbol,newgeom.fxyz[i]] for i in range(newgeom.na)])
        qe.cell_parameters=newgeom.cell
    return qe

#scf
def scf(kpoints,folder='scf'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe = get_inputfile()
    qe.control['calculation'] = "'scf'"
    qe.kpoints = kpoints
    qe.write('%s/%s.scf'%(folder,filename))
def set_ecutwfc(kpoints,energies,folder='ecutwfc'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe.control['calculation']="'scf'"
    qe.kpoints=kpoints
    for i,en in enumerate(energies):
        qe.system['ecutwfc']=en
        qe.control['prefix'] = "'%s-%s'"%(prefix,en)
        qe.write(f'{folder}/{filename}-{en}.scf')
        replace(f'{folder}/{filename}-{en}.scf','bohr','angstrom')
    return qe
def vcrelax(kpoints,folder='vc-relax'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe = get_inputfile()
    qe.control['calculation'] = "'vc-relax'"
    qe.ions['ion_dynamics']  = "'bfgs'"
    qe.cell['cell_dynamics']  = "'bfgs'"
    qe.kpoints = kpoints
    qe.write('%s/%s.relax'%(folder,filename))
def set_kmesh_relax (folder='relax-kgrid'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe = get_inputfile()
    qe.control['calculation'] = "'vc-relax'"
    qe.ions['ion_dynamics']  = "'bfgs'"
    qe.cell['cell_dynamics']  = "'bfgs'"
    grids=[]
    for i in [4,8,12]:
        grids.append([i,i,i])
    for i,grd in enumerate(grids):
       qe.kpoints=grd
       grd_s=re.sub(', ','-',str(grd).strip('[]'))
       qe.control['prefix'] = "'%s-%s'"%(prefix,grd_s)
       qe.write(f'{folder}/{filename}-{grd_s}.relax')
       #replace(f'{folder}/{filename}-{grd_s}.scf','crystal','angstrom')
       replace(f'{folder}/{filename}-{grd_s}.relax','bohr','angstrom')
def set_vacuum (kpoints,vacuum,folder='vacuum_conv'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe.control['calculation']="'scf'"
    qe.kpoints=kpoints

    # redefine SC geometry c=vacuum-zmin+zmax (if is centred)
    
    for i,z in enumerate(vacuum):
        newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,z+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])
        qe.control['prefix']="'%s-%d'"%(prefix,z)
        qe.system['celldm(3)']=sisl.SuperCell(newgeom.cell).length[2]/sisl.SuperCell(newgeom.cell).length[0]
        qe.write(f'{folder}/{filename}-{z}.scf')
        #replace(f'{folder}/{filename}-{z}.scf','crystal','angstrom')
        replace(f'{folder}/{filename}-{z}.scf','bohr','angstrom')
        newgeom.write(f'{folder}/{filename}-{z}.xsf')
    return qe

def set_kmesh (folder='kgrid'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe.control['calculation']="'scf'"
    grids=[]
    for i in range (16,20,2):
        grids.append([i,i,i])
    for i,grd in enumerate(grids):
       qe.kpoints=grd
       grd_s=re.sub(', ','-',str(grd).strip('[]'))
       qe.control['prefix'] = "'%s-%s'"%(prefix,grd_s)
       qe.write(f'{folder}/{filename}-{grd_s}.scf')
       #replace(f'{folder}/{filename}-{grd_s}.scf','crystal','angstrom')
       replace(f'{folder}/{filename}-{grd_s}.scf','bohr','angstrom')

def set_vacuum (kpoints,vacuum,folder='vacuum_conv'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe.control['calculation']="'scf'"
    qe.kpoints=kpoints

    # redefine SC geometry c=vacuum-zmin+zmax (if is centred)
    
    for i,z in enumerate(vacuum):
        newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,z+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])
        qe.control['prefix']="'%s-%d'"%(prefix,z)
        qe.system['celldm(3)']=sisl.SuperCell(newgeom.cell).length[2]/sisl.SuperCell(newgeom.cell).length[0]
        qe.write(f'{folder}/{filename}-{z}.scf')
        #replace(f'{folder}/{filename}-{z}.scf','crystal','angstrom')
        replace(f'{folder}/{filename}-{z}.scf','bohr','angstrom')
        newgeom.write(f'{folder}/{filename}-{z}.xsf')
    return qe
def set_degauss(kpoints,degauss,folder='degauss'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe.control['calculation']="'scf'"
    qe.kpoints=kpoints
    for i,dg in enumerate(degauss):
        qe.system['degauss']=dg
        qe.control['prefix'] = "'%s-%s'"%(prefix,dg)
        qe.write(f'{folder}/{filename}-{dg}.scf')
        replace(f'{folder}/{filename}-{dg}.scf','bohr','angstrom')
    return qe
def bands(folder='bands'):
    if not os.path.isdir('bands'):
        os.mkdir('bands')
    qe = get_inputfile()
    qe.control['calculation'] = "'bands'"
    qe.electrons['diago_full_acc'] = ".true."
    qe.electrons['conv_thr'] = 1e-8
    qe.system['nbnd'] = 36
    qe.system['force_symmorphic'] = ".true."
    qe.ktype = 'crystal'
    qe.set_path(p)
    qe.write('bands/%s.bands'%(filename))
    #replace(f'bands/{filename}.bands','crystal','angstrom')
    replace(f'{folder}/{filename}.bands','bohr','angstrom')



#convergence bands with respect vacuum third direction
def bands_vacuum(kpoints,vacuum):
    p = Path([ [[  0.0,  0.0,  0.0],'$\Gamma$'],
              [[  0.0,  0.0,  0.5],'Z']], [int(npoints)] )
    if not os.path.isdir('bands_vacuum'):
        os.mkdir('bands_vacuum')
    qe = get_inputfile()
    qe.kpoints=kpoints
    qe.control['calculation'] = "'bands'"
    qe.electrons['diago_full_acc'] = ".true."
    qe.electrons['conv_thr'] = 1e-8
    qe.system['nbnd'] = 36
    qe.system['force_symmorphic'] = ".true."
    qe.ktype = 'crystal'
    for i,z in enumerate(vacuum):
        newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,z+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])
        qe.control['prefix']="'%s-%d'"%(prefix,z)
        qe.system['celldm(3)']=sisl.SuperCell(newgeom.cell).length[2]/sisl.SuperCell(newgeom.cell).length[0]
        qe.write(f'bands_vacuum/{filename}-{z}.bands')
        qe.set_path(p)

def CCBox_nscf(kpoints,vacuum):
    if not os.path.isdir('CCBox_nscf'):
        os.mkdir('CCBox_nscf')
    qe = get_inputfile()
    qe.kpoints=kpoints
    qe.control['calculation'] = "'nscf'"
    qe.electrons['diago_full_acc'] = ".true."
    qe.electrons['conv_thr'] = 1e-8
    qe.system['nbnd'] = 150
    qe.system['force_symmorphic'] = ".true."
    for i,z in enumerate(vacuum):
        newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,z+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])
        qe.control['prefix']="'%s-%d'"%(prefix,z)
        qe.system['celldm(3)']=sisl.SuperCell(newgeom.cell).length[2]/sisl.SuperCell(newgeom.cell).length[0]
        qe.write(f'CCBox_nscf/{filename}-{z}.nscf')
        #replace(f'CCBox_nscf/{filename}-{z}.nscf','crystal','angstrom')

def CCBox_scf(kpoints,vacuum):
    if not os.path.isdir('CCBox_scf'):
        os.mkdir('CCBox_scf')
    qe = get_inputfile()
    qe.kpoints=kpoints
    qe.control['calculation'] = "'scf'"
    qe.electrons['conv_thr'] = 1e-8
    qe.system['force_symmorphic'] = ".true."
    for i,z in enumerate(vacuum):
        newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,z+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])
        qe.control['prefix']="'%s-%d'"%(prefix,z)
        qe.system['celldm(3)']=sisl.SuperCell(newgeom.cell).length[2]/sisl.SuperCell(newgeom.cell).length[0]
        qe.write(f'CCBox_scf/{filename}-{z}.scf')
        #replace(f'CCBox_scf/{filename}-{z}.scf','crystal','angstrom')

def replace(file, pattern, subst):
    # Read contents from file as a single string
    file_handle = open(file, 'r')
    file_string = file_handle.read()
    file_handle.close()

    # Use RE package to allow for replacement (also allowing for (multiline) REGEX)
    file_string = (re.sub(pattern, subst, file_string))

    # Write contents to file.
    # Using mode 'w' truncates the file.
    file_handle = open(file, 'w')
    file_handle.write(file_string)
    file_handle.close()


if __name__ == "__main__":

    #parse options
    parser = argparse.ArgumentParser(description='Test the yambopy script.')
    parser.add_argument ('-x','--xsf',        action="store_true",    help='Option xsf file provided for geometry')
    parser.add_argument('-e' ,'--ecutwfc',       action="store_true", help='Convergence of ecutwfc')
    parser.add_argument('-b' ,'--bands',       action="store_true", help='bands normal state')
    parser.add_argument('-k', '--kgrid',       action='store_true', help='kgrid convergence calculation')
    parser.add_argument('-s' ,'--scf',         action="store_true", help='Self-consistent calculation')
    parser.add_argument('-v' ,'--vacuum',         action="store_true", help='Convergence of vacuum')
    parser.add_argument('-kr' ,'--krelax',         action="store_true", help='relaxationconvergence')
    parser.add_argument('-cc' ,'--Cutoff',         action="store_true", help='Convergence of CCBox')
    parser.add_argument('-dg' ,'--degauss',         action="store_true", help='Convergence of degauss')
    args = parser.parse_args()

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    #Create the Geometry object using sisl
    geom=sisl.io.xsfSile(geomxsf).read_geometry()
    #Shift/noshift the molecule to the center of the box
    newgeom=geom.move([0,0,0])
    #set vacuum as you want
    #defvacuum=10
    #newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,defvacuum+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])

   #check output geometry
    newgeom.write('bottom_molecule.xsf')
    print(newgeom.cell)
    latticepar=sisl.SuperCell(newgeom.cell).length[0]*sisl.unit.unit_convert('Ang','Bohr') 
    # create input files and folders
    scf(scf_kpoints,folder='scf')
    set_ecutwfc(scf_kpoints,energies,folder='ecutwfc')
    set_kmesh(folder='kgrid')
    adegauss=[0.5,0.05,0.01,5e-03,5e-04,5e-05,5e-06]
    set_degauss(scf_kpoints,degauss=adegauss,folder='degauss')
    if args.krelax:
        set_kmesh_relax(folder='relax-kgrid')
    
    if args.bands:
        bands()

    if args.vacuum:
        set_vacuum(scf_kpoints,vacuum,folder='vacuum_conv')
        bands_vacuum(scf_kpoints,vacuum)
        CCBox_nscf(nscf_kpoints,vacuum)
        CCBox_scf(scf_kpoints,vacuum)

    if args.scf:
        print("running scf:")
        sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s' % (prefix))
        sch.add_arguments('#SBATCH --mincpus=32')
        sch.add_arguments('#SBATCH --ntasks=12')
        sch.add_arguments(f'#SBATCH --error={prefix}.err')
        sch.add_arguments(f'#SBATCH --output={prefix}.out')
        sch.add_command('source ~/modules_qe.load')
        sch.add_command('cd %s/scf/'%(work_path))
        sch.add_command(f'srun {pw} -ndiag 1 < {prefix}.scf > {prefix}.out' %(pw, prefix,prefix))
        sch.add_command('cp -r %s.save ../nscf'%(prefix))
        sch.write(f'{work_path}/scf/{prefix}-scf.sh')
        #os.system('sbatch %s-scf.sh' %prefix)
        print(f"scf sbatched!")

    if args.bands:
        print("running bands:")
        sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="12:00:00",name='%s-bands' % (prefix))
        sch.add_arguments('#SBATCH --mincpus=32')
        sch.add_arguments('#SBATCH --ntasks=12')
        sch.add_arguments(f'#SBATCH --error={prefix}-bands.err')
        sch.add_arguments(f'#SBATCH --output={prefix}-bands.out')
        sch.add_command('source ~/modules_qe.load')
        sch.add_command('cd %s/bands/'%(work_path))
        sch.add_command(f'srun {pw} -ndiag 1 < {prefix}.bands > {prefix}.out')
        sch.write(f'{work_path}/bands/{prefix}-scf.sh')
        os.system('sbatch bands/%s-scf.sh'%prefix)
        print(f"bands sbatched!")

    if args.ecutwfc:
        print("running convergence with respect to ecutwfc:")
        for i,en in enumerate(energies):
             sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-ecutwfc-%s' % (prefix,en))
             sch.add_command(f'#SBATCH --mincpus=32')
             sch.add_command('#SBATCH --ntasks=12')
             sch.add_command('#SBATCH -p thin')
             sch.add_command(f'#SBATCH --error={prefix}.err')
             sch.add_command(f'#SBATCH --output={prefix}.out')
             sch.add_command('source ~/modules_qe.load')
             sch.add_command('cd %s/ecutwfc/'%(work_path))
             sch.add_command(f'srun {pw} -ndiag 1 < {prefix}-{en}.scf > {prefix}-{en}.out')
             sch.write(f'{work_path}/ecutwfc/{prefix}-scf-{en}.sh')
             os.system('sbatch ecutwfc/{0}-scf-{1}.sh'.format(prefix,en))
             print(f"ecuwfc-{en} sbatched!")
    if args.degauss:
        print("running convergence with respect to degauss:")
        for i,dg in enumerate(adegauss):
             sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-ecutwfc-%s' % (prefix,dg))
             sch.add_command(f'#SBATCH --mincpus=32')
             sch.add_command('#SBATCH --ntasks=12')
             sch.add_command('#SBATCH -p thin')
             sch.add_command(f'#SBATCH --error={prefix}.err')
             sch.add_command(f'#SBATCH --output={prefix}.out')
             sch.add_command('source ~/modules_qe.load')
             sch.add_command('cd %s/degauss/'%(work_path))
             sch.add_command(f'srun {pw} -ndiag 1 < {prefix}-{dg}.scf > {prefix}-{dg}.out')
             sch.write(f'{work_path}/degauss/{prefix}-scf-{dg}.sh')
             os.system('sbatch degauss/{0}-scf-{1}.sh'.format(prefix,dg))
             print(f"degauss-{dg} sbatched!")

    if args.kgrid:
        grids=[]
        for i in range (16,20,2):
            grids.append([i,i,i])
        print("running convergence with respect to kgrid:")
        for i,grd in enumerate(grids):
             grd_s=re.sub(', ','-',str(grd).strip('[]'))
             sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-kgrid-%s' % (prefix,grd_s))
             sch.add_command(f'#SBATCH --mincpus=32')
             sch.add_command('#SBATCH -p thin')
             sch.add_command('#SBATCH --ntasks=12')
             sch.add_command(f'#SBATCH --error={prefix}.err')
             sch.add_command(f'#SBATCH --output={prefix}.out')
             sch.add_command('source ~/modules_qe.load')
             sch.add_command('cd %s/kgrid/'%(work_path))
             sch.add_command('srun %s -ndiag 1 < %s-%s.scf > %s-%s.out'%(pw,prefix,grd_s,prefix,grd_s))
             sch.write('%s/kgrid/%s-scf-%s.sh'%(work_path,prefix,grd_s))
             os.system('sbatch kgrid/{0}-scf-{1}.sh'.format(prefix,grd_s))
             print(f"kgrid-{grd_s} sbatched!")
    if args.krelax:
        grids=[]
        for i in [4,8,12]:
            grids.append([i,i,i])
        print("running convergence with respect to relax-kgrid:")
        for i,grd in enumerate(grids):
             grd_s=re.sub(', ','-',str(grd).strip('[]'))
             sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-relax-kgrid-%s' % (prefix,grd_s))
             sch.add_command(f'#SBATCH --mincpus=32')
             sch.add_command('#SBATCH -p thin')
             sch.add_command('#SBATCH --ntasks=12')
             sch.add_command(f'#SBATCH --error={prefix}.err')
             sch.add_command(f'#SBATCH --output={prefix}.out')
             sch.add_command('source ~/modules_qe.load')
             sch.add_command('cd %s/relax-kgrid/'%(work_path))
             sch.add_command('srun %s -ndiag 1 < %s-%s.relax > %s-%s.out'%(pw,prefix,grd_s,prefix,grd_s))
             sch.write('%s/relax-kgrid/%s-scf-%s.sh'%(work_path,prefix,grd_s))
             os.system('sbatch relax-kgrid/{0}-scf-{1}.sh'.format(prefix,grd_s))
             print(f"kgrid-{grd_s} sbatched!")

    # if args.vacuum:
    #     print("running convergence with respect to ecutwfc:")
    #     for i,en in enumerate(vacuum):
    #         sch = Scheduler.factory(scheduler="slurm",nodes=2,cores=48,cpus_per_task=1,walltime="12:00:00",qos='PRACE',name='%s' % ('mows2'))
    #         sch.add_command('source ~/modules_qenew.load')
    #         sch.add_command('cd %s/vacuum_cov/'%(work_path))
    #         sch.add_command('mpirun -np 96 %s -ndiag 1 < %s.scf > %s.out' %(pw, prefix,prefix))
    #         sch.add_command('cp -r %s.save ../nscf'%(prefix))
    #         sch.write(f'{work_path}-scf-{vacuum}.sh')
    #         #os.system('sbatch f'%{prefix}-scf-{i}.sh')
    #     print("done!")


