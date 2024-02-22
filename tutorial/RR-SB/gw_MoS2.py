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
geomxsf='MoS2.xsf' 
scf_kpoints = [6,6,1]
nscf_kpoints = [24,24,1]
db_kpoints = [24,24,1]
energies = [80,90,100,110,120,130,140,150]
vacuum = np.arange(8,21,1)
filename='mos2'
prefix = 'mos2'
work_path='/scratch-shared/rireho_scratch/MoS2/convergencestudy/'
npoints = 10
p = Path([ [[  0.0,  0.0,  0.0],'$\Gamma$'],
              [[  0.5,  0.0,  0.0],'M'],
              [[  1./3,1./3,0.],'K'],
              [[ 0.0,0.0,0.0], '$\Gamma$']],[npoints,npoints,npoints] )
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
        qe.system['ibrav'] = 4
        qe.atypes = {'Mo': [95.94, "Mo.upf"],
                 'S': [32.065,"S.upf"],}        
        qe.control['prefix'] = "'%s'"%prefix
        qe.control['verbosity'] = "'high'"
        qe.control['wf_collect'] = '.true.'
        qe.control['pseudo_dir'] = "'/home/rireho/pseudos/SO_stringent'"
        qe.control['restart_mode']="'from_scratch'"
        qe.control['tprnfor']='.true.'
        qe.control['tstress']='.true.'
        qe.control['forc_conv_thr']= '1e-06'
        qe.control['etot_conv_thr']='1e-09'
        qe.control['max_seconds']='82800'
        qe.system['force_symmorphic'] = '.true.'
        qe.system['celldm(1)'] =  6.0126744630402
        qe.system['celldm(3)'] = 4.124106681962734
        qe.system['ecutwfc'] = 110
        qe.system['occupations'] = "'fixed'"
        qe.system['nat'] = newgeom.na
        qe.system['ntyp'] = 2
        qe.system['lspinorb']='.true.'
        qe.system['noncolin']='.true.'
        qe.system['vdw_corr']="'grimme-d2'"
        qe.system['assume_isolated']="'2D'"
        qe.kpoints = [9, 9, 1]
        qe.electrons['conv_thr'] = 1e-08
        qe.electrons['diago_full_acc'] = ".true."        
        qe.electrons['electron_maxstep'] = 1000
        qe.set_atoms([['Mo',[0.666666666,        0.333333333,      0.0000000000]],
                    ['S',[0.333333333,        0.666666666,        0.1187627885]],
                    ['S',[0.333333333,        0.666666666,        -0.1187627885    ]]])
    else:
        qe = PwIn()
        qe.system['ibrav'] = 4
        qe.atypes = {'Mo': [95.94, "Mo.upf"],
                 'S': [32.065,"S.upf"],}        
        qe.control['prefix'] = "'%s'"%prefix
        qe.control['verbosity'] = "'high'"
        qe.control['wf_collect'] = '.true.'
        qe.control['pseudo_dir'] = "'/home/rireho/pseudos/SO_stringent'"
        qe.control['restart_mode']="'from_scratch'"
        qe.control['tprnfor']='.true.'
        qe.control['tstress']='.true.'
        qe.control['forc_conv_thr']= '1e-06'
        qe.control['etot_conv_thr']='1e-09'
        qe.control['max_seconds']='82800'
        qe.system['force_symmorphic'] = '.true.'
        qe.system['celldm(1)'] = sisl.SuperCell(newgeom.cell).length[0]*sisl.unit.unit_convert('Ang','Bohr') 
        qe.system['celldm(3)'] =sisl.SuperCell(newgeom.cell).length[2]/sisl.SuperCell(newgeom.cell).length[0]
        qe.system['ecutwfc'] = 100
        qe.system['occupations'] = "'fixed'"
        qe.system['nat'] = newgeom.na
        qe.system['ntyp'] = 2
        qe.system['lspinorb']='.true.'
        qe.system['noncolin']='.true.'
        qe.system['vdw_corr']="'grimme-d2'"
        qe.system['assume_isolated']="'2D'"
        qe.electrons['diago_full_acc'] = ".true."
        qe.electrons['conv_thr'] = 1e-12
        qe.electrons['electron_maxstep'] = 1000
        qe.set_atoms([[newgeom.atoms[i].symbol,newgeom.fxyz[i]] for i in range(newgeom.na)])
    return qe

#scf
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
        #replace(f'{folder}/{filename}-{en}.scf','bohr','angstrom')
    return qe

def scf(kpoints,folder='scf'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe = get_inputfile()
    qe.control['calculation'] = "'scf'"
    qe.kpoints = kpoints
    qe.write('%s/%s.scf'%(folder,filename))

def nscf(kpoints,folder='nscf'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe = get_inputfile()
    qe.control['calculation'] = "'nscf'"
    qe.system['nbnd'] = 200    
    qe.electrons['conv_thr'] = 1e-10
    qe.kpoints = kpoints
    qe.write('%s/%s.nscf'%(folder,filename))

def set_kmesh_relax (folder='relax-kgrid'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    qe=get_inputfile()
    qe = get_inputfile()
    qe.control['calculation'] = "'vc-relax'"
    qe.ions['ion_dynamics']  = "'bfgs'"
    qe.cell['cell_dynamics']  = "'bfgs'"
    qe.cell['press']="0.d0"
    qe.cell['press_conv_thr']=0.01
    grids=[]
    for i in [6,12,18,24,28]:
        grids.append([i,i,1])
    for i,grd in enumerate(grids):
       qe.kpoints=grd
       grd_s=re.sub(', ','-',str(grd).strip('[]'))
       qe.control['prefix'] = "'%s-%s'"%(prefix,grd_s)
       qe.write(f'{folder}/{filename}-{grd_s}.relax')

def bands():
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
    parser.add_argument('-s' ,'--scf',         action="store_true", help='Self-consistent calculation')
    parser.add_argument('-n' ,'--nscf',         action="store_true", help='Self-consistent calculation')
    parser.add_argument('-r' ,'--relax',         action="store_true", help='relaxationconvergence')
    parser.add_argument('-u','--update',    action="store_true",help='update positions')
    parser.add_argument('-kr' ,'--krelax',         action="store_true", help='relaxationconvergence')    
    args = parser.parse_args()

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    #Create the Geometry object using sisl
    geom=sisl.io.xsfSile(geomxsf).read_geometry()
    #Shift the molecule to the center of the box
    newgeom=geom.move([0,0,0])

    #set vacuum as you want
    defvacuum=10
    newgeom.set_sc([newgeom.cell[0],newgeom.cell[1],[0,0,defvacuum+np.max(newgeom.xyz[:,2])-np.min(newgeom.xyz[:,2])]])

    #check output geometry
    newgeom.write('centred_molecule.xsf')
    print(newgeom.cell)

    if args.ecutwfc:
        set_ecutwfc(scf_kpoints,energies,folder='ecutwfc')
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

    if args.scf:
       scf(scf_kpoints)
       sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-scf' % (prefix))
       sch.add_command(f'#SBATCH --mincpus=32')
       sch.add_command('#SBATCH -p thin')
       sch.add_command('#SBATCH --ntasks=12')
       sch.add_command(f'#SBATCH --error={prefix}.err')
       sch.add_command(f'#SBATCH --output={prefix}.out')
       sch.add_command('source ~/modules_qe.load')
       sch.add_command('cd %s/scf/'%(work_path))
       sch.add_command('srun %s -ndiag 1 < %s.scf > %s.out'%(pw,prefix,prefix))
       sch.write('%s/scf/%s-scf.sh'%(work_path,prefix))
       os.system('sbatch scf/{0}-scf.sh'.format(prefix))
       print(f"scf sbatched!")
    if args.nscf:
       nscf(nscf_kpoints)
       sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-nscf' % (prefix))
       sch.add_command(f'#SBATCH --mincpus=32')
       sch.add_command('#SBATCH -p thin')
       sch.add_command('#SBATCH --ntasks=12')
       sch.add_command(f'#SBATCH --error={prefix}.err')
       sch.add_command(f'#SBATCH --output={prefix}.out')
       sch.add_command(f'#SBATCH --begin=now+30')
       #sch.add_command('source ~/modules_qe.load')
       sch.add_command('cd %s/nscf/'%(work_path))
       sch.add_command(f'cp -r {work_path}/scf/{prefix}.save {work_path}/nscf/')
       sch.add_command('srun %s -ndiag 1 < %s.nscf > %s.out'%(pw,prefix,prefix))
       sch.write('%s/nscf/%s-nscf.sh'%(work_path,prefix))
       #os.system('sbatch nscf/{0}-nscf.sh'.format(prefix))
       print(f"nscf sbatched!")
    if args.bands:
       bands()
       sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-bands' % (prefix))
       sch.add_command(f'#SBATCH --mincpus=32')
       sch.add_command('#SBATCH -p thin')
       sch.add_command('#SBATCH --ntasks=12')
       sch.add_command(f'#SBATCH --error={prefix}.err')
       sch.add_command(f'#SBATCH --output={prefix}.out')
       sch.add_command('source ~/modules_qe.load')
       sch.add_command('cd %s/bands/'%(work_path))
       sch.add_command(f'cp -r {work_path}/scf/{prefix}.save {work_path}/bands/')
       sch.add_command('srun %s -ndiag 1 < %s.bands > %s.out'%(pw,prefix,prefix))
       sch.write('%s/bands/%s-bands.sh'%(work_path,prefix))
       os.system('sbatch bands/{0}-bands.sh'.format(prefix))
       print(f"nscf sbatched!")
    if args.krelax:
        set_kmesh_relax(folder='relax-kgrid')
        grids=[]
        for i in [12,18,24,30]:
            grids.append([i,i,1])
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

    if args.relax:
        grd=[10,10,10]
        grd_s=re.sub(', ','-',str(grd).strip('[]'))
        sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="01:00:00",name='%s-relax-%s' % (prefix,grd_s))
        sch.add_command(f'#SBATCH --mincpus=32')
        sch.add_command('#SBATCH -p thin')
        sch.add_command('#SBATCH --ntasks=12')
        sch.add_command(f'#SBATCH --error={prefix}.err')
        sch.add_command(f'#SBATCH --output={prefix}.out')
        sch.add_command('source ~/modules_qe.load')
        sch.add_command('cd %s/relax/'%(work_path))
        sch.add_command('srun %s -ndiag 1 < %s-%s.relax > %s-%s.out'%(pw,prefix,grd_s,prefix,grd_s))
        sch.write('%s/relax/%s-scf-%s.sh'%(work_path,prefix,grd_s))
        #os.system('sbatch relax/{0}-scf-{1}.sh'.format(prefix,grd_s))
        print(f"kgrid-{grd_s} sbatched!")
    if args.update:
        update_positions('PbTe-12-12-12','./','.')
