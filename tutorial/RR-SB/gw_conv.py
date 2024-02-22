#
# Convergence GW on hexagonal BN
# Alejandro Molina-Sanchez & Henrique P. C. Miranda
#
from __future__ import print_function
import sys
from yambopy     import *
from qepy        import *
from schedulerpy import *
import os
import argparse
import matplotlib.pyplot as plt

yambo = '/home/rireho/codes/yambo/branches/5.1/bin/yambo'
p2y = '/home/rireho/codes/yambo/branches/5.1/bin/p2y'
prefix = 'PbTe'
work_path='/gpfs/work2/0/prjs0229/rireho/PbTe/qe/convergencestudy/stringent/yamboconv'
bash = Scheduler.factory
database='database_18x18x18'
gw_conv='gw_conv_18x18x18'
nscf='/gpfs/work2/0/prjs0229/rireho/PbTe/qe/convergencestudy/stringent/yamboconv/nscf_18x18x18'
auxdir='screening_60b'

def create_save():
    #check if the nscf cycle is present
    if os.path.isdir(f'{nscf}/{prefix}.save'):
        print('nscf calculation found!')
    else:
        print('nscf calculation not found!')
        exit() 

    #check if the SAVE folder is present
    if not os.path.isdir(f'{database}'):
        print('preparing yambo database')
        shell = bash() 
        shell.add_command(f'mkdir -p {database}')
        shell.add_command(f'cp yambo.in {nscf}/{prefix}.save')
        shell.add_command(f'cd {nscf}/{prefix}.save; {p2y}; {yambo}')
        shell.add_command(f'mv SAVE  ../../{database}/')
        shell.run()
        shell.clean()

def get_inputfile():
    """ Define a Yambo GW input file for Wse2
    """
    if not os.path.exists('gw_conv/SAVE'):
        os.system(f'ln -s {work_path}/{database}/SAVE {work_path}/{gw_conv}/')
    y = YamboIn.from_runlevel('%s -o -d -x -p p -X -d -r -g n'%yambo,folder=f'{gw_conv}')
    #Read values from QPkrange
    values, units= y['QPkrange']
    kpoint_start, kpoint_end, band_start, band_end = values
    #set the values of QPkrange
    y['ElecTemp']=[0.000,'eV']
    y['BoseTemp']=[0.000,'eV']
    y['QPkrange'] = [10,10,46,47] #usually calculate only q point at gap
    y['Chimod']= "HARTREE"
    y['PAR_def_mode']  = "balanced"         # [PARALLEL] Default distribution mode ("balanced"/"memory"/"workload")
    y['X_and_IO_CPU']  = "2.1.2.8.1"       # [PARALLEL] CPUs for each role
    y['X_and_IO_ROLEs']= "q.g.k.c.v"        # [PARALLEL] CPUs roles (q,g,k,c,v)
    y['DIP_CPU']  = "4.8.1"                # [PARALLEL] CPUs for each role
    y['DIP_ROLEs']= "k c v"                 # [PARALLEL] CPUs roles (k,c,v)
    y['SE_CPU']   = "4.1.8"                # [PARALLEL] CPUs for each role
    y['SE_ROLEs'] = "q.qp.b"                # [PARALLEL] CPUs roles (q,qp,b)    
    y['EXXRLvcs'] = [100,'mRy']       # Self-energy. Exchange
    y['BndsRnXp'] = [[1,180],'']          # Screening. Number of bands
    y['NGsBlkXp'] = [100,'mRy']        # Cutoff Screening
    y['GbndRnge'] = [[1,56],'']          # Self-energy. Number of bands
    y['FFTGvecs'] = [100,'mRy']       #FFT
    y['GTermKind']= 'BG'
    y['XTermKind']= 'BG'
    y.write(f'{gw_conv}/gw_ppa.in')
    return y

def gw_convergence():
    #create the folder to run the calculation
    if not os.path.isdir(f'{gw_conv}'):
        shell = bash() 
        shell.add_command(f'mkdir -p {gw_conv}')
        shell.add_command(f'cp -r {database}/SAVE {gw_conv}/')
        shell.run()
        shell.clean()
    conv = {#'FFTGvecs': [[50,60,70,80,90,100],'Ry']}
            #'FFTGvecs': [[1,5,10,15,20,30,40,50],'Ry']} #First uncomment only FFTGves, then EXXRLvcs etc... one at a time
            # 'EXXRLvcs': [[1,2,5,10,20,30,40,50,60,70,80,90,100],'Ry']}
            # 'NGsBlkXp': [[1,2,3,4,5,6,7,8,9,10], 'Ry']}
            'BndsRnXp': [[[1,110]],'']}
            #[[[1,40],[1,50],[1,60],[1,70],[1,80],[1,90],[1,100],[1,110],[1,120],[1,130],[1,140],[1,150],[1,160],[1,170],[1,180],[1,190],[1,200]],'']}
            # 'GbndRnge': [[[1,50],[1,60],[1,70],[1,80],[1,90],[1,100],[1,110],[1,120],[1,130],[1,140],[1,150],[1,160],[1,170],[1,180],[1,190],[1,200]],''] }
    return conv
def run(filename):
        """ Function to be called by the optimize function """
        folder = filename.split('.')[0]
        print(filename,folder)
        sch = Scheduler.factory(scheduler="slurm",nodes=1,cpus_per_task=1,walltime="03:00:00",name='%s-gw-%s' % (prefix,folder))
        #sch.add_command(f'#SBATCH --mincpus=32')
        sch.add_command('#SBATCH -p thin')
        sch.add_command('#SBATCH --exclusive')
        sch.add_command('#SBATCH --ntasks=32') #you can change me
        sch.add_command(f'#SBATCH --error={prefix}-{folder}.err')
        sch.add_command(f'#SBATCH --output={prefix}-{folder}.out')
        sch.add_command('source ~/modules_yambo.load')
        sch.add_command(f'cd {work_path}/{gw_conv}/')
        sch.add_command('rm -f *.json %s/o-*'%folder) #cleanup
        sch.add_command('srun %s -F %s -J %s -C "%s" 2> %s.log'%(yambo,filename,folder,folder,folder))
        sch.write(f'{work_path}/{gw_conv}/{folder}.sh')
        os.system(f'sbatch {gw_conv}/{folder}.sh')
        print(f"{folder} sbatched!")


if __name__ == "__main__":
    #parse options
    parser = argparse.ArgumentParser(description='GW convergence')
    parser.add_argument('-c'  ,'--convergence',  action="store_true", help='Run convergence calculations')


    args = parser.parse_args()

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    if args.convergence:
        print('running convergence GW with respect to FFTGvecs')        
        create_save()
        conv=gw_convergence()
        y=get_inputfile()
        y.optimize(conv,folder=f'./{gw_conv}',run=run,ref_run=False)
