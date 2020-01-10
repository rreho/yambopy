#
# Author: Henrique Pereira Coutada Miranda
# Run a BSE calculation using yambo
#
from __future__ import print_function
from builtins import range
import sys
from yambopy import *
from qepy import *
from schedulerpy import *
import argparse
import matplotlib.pyplot as plt

prefix = 'bn'
folder = 'bse'
yambo  = 'yambo'
p2y    = 'p2y'
ypp    = 'ypp'
layer_separation = 12
bash = Scheduler.factory

def create_save():
    #check if the nscf cycle is present
    if os.path.isdir('nscf/%s.save'%prefix):
        print('nscf calculation found!')
    else:
        print('nscf calculation not found!')
        exit()

    #check if the SAVE folder is present
    if not os.path.isdir('database'):
        print('preparing yambo database')
        shell = bash()
        shell.add_command('mkdir -p database')
        shell.add_command('cd nscf/%s.save; %s; %s'%(prefix, p2y, yambo))
        shell.add_command('mv SAVE ../../database/')
        shell.run()
        shell.clean()

"""
    if doublegrid:
        #check if the double grid nscf cycle is present
        if os.path.isdir('nscf_double/%s.save'%prefix):
            print('nscf_double calculation found!')
        else:
            print('nscf_double calculation not found!')
            exit()

        if not os.path.isdir('database_double/SAVE'):
            print('preparing yambo double database')
            shell = scheduler()
            shell.add_command('pushd nscf_double/%s.save; %s; %s'%(prefix,p2y,yambo))
            shell.add_command('popd')
            shell.add_command('mkdir -p database_double')
            shell.add_command('mv nscf_double/%s.save/SAVE database_double'%prefix)
            shell.run()

        if os.path.isfile("%s/SAVE/ndb.Double_Grid"%folder):
            #initialize the double grid
            print("creating double grid")
            yppin = YamboIn('ypp -m',filename='ypp.in',folder='database')
            yppin['DbGd_DB1_paths'] = ["../database_double"]
            yppin.write('database/ypp.in')
            shell = scheduler()
            shell.add_command('cd database; %s'%ypp)
            shell.add_command('mv SAVE/ndb.Double_Grid ../%s/SAVE'%folder)
            print(shell)
            shell.run()
"""

def run(nthreads=1,cut=False):
    #create the folder to run the calculation
    if not os.path.isdir('bse'):
        shell = bash()
        shell.add_command('mkdir -p bse')
        shell.add_command('cp -r database/SAVE bse/')
        shell.run()
        shell.clean()

    #create the yambo input file
    y = YamboIn.from_runlevel('%s -r -b -o b -k sex -y d -V all'%yambo,folder='bse')

    if cut:
        y['CUTGeo'] = 'box z'
        y['CUTBox'] = [0,0,layer_separation-1]

    y['FFTGvecs'] = [30,'Ry']
    y['NGsBlkXs'] = [1,'Ry']
    y['BndsRnXs'] = [1,80]
    y['BSEBands'] = [3,6]
    y['BEnSteps'] = 500
    y['BEnRange'] = [[0.0,10.0],'eV']
    y['KfnQP_E']  = [2.91355133,1.0,1.0] #some scissor shift
    y.arguments.append('WRbsWF')

    if nthreads > 1:
        y['X_all_q_ROLEs'] = "q"
        y['X_all_q_CPUs'] = "%d"%nthreads

    y.write('bse/yambo_run.in')

    print('running yambo')
    shell = bash()
    if nthreads <= 1:
        shell.add_command('cd bse; %s -F yambo_run.in -J yambo'%yambo)
    else:
        shell.add_command('cd bse; mpirun -np %d %s -F yambo_run.in -J yambo'%(nthreads,yambo))
    shell.run()

if __name__ == "__main__":

    #parse options
    parser = argparse.ArgumentParser(description='Run BSE calculations on BN.')
    """parser.add_argument('-dg','--doublegrid', action="store_true", help='Use double grid')
    """
    parser.add_argument('-r', '--run',        action="store_true", help='Run BSE calculation')
    parser.add_argument('-c', '--cut',        action="store_true", help='Use coulomb truncation')
    parser.add_argument('-t' ,'--nthreads',                        help='Number of threads', default=1)
    args = parser.parse_args()
    nthreads = int(args.nthreads)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    cut = args.cut
    """dg = args.doublegrid
    """
    create_save()
    if args.run:     run(nthreads,cut)
