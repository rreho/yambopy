##############################################################################
#
# Author: Alejandro Molina-Sanchez
# Run real-time simulations with yambo
# 
# Warning: Real-time simulations requires several data folders for running
# properly. Before using this scripts compulsively is recommended
# to understand the different run levels.
#
# Intructions: 
# The dictionary 'job' stores the variable of the calculation
# calculation : 'collision', 'tdsex', 'negf', 'dissipation'
# folder-col  : collision data
# folder-run  : results (only work if collisions have been previously calculated)
# DG          : True or False if we use the double-grid
# nodes       : cpu-dependent variable
# cores       :          "
#
# Calculations are done inside the folder FixSymm (feel free to rename it)
#
##############################################################################
#from __future__ import print_function
from yambopy.inputfile import *
from pwpy.inputfile import *
from pwpy.outputxml import *

yambo    = 'yambo'
yambo_rt = 'yambo_rt'
ypp_rt   = 'ypp_rt'
ypp_ph   = 'ypp_ph'

# check if the database is present
if not os.path.isdir('database'):
    os.mkdir('database')

#check if the nscf cycle is present
if os.path.isdir('nscf/si.save'):
    print('nscf calculation found!')
else:
    print('nscf calculation not found!')
    exit() 

#check if the SAVE folder is present
if not os.path.isdir('database/SAVE'):
    print('preparing yambo database')
    os.system('cd nscf/si.save; p2y')
    os.system('cd nscf/si.save; yambo')
    os.system('mv nscf/si.save/SAVE database')

#check if the FixSymm folder is present
if os.path.isdir('FixSymm/SAVE'):
  print('symmetries for carrier dynamics ready') 
if not os.path.isdir('FixSymm'):
  print('breaking symmetries')
  os.system('mkdir -p rt')
  os.system('cp -r database/SAVE rt')
  ypp = YamboIn('ypp_ph -n -V all',folder='rt',filename='ypp.in')
  ypp['Efield1'] = [[0.0,1.0,0.0],''] # Field in the Y-direction
  ypp.arguments.append('RmTimeRev')   # Remove Time Symmetry
  ypp.write('rt/ypp.in')
  os.system('cd rt ; %s -F ypp.in'%ypp_ph )
  os.system('cd rt ; cd FixSymm ; yambo ' )
  os.system('mv rt/FixSymm .')
  os.system('rm -r rt')
  print('FixSymm folder created')

# you can select the calculation among : 'collision', 'tdsex', 'pump', 'dissipation'

job = dict()
job['calculation'] = 'pump'
job['folder-col']  = 'COLLISION'
job['folder-run']  = ''
job['DG']          = (False,'DG-60x60')
job['nodes']       = 1
job['cores']       = 1

if job['calculation']=='collision':
  run = YamboIn('%s -r -e -v c -V all'%yambo_rt,folder='FixSymm')
  os.system('rm FixSymm/yambo.in')
if job['calculation']=='tdsex':
  run = YamboIn('%s -q p -v c -V all'%yambo_rt,folder='FixSymm')
if job['calculation']=='pump':
  run = YamboIn('%s -q p -v c -V all'%yambo_rt,folder='FixSymm')
if job['calculation']=='dissipation':
  run = YamboIn('%s -s p -q p -v c -V all'%yambo_rt,folder='FixSymm')

# System Common variables
run['FFTGvecs']  = [5,'Ha']
run['EXXRLvcs']  = [5,'Ha']
run['SCBands']   = [[4,5],'']

# Collision variables
if job['calculation']=='collision':
  run['NGsBlkXp']  = [ 100,'mHa']
  run['BndsRnXs' ] = [[1,30],'']
  #run['CUTGeo']    = 'box z'               # Only in layers
  #run['CUTBox']    = [[0,0,38], '']
  run.write('FixSymm/03_COLLISION')

# Common time-dependent variable
if job['calculation']=='tdsex' or 'pump' or 'dissipation':
  run['GfnQP_Wv']   = [ [0.10,0.00,0.00],'' ]
  run['GfnQP_Wc']   = [ [0.10,0.00,0.00],'' ]
  run['GfnQPdb']    = 'none' 
  # Time-propagation 
  run['RTstep']     = [0.50,'fs']
  run['NETime']     = [ 100,'fs']
  run['Integrator'] = "RK2 RWA"
  run['IOtime']     = [ [1.00, 1.00, 1.00], 'fs' ] 
  # Pump Pulse
  run['Field1_Int']  = [ 1E6 , 'kWLm2']    # Intensity pulse
  run['Field1_Dir']  = [[0.0,1.0,0.0],'']  # Polarization pulse
  run['Field1_pol']  = "linear"            # Polarization type (linear or circular) 

# Time-dependent COHSEX -- DELTA PULSE
if job['calculation']=='tdsex':
  run['Field1_kind'] = "DELTA"
  run.write('FixSymm/04_TDSEX')

# Pumping with finite pulse
if job['calculation']=='pump' or 'dissipation':
  run['Field1_kind'] = "QSSIN"
  run['Field1_Damp'] = [  2,'fs']
  run['Field1_Freq'] = [[2.5,0.0],'eV']

if job['calculation']=='pump':
  run.write('FixSymm/05_NEGF')

# Pumping with finite pulse and electron-phonon dissipation
if job['calculation']=='dissipation':
# Interpolation 
  run['LifeInterpKIND']  = 'FLAT'
  run['LifeInterpSteps'] = [ [10.0,2.5], 'fs' ] 
  run.write('FixSymm/06_DISS')

# Run

# Collisions

if job['calculation'] == 'collision':
  print('running yambo-collision')
  print ('cd FixSymm; %s -F 03_COLLISION -J %s'%(yambo_rt,job['folder-col']))
  os.system('cd FixSymm; %s -F 03_COLLISION -J %s'%(yambo_rt,job['folder-col']))

# Time-dependent without/with Double Grid 

if job['calculation'] == 'tdsex':
  job['folder-run'] += 'tds-int-%s' % ( str(int(run['Field1_Int'][0])) )
  print('running TD-COHSEX in folder: ' + str(job['folder-run']))
  if not os.path.isdir('FixSymm/'+job['folder-col']):
    print 'Collisions not found'
    exit()
  if job['DG'][0]:
    print('with Double Grid from folder %s'%job['DG'][1])
    print('cd FixSymm; %s -F 04_TDSEX -J \'%s,%s,%s\''%(yambo_rt,job['folder-run'],job['folder-col'],job['DG'][1]))
    os.system('cd FixSymm; %s -F 04_TDSEX -J \'%s,%s,%s\' -C %s'%(yambo_rt,job['folder-run'],job['folder-col'],job['DG'][1],job['folder-run']))
  else:
    print ('cd FixSymm; %s -F 04_TDSEX -J \'%s,%s\''%(yambo_rt,job['folder-run'],job['folder-col']))
    os.system ('cd FixSymm; %s -F 04_TDSEX -J \'%s,%s\' -C %s'%(yambo_rt,job['folder-run'],job['folder-col'],job['folder-run']))

# Time-dependent with a pulse and without/with Double Grid 

if job['calculation'] == 'pump':
  print('running pumping with finite pulse')
  job['folder-run'] += 'negf-int-%s-damp-%sfs-freq-%seV' % ( str(int(run['Field1_Int'][0])), str(run['Field1_Damp'][0]),run['Field1_Freq'][0][0] )
  print('running NEGF in folder: ' + str(job['folder-run']))
  if not os.path.isdir('FixSymm/'+job['folder-col']):
    print 'Collisions not found'
    exit()
  if job['DG'][0]:
    print('with Double Grid from folder %s'%job['DG'][1])
    print('%s -F 05_NEGF -J \'%s,%s,%s\' -C %s'%(yambo_rt,job['folder-run'],job['folder-col'],job['DG'][1],job['folder-run']))
  else:
    print ('%s -F 05_NEGF -J \'%s,%s\''%(yambo_rt,job['folder-run'],job['folder-col']))
    os.system ('cd FixSymm; %s -F 05_NEGF -J \'%s,%s\' -C %s'%(yambo_rt,job['folder-run'],job['folder-col'],job['folder-run']))

# Time-dependent with a pulse and dissipation and without/with Double Grid 

if job['calculation'] == 'dissipation':
  print('running pumping with finite pulse and with electron-phonon scattering')
  print('this run level needs the GKKP folder to run')
  exit()
'''
  if job['DG'][0]:
    print('with Double Grid from folder %s'%job['DG'][1])
    print('%s -F 06_DISS -J \'%s,%s,%s\''%(yambo_rt,job['folder-run'],job['folder-col'],job['DG'][1]))
  else:
    print ('%s -F 06_DISS -J \'%s,%s\''%(yambo_rt,job['folder-run'],job['folder-col']))
    os.system ('cd FixSymm; %s -F 06_DISS -J \'%s,%s\''%(yambo_rt,job['folder-run'],job['folder-col']))
'''