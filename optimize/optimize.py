import os
import shutil
import random
from time import time
from datetime import datetime
from optparse import OptionParser
import subprocess
import numpy as np
from scipy.optimize import minimize
from pso_mod import pso

PWD=os.getcwd()

random.seed()
start_time = datetime.now()
start_t = time()

# python ./optimize.py -r 1000 -S 256 -p 16

parser = OptionParser()

#parser = OptionParser()

parser.add_option("-r", type="int", dest="repeat", default=1)

parser.add_option("-S", type="int", dest="S", default=0)

parser.add_option("-p", type="int", dest="p", default=1)

options, args = parser.parse_args()

# default to neg (if objective is a physically positive value)
if options.S > 0:
    sign = 1.0
#        sign=-1.0
else:
    sign = -1.0

print(options)

# lower and upper bounds on molar fractions
lower, upper = 0.00, 1.00

# physical constants
e = 1.602176565e-19  # J/eV
kB = 1.3806488e-23  # J/K

# absolute temperature (K)
T = 300


# Boltzmann's constant times temperature (eV)
#kBT = (kB/e)*T

bufsize = 0
#outfile=open ('results','a', buffering=bufsize)
# if os.path.exists('results'):
#        outfile=open ('results','a')
# else:
#        outfile=open ('results','w')

# infeasibe penalty (just enough to make the result positive)
infeasible_penalty = 1000000000

def is_non_zero_file(fpath):
    """is_non_zero_file
    verify that file exists
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

# optimize over error result x, constrained to not nans
def is_not_nan(x):
    """is_not_nan
    positive penalty for unfeasible parameter sets 
    """
    forces_output = open(fpath, 'r')
    if all(['nan' not in line for line in b]):
        return 0
    else:
        return infeasible_penalty

# read the input training data from text files

class RFFOpt():
    """RFFOpt
    Optimize ReaxFF forcefields using multiple steps:
    - Create dictionary containing allowed parameter ranges for the free parameters
    - Pass this info to the optimizer, which runs LAMMPS and pulls outputs to compare
    """

    def __init__(self, params_path, trainset_path):
        """__init__.

        Parameters
        ----------
        params_path : string
            Path to starting parameters
        trainset_path : string
            Path to training data
        param_Nminmax : dict
            Lower and upper bounds for the free parameters
        params_Names : list
            All the free parameters to be varied during optimization
        """
        ## properly intiialize
        self.params_path = params_path
        self.trainset_path = trainset_path
        ## import parameter ranges
        self.params_Nminmax = {}
        self.params_Names = []
        params_classifier = {}

        with open(self.params_path, "r") as a:
            for line in a:
                varname_str   = line.split()[0].strip()
                minrange_val    = float(line.split()[1].strip())
                maxrange_val    = float(line.split()[2].strip())
                self.params_Nminmax[ varname_str ] = ( minrange_val, maxrange_val )
                self.params_Names.append( varname_str )
        print('Input Data', self.params_Names, self.params_Nminmax)
        boundpass = []
        for i in self.params_Names:
            boundpass.append(self.params_Nminmax[i])

        self.referencedata_dict = {}
        with open(trainset_path, 'r') as e:
            for lineno,line in enumerate(e):
                if line[0].strip() == '#':
                    pass
                else:
                    traintype      = str( line.split('XX')[0].strip() )

                    if traintype == "bulkeng":
                        trainweight = float( line.split('XX')[1].strip() )
                        trainval = float( line.split('XX')[3].strip() )

                        traincalc = line.split('XX')[2].strip()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc, trainval ]

                    elif traintype == "coheng":
                        trainweight    = float( line.split('XX')[1].strip() )
                        trainval       = float( line.split('XX')[4].strip() )

                        traincalc     = line.split('XX')[2].strip().split()

                        traineqn       = line.split('XX')[3].strip().strip('\"')
                        traineqn_list = traineqn.split()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc, traineqn_list, trainval ]

                    elif traintype == "surfeng":
                        trainweight    = float( line.split('XX')[1].strip() )
                        trainval       = float( line.split('XX')[4].strip() )

                        traincalc      = line.split('XX')[2].strip()

                        traineqn       = line.split('XX')[3].strip().strip('\"')
                        traineqn_list = traineqn.split()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc, traineqn_list, trainval ]

                    elif traintype == "energydiffeq":
                        trainweight    = float( line.split('XX')[1].strip() )
                        trainval       = float( line.split('XX')[4].strip() )

                        traincalcs     = line.split('XX')[2].strip()
                        traincalc_list = traincalcs.split()

                        traineqn       = line.split('XX')[3].strip().strip('\"')
                        traineqn_list = traineqn.split()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc_list, traineqn_list, trainval ]

                    elif traintype == "forceval":
                        trainweight    = float( line.split('XX')[1].strip() )
                        trainval       = float( line.split('XX')[3].strip() )

                        traincalc      = line.split('XX')[2].strip()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc, trainval ]

                    elif traintype == "fbond":
                        trainweight    = float( line.split('XX')[1].strip() )
                        trainval       = float( line.split('XX')[4].strip() )

                        traincalc      = line.split('XX')[2].strip()

                        trainind       = line.split('XX')[3].strip()
                        trainind_list  = trainind.split()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc, trainind_list, trainval ]

                    elif traintype == "angle":
                        trainweight    = float( line.split('XX')[1].strip() )
                        trainval       = float( line.split('XX')[4].strip() )

                        traincalc      = line.split('XX')[2].strip()

                        trainind       = line.split('XX')[3].strip()
                        trainind_list  = trainind.split()

                        self.referencedata_dict[ str(lineno) ] = [ traintype, trainweight, traincalc, trainind_list, trainval ]

                    else:
                        print("Trainset entry type " + str(traintype) + " not available" + "\n")
                        NotImplementedError('Is that type of training data implemented yet in the objective function?')
        print(self.referencedata_dict)

# instantiate training data and free parameter names/ranges
job = RFFOpt(params_path = './params_ranges', trainset_path = './trainset')

def collect_ene(ene_dir):
    """collect_ene
    collect energy from log.lammps
    """
    grepout = os.popen("grep -n 'Energy initial' " + ene_dir)
    line = grepout.read()
    if not line:
        # error if the file wasn't located
        IOError('Couldn''t find the ''Energy Initial'' line in log.lammps for ' + ene_dir)
    else:
        lineno = int(line.split()[0].strip(':'))
    a = open(ene_dir, 'r')
    b = a.readlines()
    if lineno:
        ene = b[lineno].split()[-1]
    else:
        ValueError('Error grepping for the energy in log.lammps for ' + ene_dir)
    return ene

def collect_dat(dat_dir):
    """collect dat
    collect dat from forces.dat
    """
    try:
        a = open(dat_dir, 'r')
    except:
        IOError('forces.dat for this directory: ' + dat_dir + '\n' + 'did not copy correctly or the LAMMPS task did not run correctly')

    if a:
        b = a.readlines()
        # trim the header and the timestep lines in between so only the data remains
        c = [line for line in b[9:] if 'ITEM' not in line and len(line.strip().split()) == 7]
    else:
        ValueError('forces.dat is empty for ' + dat_dir + '\n' + 'did not copy correctly or the LAMMPS task did not run correctly')
    return np.genfromtxt(c)
    



# error scaling parameters and other necessary refs for objective functions
alpha = [1, 1, 1]
#scale_mult = 0.0562
#scale_exp = 0.5587
#rand_add = 0.4
#rand_mult = 0.1
eng_exp_scale = 2
frc_exp_scale = 2
pos_exp_scale = 2
axesmap = {'x' : 0, 'y' : 1, 'z' : 2}
# objective function
def objective(x):
    """objective
    objective function for global optimization using pyswarm pso
    modified for ReaxFF forcefield parameter optimization (Cu Ni O)
    x : list
        free parameter values chosen for one particle
    """
    # use readlines and lammps/pizza logfile.py to pull the necessary data 
    # using bash that executes a custom pizza.py script to obtain data
    # make the temporary forcefield from the list of passed parameters x
    print('x is the following: ', x)
    with open(os.path.join(PWD, 'ffield.reax'), 'r') as staticfile:
        FFdata = staticfile.read()

    #TODO MAKE RANDOM NUMBER HERE FOR THE BASH SCRIPTS TO RUN THIS PARTICLE IN AS TEMP DIR
    rand_folder= 'run-'+str(random.randrange(1e10))
    os.makedirs(os.path.join(PWD, rand_folder))
    for i in range(len(job.params_Names)):
        FFdata = FFdata.replace(str(job.params_Names[i]) + " ", '{0:.4f}'.format(x[i]))

    with open(os.path.join(PWD ,rand_folder, 'ffield.reax.temp'), 'w') as tempfile:
        tempfile.write(FFdata)
    # run the forcefield for the bulk structures used in later calculations
    #try:
    #   ene = os.popen("bash ./run_lammps.bash " + trainstruct + rand_folder)
    #except:
    #    RuntimeError('Couldn''t run LAMMPS for the bulk references')
    # pull the energies into a dictionary    
    # this is manually changed for the atoms in the forcefield
    bulk_ene_dict = {}
    for i in ['Cu', 'Ni', 'O']:
        if i == 'Cu' or i == 'Ni':
            # all bulk dicts have 4 metal atoms
            proc = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), i+"-bulk", rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            proc.wait()
            if 'nan' in out.decode('utf-8'):
                return infeasible_penalty
            else:
                try:
                    bulk_ene_dict[i] = float(out.split(b'\n')[-2].decode('utf-8'))/4
                except:
                    print('This is the output ', out)
                    print('This is the split ', out.split(b'\n')[-2])
        elif i == 'O':
            # diatomics
            proc = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), i+"-bulk", rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            proc.wait()
            if 'nan' in out.decode('utf-8'):
                return infeasible_penalty
            else:
                try:
                    bulk_ene_dict[i] = float(out.split(b'\n')[-2].decode('utf-8'))/2
                except:
                    print('This is the output ', out)
                    print('This is the split ', out.split(b'\n')[-2])
        else:
            LookupError("Missing bulk reference data: you might need to execute run_bulk_lammps.bash first")
    # early exit if the forcefield can't even make bulk energies
    #print('bulk ene dict is ', bulk_ene_dict)
    if 'nan' in bulk_ene_dict.values():
	    print('Bulk values are NaN')
	    print(bulk_ene_dict)
	    return infeasible_penalty
    # loop over all the structures, running LAMMPS in a loop before referencing these values
    rand_dict = {}
    ene_dict = {}
    # pre-initializing some state variables for cleaner errors here
    #rand_dir = None
    #ene = None
    #rand_dir1 = None
    #ene1 = None
    #rand_dir2 = None
    #ene2 = None
    for trainstruct in job.referencedata_dict.values():
        #print(trainstruct[2])
        #print(type(trainstruct[2]))
        if trainstruct[0] == 'bulkeng':
            continue
        elif trainstruct[0] == 'fbond':
            continue
        elif trainstruct[0] == 'forceval':
            continue
        elif trainstruct[0] == 'energydiffeq':
            proc_one = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), trainstruct[2][0], rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out_one, err_one = proc_one.communicate()
            proc_one.wait()
            #print('This is the output', proc.stdout)
            proc_two = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), trainstruct[2][1], rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out_two, err_two = proc_two.communicate()
            proc_two.wait()
            if 'nan' in out_one.decode('utf-8'):
                return infeasible_penalty
            else:
                try:
                    ene_dict[trainstruct[2][0]] = float(out_one.split(b'\n')[-2].decode("utf-8"))
                except:
                    print('This is the structure \n', trainstruct[2][0])
                    print('This is the output \n', out_one)
                    print('This is the error \n', err_one)
                    print('this is the split \n', out_one.split(b'\n')[-2])
            if 'nan' in out_two.decode('utf-8'):
                return infeasible_penalty
            else:
                try:
                    ene_dict[trainstruct[2][1]] = float(out_two.split(b'\n')[-2].decode("utf-8")) 
                except:
                    print('This is the structure \n', trainstruct[2][1])
                    print('This is the output \n', out_two)
                    print('This is the error \n', err_two)
                    print('This is the split \n', out_two.split(b'\n')[-2])

        elif trainstruct[0] == 'coheng':
             proc = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), trainstruct[2][0], rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             out, err = proc.communicate()
             proc.wait()
             if 'nan' in out.decode('utf-8'):
                 return infeasible_penalty
             else:
                 try:
                     ene_dict[trainstruct[2][0]] = float(out.split(b'\n')[-2].decode("utf-8"))
                 except:
                    print('This is the structure \n', trainstruct[2])
                    print('This is the output \n', out)
                    print('This is the error \n', err)
                    print('This is the split \n', out.split(b'\n')[-2])

         #rand_dict[trainstruct[2][0]] = rand_dir1
            #rand_dict[trainstruct[2][1]] = rand_dir2
            #if is_non_zero_file(os.path.join('storage/','ene-'+rand_dir1)):
            #    try:
            #        ene1 = collect_ene(os.path.join('storage/', 'ene-' + rand_dir1))
            #    except:
            #        ValueError('file ene was not read properly ' + rand_dir1 + '/ene\n' + 'error for trainstruct ' + str(trainstruct[2][0]))
            #else:
            #    LookupError('file ene does not exist ', rand_dir + '/ene\n' + 'error for trainstruct ' + str(trainstruct[2][0]))
            #if is_non_zero_file(os.path.join('storage/', 'ene-'+rand_dir2)):
            #    try:
            #        ene2 = collect_ene(os.path.join('storage/', 'ene-' + rand_dir2))
            #    except:
            #        ValueError('file ene was not read properly ' +  rand_dir2 + '/ene\n' + 'error for trainstruct ' +str(trainstruct[2][1]))
            #else:
            #    LookupError('file ene does not exist ', rand_dir + '/ene\n' + 'error for trainstruct ' + str(trainstruct[2][1]))
                    
            #ene_dict[rand_dir1] = float(ene1)
            #ene_dict[rand_dir2] = float(ene2)
        # all other traintypes    
        else:
            if isinstance(trainstruct[2], str): 
                proc = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), trainstruct[2], rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = proc.communicate()
                proc.wait()
                if 'nan' in out.decode('utf-8'):
                    return infeasible_penalty
                else:
                    try:
                        ene_dict[trainstruct[2]] = float(out.split(b'\n')[-2].decode('utf-8'))
                    except:
                        print('This is the structure \n', trainstruct[2])
                        print('This is the output \n', out)
                        print('This is the error \n', err)
                        print('This is the split \n', out.split(b'\n')[-2])
            elif isinstance(trainstruct[2], list):
                proc = subprocess.Popen(["bash", os.path.join(PWD, "run_lammps.bash"), trainstruct[2][0], rand_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = proc.communicate()
                proc.wait()
                if 'nan' in out.decode('utf-8'):
                    return infeasible_penalty
                else:
                    try:
                        ene_dict[trainstruct[2][0]] = float(out.split(b'\n')[-2].decode("utf-8"))
                    except:
                        print('This is the structure \n', trainstruct[2])
                        print('This is the output \n', out)
                        print('This is the error \n', err)
                        print('This is the split \n', out.split(b'\n')[-2])
            #if is_non_zero_file(os.path.join('storage/','ene-'+rand_dir)):
            #    try:
            #        ene = collect_ene(os.path.join('storage/', 'ene-' + rand_dir))
            #        print(" energy %s %10.6f " % (rand_dir, ene))
            #    except:
            #        LookupError('file ene was not read properly ', rand_dir +'/ene\n' + 'error for trainstruct ' + str(trainstruct[2]))    
            #else:
            #    LookupError('file ene does not exist ', rand_dir + '/ene\n' + 'error for trainstruct ' + str(trainstruct[2]))

            #if isinstance(trainstruct[2], str):
            #    rand_dict[trainstruct[2]] = rand_dir
            #    ene_dict[rand_dir] = float(ene)
            #elif isinstance(trainstruct[2], list):
            #    rand_dict[trainstruct[2][0]] = rand_dir
            #    ene_dict[rand_dir] = float(ene)
            #else:
            #    TypeError('Traintype of ' + trainstruct[0] + 'was mishandled, inspect trainset file')

    
    if 'nan' in ene_dict.values():
	    print('One of the energy values was NaN')
	    print(ene_dict)
	    return infeasible_penalty

    errorsum_list = []
    for lineno, trainstruct in job.referencedata_dict.items():
        #print(trainstruct)
        # continue on if traintype = bulkeng since that is done already
        if trainstruct[0] == 'bulkeng':
            errorsum_list.append((bulk_ene_dict[trainstruct[2].split('-')[0]]- trainstruct[3])**(eng_exp_scale) * trainstruct[1] * alpha[0])
            continue
        # run lammps for the next structure
        # bringing in some of Matt's code
        varsign = 1
        if trainstruct[0] == 'coheng':
            equationsolve_list = [ene_dict[trainstruct[2][0]]]
            for snip in trainstruct[3]:
                if '*' in snip:
                    natoms = int(snip.split('*')[0])
                    atomtype = snip.split('*')[1]
                    equationsolve_list.append(natoms*bulk_ene_dict[atomtype]*varsign)
                elif snip == '-':
                    varsign = -1
                elif snip == '+':
                    varsign = 1
                else:
                    NotImplementedError("Unrecognized string in training input of type coheng on lineno " + str(lineno))

            equationsolve_scalar = sum(equationsolve_list)
            errorsum_list.append((equationsolve_scalar - trainstruct[4])**(eng_exp_scale) * trainstruct[1] * alpha[0])

        elif trainstruct[0] == 'surfeng':
            equationsolve_list = [ene_dict[trainstruct[2]]]
            for snip in trainstruct[3]:
                if '*' in snip:
                    natoms = int(snip.split('*')[0])
                    atomtype = snip.split('*')[1]
                    equationsolve_list.append(natoms*bulk_ene_dict[atomtype]*varsign)

                elif snip =='-':
                    varsign = -1
                elif snip == '+':
                    varsign = 1
                elif '2xArea' in snip:
                    equationsolve_list = [entry/float(snip.split(':')[1]) for entry in equationsolve_list]
                else:
                     ValueError("Unrecognized string in training input of type surfeng " + str(lineno))

            equationsolve_scalar = sum(equationsolve_list)
            errorsum_list.append((equationsolve_scalar - trainstruct[4])**(eng_exp_scale) * trainstruct[1] * alpha[0])

        elif trainstruct[0] == 'energydiffeq':
            equationsolve_list = []
            first = True
            for snip in trainstruct[3]:
                if len(snip) > 1:
                    #dividend = snip.split('/')[0]
                    #divisor = snip.split('/')[1]
                    if snip in job.referencedata_dict[lineno][2]:
                        try:
                            if first:
                                varsign = 1
                                equation_insert = ( ene_dict[trainstruct[2][0]] * varsign )
                                first = False
                            else:
                                equation_insert  = ( ene_dict[trainstruct[2][1]] * varsign )
                        except:
                            TypeError('Division during energydiffeq failed due to non-strings')

                        equationsolve_list.append( equation_insert )
                        #varsign = 1
                    else:
                        LookupError("Dividend directory entry not found for this energydiffeq: " + str(lineno) + '/n'
                                + 'Bad string: ' + snip +'\n'
                                + 'Bad traindata: ' + str(job.referencedata_dict[lineno]))
                elif snip == '-':
                    varsign = -1
                elif snip == '+':
                    varsign = 1
                else:
                    ValueError("Unrecognized reference file input: energydiffeq")

            equationsolve_scalar = sum(equationsolve_list)
            errorsum_list.append((equationsolve_scalar - trainstruct[4])**(eng_exp_scale) * trainstruct[1] * alpha[0])

        elif trainstruct[0] == 'forceval':
            if is_non_zero_file(os.path.join('storage/','forces-'+rand_dict[trainstruct[2]])):
                data = collect_dat(os.path.join('storage/','forces-'+rand_dict[trainstruct[2]]))
                atomindex = trainstruct[2].split('-')[-2]
                axisdir = trainstruct[2].split('-')[-1]
                axisindex = axesmap[axisdir]
                forceval = data[atomindex-1,3+axisindex]
                print("force %s %10.6f " %(rand_dict[trainstruct[2]], forceval))
                errorsum_list.append(( forceval - trainstruct[4])**(frc_exp_scale) * trainstruct[1] * alpha[1])
            else:
                LookupError('file for forceval does not exist ' + rand_dict[trainstruct[2]] +'/ene' + '\n' +'couldn''t locate file : ' + trainstruct[2] + ' on line ' + str(lineno))
        
        elif trainstruct[0] == 'fbond':
            if 'bulk' in trainstruct[2]:
                if is_non_zero_file(os.path.join('storage/', 'forces-' + trainstruct[2])):
                    data = collect_dat(os.path.join('storage/', 'forces-' + trainstruct[2]))
                else:
                    LookupError('file for fbond does not exist ', '/forces-' + trainstruct[2])
            else:
                if is_non_zero_file(os.path.join('storage/','forces-'+rand_dict[trainstruct[2]])):
                    data = collect_dat(os.path.join('storage/', 'forces-'+rand_dict[trainstruct[2]]))
                    atomindex1, atomindex2 = tuple(trainstruct[3])
                    simpos1 = data[atomindex1-1,1:4] 
                    simpos2 = data[atomindex2-1,1:4]
                    dist_12 = np.linalg.norm(simpos1 - simpos2)    
                    errorsum_list.append(( dist_12 - trainstruct[4])**(pos_exp_scale) * trainstruct[1] * alpha[2])
                else:
                    LookupError('file for fbond does not exist ','/forces-' + rand_dict[trainstruct[2]])

        elif trainstruct[0] == 'angle':
            if is_non_zero_file(os.path.join('storage/', 'forces-'+rand_dict[trainstruct[2]])):
                try:
                    data = collect_dat(os.path.join('storage/', 'forces-'+rand_dict[trainstruct[2]]))
                    atomindex1, atomindex2, atomindex3 = tuple(trainstruct[3])
                    simpos1 = data[atomindex1-1,1:4] 
                    simpos2 = data[atomindex2-1,1:4]
                    simpos3 = data[atomindex3-1,1:4]
                    dist_12 = simpos1 - simpos2
                    dist_32 = simpos3 - simpos2
                    cos_123 = np.dot(dist_12, dist_32) / ( np.linalg.norm(dist_12) * np.linalg.norm(dist_32) )
                    ang_123 = np.degrees(np.arccos(cos_123))
                except:
                    KeyError("Invalid angle term constructions") 
                errorsum_list.append((ang_123 - trainstruct[4])**(pos_exp_scale) * trainstruct[1] * alpha[2])

            else:
                LookupError('file for angle does not exist ','/forces-' + rand_dict[trainstruct[2]])

        else:
            NotImplementedError('Did you implement this training value?')
    
    errorsum = sum(errorsum_list)
    print(errorsum_list)
    #sys.stdout.flush()
    shutil.rmtree(os.path.join(PWD, rand_folder))
    return errorsum


# eventually need to loop over all training structures/data types and add a trainstruct input to objective_neg
lb = [job.params_Nminmax[key][0] for key in job.params_Nminmax.keys()]
ub = [job.params_Nminmax[key][1] for key in job.params_Nminmax.keys()]
print(lb)
print(ub)
xopt,fopt = pso(objective, lb , ub, swarmsize=options.S, omega=0.6, phip=0.8, phig=0.5, maxiter=options.repeat, minstep=1e-4, minfunc=10, debug=True, processes=options.p)
# setup and run pyswarm
# defined lb and ub lists using params_Nminmax from Matt's code
#xopt, fopt = pso(objective_neg, lb, ub,  swarmsize=options.S, omega=0.5, phip=0.5, phig=0.5, maxiter=options.repeat, minstep=1e-8,
 #                minfunc=1e-8, debug=True, processes=options.p)

resx = xopt


def print_duration(t_start, t_stop):
    'print the duration to screen'
    t_stop = time()
    dt = int(t_stop - t_start)
    h = dt // 3600
    m = (dt - h*3600) // 60
    s = dt % 60
    print('({:d}h {:d}min {:d}s)'.format(h, m, s))


end_time = datetime.now()
end_t = time()
print_duration(start_t, end_t)

print('Duration: {}'.format(end_time - start_time))
