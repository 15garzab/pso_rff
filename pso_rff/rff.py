import os
import numpy as np


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
