# *** warning supresion *** #
import warnings
warnings.filterwarnings("ignore")

try:
	if not (sys.version_info.major == 3 and sys.version_info.minor >= 0):
	    print("To use this script we recomends Python 3.x or higher!")
	    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    #sys.exit(1)
except: 
	pass

# *** numeric libraries *** #
import sys, os, ctypes 
def set_mkl_threads( num_threads=1 ):
	'''
	(*) np.convolve running on max threads with numpy >= 1.14.6 #13888

	There are more than the 3 mentioned environmental variables.
	 The followings are the complete list of environmental variables and the package that 
	 uses that variable to control the number of threads it spawns. Note than you need to 
	 set these variables before doing import numpy:
	
	OMP_NUM_THREADS: openmp,
	OPENBLAS_NUM_THREADS: openblas,
	MKL_NUM_THREADS: mkl,
	VECLIB_MAXIMUM_THREADS: accelerate,
	NUMEXPR_NUM_THREADS: numexpr
	
	We will probably use this code as a solution for now. It is hard to test this because 
	you never know how are people/distributions compiling numpy. But this code I believe 
	should work at least for MKL.

	Actually, this is being set not when you import numpy, but when you first invoke 
	MKL-optimized Numpy code. For example, np.ones() does not trigger reading these 
	variables, but np.dot() does
	'''
	try: 
		import mkl
		mkl.set_num_threads(num_threads)
		return 0
	except:
		pass 
    
	for name in [ "libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]: 
		try: 
			mkl_rt = ctypes.CDLL(name)
			mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))
			return 0 
		except:
			pass   

	os.environ["OMP_NUM_THREADS"] = str(num_threads) # export OMP_NUM_THREADS=4
	os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads) # export OPENBLAS_NUM_THREADS=4 
	os.environ["MKL_NUM_THREADS"] = str(num_threads) # export MKL_NUM_THREADS=6
	os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads) # export VECLIB_MAXIMUM_THREADS=4
	os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads) # export NUMEXPR_NUM_THREADS=6
	os.system("export OMP_NUM_THREADS={0}".format(str(num_threads)) )

set_mkl_threads( num_threads=1 ) # set max threads
import numpy as np # import numpy after set max threads

try:	
	from sklearn.datasets import load_iris
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression
except: print('WARNNING :: main_simulation.py :: can NOT correctly load "sklearn" libraries')

# *** chemometric libraries *** #
from analisys import MCR, PARAFAC, GPV
from alig import FAPV32, COW
from data import DATA
from alig import FAPV21, ICOSHIFT
from simulation import GENERATOR
from simulation import SIMULATION

# *** graph libraries *** #
try:
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pl
except: 
	print('WARNNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: "pip3 install matplotlib"	')
	
# *** python libraries *** #
import time, itertools

def piluso_queue(n=1):
	pass

def run_(data_generation, hyperparameters, aling_models, trainnig_models, save_path, repeat, ):
	# make simulation obj #
	simulation = SIMULATION(verbose=1)
	# make data conteiner #
	simulation.data = DATA()
	# make  data set generator obj #
	simulation.generator = data_generation
	# select alination models #
	simulation.model_aling = aling_models
	# select processing models #  
	simulation.model_trainning = trainnig_models
	# choose save path #
	simulation.save_path = save_path
	# number of repetitions of each hyper parameter # 
	simulation.repeat = repeat
	# start hyper paremeters esploration #

	simulation.exploration( hyperparameters )

# ------ Data simulation ------- #
# choose hyper parameters #
dic_hp = {
		# ----------------- volume ----------------- #
		'ways' 			:	[3]								,	# number of ways # 				eg: ways=3
		'dimentions' 	:	[[60, 90, 100]] 					,	# dimentionality of each way # 	eg: dimentions=[50,60,40]
		'N' 			:	[[15, 0,  30]] 						,	# [trainnig samples, validation samples, test samples] # eg: N=[6,7,5]
		# ----------------- Compose ----------------- #dis
		'factors' 		:	[3] 							,	# number of calibrated factors #		eg:  factors=3
		'interferentes' :	[0] 							,	# number of interferents #				eg:  interferentes=0 
		'aligned' 		:	[2] 							,	# number spected aligned channels #		eg:  aligned=2 
		# ----------------- complexity ----------------- #
		'functional' 	:	['Gaussian'] 					,	# functional basee form of pure vectors # eg: functional='Gaussian'
		'deformations'	:	[	[								# deformations # 	eg: deformations=[[mu, sd, Gnum], [channel_2], [channel_3]]		
								[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]],
								] 	,							# deformations # 	eg: deformations=[[mu, sd, Gnum], [channel_2], [channel_3]]			
		'warping' 	 	:	[ 									# warping # 		eg: warping=[[0.0,0.0],[0,0],[0,0]] 
								[[0,0],[0.0,0.0],[float(n+0)/10,float(n+0)/10]] for n in range(10) ]			,	# warping # 					eg: warping=[[0.0,0.0],[0,0],[0,0]] 
		'sd' 			:	[[3,3,3]]								,	# SD of pure vec1tors # 			eg: sd=2
		'overlaping' 	:	[[0.00,0.00,0.00]]								,	# overlapping of pure vectors # eg: overlaping=0
		# ----------------- noise ----------------- #
		'noise_type' 	:	['random']						,	# noise structure #				eg: noise_type='random'
		'intencity' 	:	[0.000001, 0.0001, 0.00001]						,	# noise intensity # 			eg: intencity=0.001
		}


# --- make  data set generator obj --- #
generator = GENERATOR.SAMPLE()

# --- select alination models --- #
# eg.  model_aling = [{'model':None}, {'model': 'ICOSHIFT' }, {'model': 'FAPV32', 'init':'self-vector'},   {'model':'MCR-ICOSHIFT'}, ]
model_aling = [{'model':'MCR-ICOSHIFT'}  ]

# --- select processing models --- #  
model_trainning = [ 'PARAFAC', 'MCR',]

# --- choose save path --- #
# (0) eg.  save_path = str(sys.argv[1]) 
# (1) eg. 'save_path = cases/files/simulation01'
if type(sys.argv) == list and len(sys.argv) > 1:
	save_path = str(sys.argv[1]) 
else:
 	save_path = 'files/simulations' 

# --- number of repetitions of each hyper parameter --- # 
repeat = 2

try:
	run_(	data_generation	=	generator, 
			hyperparameters	=	dic_hp, 
			aling_models	=	model_aling,
			trainnig_models	=	model_trainning, 
			save_path		=	save_path, 
			repeat			=	repeat, )
except OSError as err:
    print("OS error: {0}".format(err))
#except ValueError:
#Could not convert data to an integer    print("Could not convert data to an integer.")
except:
    print('ERROR :: main_simulation() :: can NOT run simulation')
    print("Unexpected error:", sys.exc_info()[0])
    raise


# REV next version.
# (0)	sd must be a vector
# (1)	color noise per way
# (2) 	functional per way

