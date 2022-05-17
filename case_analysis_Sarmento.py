###################################
# Step 1 || Load python libraries #
###################################
print('(1) Loading python libraries...')
# warning supresion
import warnings
warnings.filterwarnings("ignore")

# load numeric libraries  #
import numpy as np

# load graph libraries 
import matplotlib.pyplot as plt

# load chemometric libraries #
try:
	from analisys import MCR, PARAFAC, GPV
	from alig import FAPV32, COW
	from data import DATA
	from alig import FAPV21, ICOSHIFT
	from simulation import  SIMULATION
except: 
	print('WARNNING :: SIMULATION.py :: can NOT correctly load chemometric libraries !')
# Chemometrics and Intelligent Laboratory Systems

np.random.seed(12)
###################################
# Step 2 || Load data 			  #
###################################
print('(2) Load data...')
data = DATA()
path = '/home/akaris/Documents/code/Chemometrics/files/case_001/data/linear'
data.load_folther(path)
spectra = data.load_spectra('/home/akaris/Documents/code/Chemometrics/files/case_001/data/data_set/spectra.txt')

###################################
# Step 2 || Load data 			  #
###################################
'''
from functools import reduce

Na = data.Na
Nc = data.Nc
Z = data.S[0][:Na, :]
Y = np.array(reduce(lambda x, y:  np.concatenate( (x ,  y), axis=1), data.D[:,:,:] ))
Lc =  np.dot(Y.T, np.linalg.pinv(Z))
Lc= np.array([ Lc[n*391:(n+1)*391, :] for n in range(31)] )

D1 = np.tensordot( data.S[0][0, :], Lc[:, :, 0] , axes=0) 
D1[:,13:,:] = D1[:,13:,:]*1.02
D2 = np.tensordot( data.S[0][1, :], Lc[:, :, 1] , axes=0) 
D2[:,13:,:] = D2[:,13:,:]*0.70

D = np.rollaxis(D1+D2, 1)
data.D = D
data.X = D
data.export( './files/exp' )
'''

#####################################
# Step 2B || Pure vector estimation #
#####################################
print('(2) Pure vector estimation...')

###################################
# Step 3 || Aling  data			  #
###################################
print('(3) Aling  data...')
aling = 'FAPV' 
if aling == 'FAPV':
	fapv21 = FAPV21() 
	data.inject_data(fapv21)
	data.aling = fapv21
	Da, a, b = data.aling.aling( 	
								area='gaussian_coeficient', 		
								mu_range=np.arange(150, 300, 6), 	
								sigma_range=np.arange(4, 20, 0.7), 	
								non_negativity=False,				
								SD={'mode':'constant'},	
								interference_elimination=False, shape='gaussian'

								)									 	
	data.X = Da 
	data.D = Da

###################################
# Step 4 || Train model			  #
###################################
print('(4) Train model...')
model = PARAFAC()
data.inject_data(model)
data.model = model
s, Nloadings, model_mse = data.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']

###################################
# Step 5 || Predic with model	  #
###################################
print('(5) Predic with model...')
data.predic(v=0)#restriction = 'non-negative')
data.summary( )
data.model.plot_45()
data.model.plot_loadings()
data.model.plot_convergence()
plt.show()

###################################
# Step 6 || Save data 			  #
###################################
print('(6) Saving results...')
#data.export( './files/exp' )

