import numpy as np

# *** warning supresion *** #
import warnings
warnings.filterwarnings("ignore")
try:
	import os, functools, pickle
except:
	print('WARNNING :: SIMULATION.py :: can NOT import os. ( needed for SIMULATION.recursive_read() )')

import numpy as np

try:
	import matplotlib.pyplot as plt
except: 
	print('WARNNING :: SIMULATION.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: "pip3 install matplotlib" )')
	
import itertools, copy

# *** chemometric libraries *** #
try:
	from analisys import MCR, PARAFAC, GPV
	from alig import FAPV32, COW
	from data import DATA
	from alig import FAPV21, ICOSHIFT
	from simulation import  SIMULATION
except: 
	print('WARNNING :: SIMULATION.py :: can NOT correctly load chemometric libraries !')

import pandas as pd
df_new = pd.DataFrame()
for i, df in enumerate(pd.read_csv( 'dataframe.csv', chunksize=1000000 )):
	print(i)


# ---------------------------- NOISE ---------------------------- #
# -- make SIMULATION obj -- #
simulation = SIMULATION()
# -- Define path to data -- #
data_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/PARAFAC2_0' # PARAFAC 2
data_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/noise00' # PARAFAC 2
data_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/overlap01' # PARAFAC 2
data_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/shift02' # PARAFAC 2
data_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/MCR_ICO_00' # PARAFAC 2
# -- read alll results -- #
results = simulation.recursive_read(path=data_path, ) #save={'mode':'pkl','path':data_path, 'name':'data'})
# -- summary  results -- #
simulation.simulation_result_summary(data=results['data_dict'], hcs=results['hyperparameters_configuration_space'])


# -- Load results pickle -- #
#pickle_file = open('{}/data.pkl'.format(data_path), "rb")
#data_dict = pickle.load(pickle_file)
data_dict = results['data_dict']

# -- PLOT  results -- #
for n in data_dict:
	print(n)

color1 = [ 	[float(n)/3+0.35,0,0] for n in range(4)  ]
color1 = [ 	[float(n)/55+0.15,0,0] for n in range(45)  ]
color2 =[ 	[0, 0, float(n)/3+0.35] for n in range(4)  ]
color2 =[ 	[0, 0, float(n)/55+0.15] for n in range(45)  ]

noise = ['9random0.0001', 	'9random7.5e-05', '9random5e-05', '9random2.5e-05', '9random1e-05', 
							'9random7.5e-06', '9random5e-06', '9random2.5e-06', '9random1e-06',
							'9random7.5e-07', '9random5e-07', '9random2.5e-07', '9random1e-07']
overlap = ['0', '05', '1', '15', '2', '25', '3', '35', '4', '45', '5', '55', '6', '65', '7', '75', '8', '85', '9', '95', ]
overlap2 = ['0.0', '0.03333333333333333', '0.06666666666666667', '0.1', '0.13333333333333333', '0.16666666666666666', 
			'0.2', '0.23333333333333334', '0.26666666666666666', '0.3', '0.3333333333333333', '0.36666666666666664', 
			'0.4', '0.43333333333333335', '0.4666666666666667',  '0.5', '0.5333333333333333',  '0.5666666666666667',
			'0.6', '0.6333333333333333',  '0.6666666666666666',  '0.7', '0.7333333333333333','0.7666666666666667', 
			'0.8', '0.8333333333333334',  '0.8666666666666667',  '0.9', '0.9333333333333333', '0.9666666666666667']

shift = ['0.0', '0.03333333333333333','0.06666666666666667', '0.1', '0.13333333333333333', '0.16666666666666666','0.2', '0.23333333333333334', '0.26666666666666666',
		 '0.3',	'0.3333333333333333', '0.36666666666666664', '0.4', '0.43333333333333335', '0.4666666666666667', '0.5', '0.5333333333333333',  '0.5666666666666667',
		 '0.6', '0.6333333333333333', '0.6666666666666666',  '0.7', '0.7333333333333333',  '0.7666666666666667', '0.8', '0.8333333333333334',  '0.8666666666666667',
		 '0.9', '0.9333333333333333', '0.9666666666666667']

#shift = ['0.4', '0.8', ]

for i, n in enumerate(overlap2):
	ref=n
	#parafac2
	#data1 = data_dict['PARAFACNone3[60,60,70][15,0,30]302Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[0.{0},0.{0}]]30.7random1e-06'.format(ref)]['mean_error']
	#parafac noise
	#data1 = data_dict['MCRNone3[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[0.6,0.6]]30.{}'.format(ref)]['mean_error']
	#data1 = data_dict['PARAFACNone3[60,60,70][15,0,30]302Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[{0},{0}]]30.7random1e-06'.format(ref)]['mean_error']
	data1 = data_dict['PARAFACNone3[60,60,70][15,0,30]302Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[{0},{0}]]30.0random1e-06'.format(ref)]['mean_error']
	data1 = np.array(data1)
	data1 = data1[data1[:,1]<2, 1] 
	print(color1[i], color2[i])
	plt.figure(2), plt.hist( data1, bins=500, color=color1[i], alpha=0.8 )

	#parafac2
	#data2 = data_dict['PARAFACFAPV323[60,60,70][15,0,30]302Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[0.{0},0.{0}]]30.7random1e-06'.format(ref)]['mean_error']
	#parafac noise
	#data2 = data_dict['MCRFAPV323[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[0.6,0.6]]30.{}'.format(ref)]['mean_error']
	#data2 = data_dict['PARAFACFAPV323[60,60,70][15,0,30]302Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[{0},{0}]]30.7random1e-06'.format(ref)]['mean_error']
	data2 = data_dict['PARAFACICOSHIFT3[60,60,70][15,0,30]302Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[{0},{0}]]30.0random1e-06'.format(ref)]['mean_error']
	data2 = np.array(data2)
	data2 = data2[data2[:,1]<2, 1] 
	plt.figure(1), plt.hist( data2, bins=500, color=color2[i], alpha=0.8 )

	plt.figure(3)
	plt.hist( data1, bins=200, color=color1[i], alpha=0.8 )
	plt.hist( data2, bins=200, color=color2[i], alpha=0.8 )

plt.figure(2)
plt.title('MCR - overlap')
plt.xlabel('NRMSD')

plt.figure(1)
plt.title('FAPV32 + MCR  - overlap')
plt.xlabel('NRMSD')

plt.show()
erorror

# -- summary  results -- #

simulation.simulation_result_summary(data=data_dict)

error

#print(results['data_dict'])

data_dict = pickle.load(pickle_file)
print( data_dict.keys() )

if 1 == 1:
	for n in range( 10 ):
	#for i, n in enumerate(['025','05','075','1','125','15','175','2','225','25','275','3','325','35','375','4','425','45','475']): # 3
	#for i, n in enumerate(['1','2','3','4',]): # 3
		lims=[0.019, 0.033, 0.041, 0.061, 0.080, 0.10, 0.12, 0.14, 0.17, 0.18, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24]
		error = ['0random0.001', '0random0.0001', '0random1e-05']
		marks = [".", "o", "v", "s", "p", "P", "*", ".", ]
		m = error[1]

		name1 = 'MCRFAPV323[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[1.2,1.2]]30.8random1e-06'
		name1 = 'PARAFACFAPV323[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[{0:.1f},{0:.1f}]]30.9random1e-06'.format(0.8+float(9-n)/10)
		name2 = 'MCRNone3[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[{0:.1f},{0:.1f}]]30.9random1e-06'.format(0.8+float(9-n)/10)

		name1 = 'PARAFACFAPV323[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[0.8,0.8]]3{0:.1f}random1e-06'.format(float(n)/10)
		name2 = 'MCRNone3[60,60,70][15,0,30]332Gaussian[[0.0,0.0,1],[0.0,0.0,1],[0.0,0.0,5]][[0,0],[0.0,0.0],[0.8,0.8]]3{0:.1f}random1e-06'.format(float(n)/10 )


		label1 = 'FAPV32 + PARAFAC'
		label2 = 'MCR'

		estat = 'RMSD_error'
		data11 = np.array(data_dict[name1][estat] )
		data12 = np.array(data_dict[name2][estat] )

		print('pre', data11.shape)
		print('pre', data12.shape)

		data11 = data11[:np.min([data11.shape[0], data12.shape[0]]),0]
		data12 = data12[:np.min([data11.shape[0], data12.shape[0]]),0]
		#data11 += 0.0014 

		data11 = data11[ data12<1.5 ]
		data12 = data12[ data12<1.5 ]

		data12 = data12[ data11<1.5 ]
		data11 = data11[ data11<1.5 ]
		print(data11.shape)
		print(data12.shape)
		#data11 = data11[ data12<lims[i] ]
		#data12 = data12[ data12<lims[i] ]

		plt.figure(1), plt.plot( data11, data12, 'o', ms=0.5 )
		plt.xlabel('{} - NRMSD'.format(label1) )
		plt.ylabel('{} - NRMSD'.format(label2) )
		plt.figure(2), plt.hist( data11, bins=200, weights=np.ones_like(data11)/len(data11) )
		plt.title( str(label1) )
		plt.xlabel('{} - NRMSD'.format(label1))
		plt.ylabel('Freq rel - NRMSD')

		plt.figure(3), plt.hist( data12, bins=200, weights=np.ones_like(data11)/len(data11) )
		plt.title(str(label2))
		plt.xlabel('{} - NRMSD'.format(label2))
		plt.ylabel('Freq rel - NRMSD')
		
	plt.show()


