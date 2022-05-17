###################################
# Step 1 || Load python libraries #
###################################
print('(1) Loading python libraries...')
# warning supresion
import warnings
warnings.filterwarnings("ignore")

# load numeric libraries 
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

###################################
# Step 2 || Load data 			  #
###################################
data_OFL = DATA()
data_CPF = DATA()
factors = ['OFL', 'CPF']
# ******** LOAD EXPERIMENTAL SPECTRAS DATA ******** #
espectra_OFlm = 'files/case_000/Experimental/OFL1Bm.txt'
espectra_OFlx = 'files/case_000/Experimental/OFL1Bx.txt'

espectra_CIP1Am = 'files/case_000/Experimental/CIP1Am.txt'
espectra_CIP1Ax = 'files/case_000/Experimental/CIP1Ax.txt'

espectra_DAN1Am = 'files/case_000/Experimental/DAN1Am.txt'
espectra_DAN1Ax = 'files/case_000/Experimental/DAN1Ax.txt'

espectra_ENO1Bm = 'files/case_000/Experimental/ENO1Bm.txt'
espectra_ENO1Bx = 'files/case_000/Experimental/ENO1Bx.txt'

espectra_MAR1Bm = 'files/case_000/Experimental/MAR1Bm.txt'
espectra_MAR1Bx = 'files/case_000/Experimental/MAR1Bx.txt'

espectra_OFlm = np.loadtxt(espectra_OFlm)
espectra_OFlx = np.loadtxt(espectra_OFlx)


all_espectras = [	espectra_OFlm, espectra_OFlx, 
					espectra_CIP1Am, espectra_CIP1Ax,
					espectra_DAN1Am, espectra_DAN1Ax, 
					espectra_ENO1Bm, espectra_ENO1Bx,
					espectra_MAR1Bm, espectra_MAR1Bx]


for n in all_espectras:
	plt.plot( n[:,1] )
plt.show()
error

print('(2a) Loading {} concentrations...'.format( ', '.join(factors) ) )
# ******** LOAD OFL DATA ******** #
print(' **** loading OFL ')
Ycall_file_OFl = 'files/case_000/data/Original/OLF/Y' # file path
Xcall_file_OFl = 'files/case_000/OFL.npy' # file path
data_OFL.load_Yfrom_file(Ycall_file_OFl)
data_OFL.load_Xfrom_npfile(Xcall_file_OFl)
data_OFL.transpose_order_channels( channels_info = [0,1,-1,1] )

# - data information - #
data_OFL.Ni = 2
data_OFL.Na = 1
data_OFL.Nc = 5
data_OFL.N = 5+7+8 
data_OFL.Nt = 7+8
data_OFL.a = 1

# ******** LOAD CPF DATA ******** #
Ycall_file_CPF = 'files/case_000/Y_CPF.npy' # file path
Ycall_file_CPF = 'files/case_000/data/Original/CPF/Y' # file path
Xcall_file_CPF = 'files/case_000/CPF.npy' # file path
data_CPF.load_Yfrom_file(Ycall_file_CPF)
data_CPF.load_Xfrom_npfile(Xcall_file_CPF)
data_CPF.transpose_order_channels( channels_info = [0,1,-1,1] )

# - data information - #
data_CPF.Ni = 2 # variable
data_CPF.Na = 1 
data_CPF.Nc = 6
data_CPF.N = 6+7+8 
data_CPF.Nt = 7+8
data_CPF.a = 1

#####################################
# Step 2B || Pure vector estimation #
#####################################
pure_vectors_source = 'load'
if pure_vectors_source == 'estimate': 
	print('(2b) Estimating pure vector...')
	pure_vectors = []
	
	# ******** CPF pure vector ******** #
	model_CPF = PARAFAC()
	data_CPF.inject_data(model_CPF)
	#model.compress_3WAY()
	data_CPF.model = model_CPF
	data_CPF.model.X = data_CPF.model.X[:6,:,:,:]
	s, Nloadings, model_mse = data_CPF.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']

	pure_vectors.append( Nloadings[1] )
	pure_vectors.append( Nloadings[2] )
	pure_vectors.append( Nloadings[3] )

	# ******** OLF pure vector ******** #
	model_OFL = PARAFAC()
	data_OFL.inject_data(model_OFL)
	data_OFL.model = model_OFL
	data_OFL.model.X = data_OFL.model.X[:5,:,:,:]

	data_OFL.model = model_OFL
	s, Nloadings, model_mse = data_OFL.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']

	# ******** SAVE DATA ******** #
	pure_vectors[0] = np.concatenate( (pure_vectors[0], Nloadings[1]), axis=0 )
	pure_vectors[1] = np.concatenate( (pure_vectors[1], Nloadings[2]), axis=0 )
	pure_vectors[2] = np.concatenate( (pure_vectors[2], Nloadings[3]), axis=0 )

	np.savetxt( fname='pure_vectors_mode1.dat', X=pure_vectors[0] )
	np.savetxt( fname='pure_vectors_mode2.dat', X=pure_vectors[1] )
	np.savetxt( fname='pure_vectors_mode3.dat', X=pure_vectors[2] )

	# ******** PLOT DATA ******** #
	#plt.figure(1), plt.plot(pure_vectors[0].T) # not necessary plot 
	#plt.figure(2), plt.plot(pure_vectors[1].T) # not necessary plot 
	#plt.figure(3), plt.plot(pure_vectors[2].T) # not necessary plot 
	#plt.show()

elif pure_vectors_source == 'load':
	print('(2b) Loading pure vectors...')

	# ******** LOAD pure vectors ******** #
	pure_vectors_mode1 = np.loadtxt( 'files/case_000/data/pure_vectors/2Na2Ni/B_trainningset/pure_vectors_mode1.dat' )
	pure_vectors_mode2 = np.loadtxt( 'files/case_000/data/pure_vectors/2Na2Ni/B_trainningset/pure_vectors_mode2.dat' )
	pure_vectors_mode3 = np.loadtxt( 'files/case_000/data/pure_vectors/2Na2Ni/B_trainningset/pure_vectors_mode3.dat' )

	data_CPF.S = [ pure_vectors_mode1[:2,:], pure_vectors_mode2[:2,:], pure_vectors_mode3[:2,:]   ]
	data_OFL.S = [ pure_vectors_mode1[2:,:], pure_vectors_mode2[2:,:], pure_vectors_mode3[2:,:]   ]
	# ******** PLOT pure vectors ******** #
	#plt.figure(1), plt.plot(pure_vectors_mode1.T)
	#plt.figure(2), plt.plot(pure_vectors_mode2.T)
	#plt.figure(3), plt.plot(pure_vectors_mode3.T)
	#plt.show()


###################################
# Step 3 || Aling  data			  #
###################################
print('(3) Data alignement...')
model_aling = 'FAPV'
if model_aling == 'FAPV': #parafac
	# ******** CPF FAPV analysis ******** #
	plt.plot(pure_vectors_mode1[:,:].T)
	plt.show()
	FAPV_CPF = FAPV32()
	data_CPF.inject_data(FAPV_CPF)
	FAPV_CPF.S = [ pure_vectors_mode1[:2,:], pure_vectors_mode2[:2,:], pure_vectors_mode3[:2,:]   ]
	Da, a, b = FAPV_CPF.aling( )

	data_CPF.D = Da
	data_CPF.X = Da

	print(data_CPF.D.shape)
	
error

###################################
# Step 4 || Train model			  #
###################################
print('(4) Trainning model...')
model = 'MCR'

if model == 'parafac': #parafac
	# ******** CPF PARAFAC analysis ******** #
	model_CPF = PARAFAC()
	data_CPF.inject_data(model_CPF)
	data_CPF.model = model_CPF

	CPF_s, CPF_Nloadings, CPF_model_mse = data_CPF.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']

	# ******** OLF PARAFAC analysis ******** #
	model_OFL = PARAFAC()
	data_OFL.inject_data(model_OFL)
	data_OFL.model = model_OFL

	OLF_s, OLF_Nloadings, OLF_model_mse = data_OFL.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']


if model == 'MCR': #parafac
	# ******** CPF PARAFAC analysis ******** #
	model_CPF = MCR()
	data_CPF.inject_data(model_CPF)
	model_CPF.compress_3WAY()
	data_CPF.model = model_CPF

	CPF_S, CPF_C = data_CPF.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']

	# ******** OLF PARAFAC analysis ******** #
	model_OFL = MCR()
	data_OFL.inject_data(model_OFL)
	model_OFL.compress_3WAY()
	data_OFL.model = model_OFL

	OFL_S, OFL_C = data_OFL.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']


###################################
# Step 5 || Predic with model	  #
###################################
print('(5) Predic with model...')

if model == 'parafac': #parafac
	# ******** CPF PARAFAC analysis ******** #
	# Predic CPF
	data_CPF.predic(v=0)#restriction = 'non-negative')
	data_CPF.summary( )

	# ******** OLF PARAFAC analysis ******** #
	# Predic data_OFL
	data_OFL.predic(v=0)#restriction = 'non-negative')
	data_OFL.summary( )

if model == 'MCR': #MCR
	# ******** CPF PARAFAC analysis ******** #
	# Predic CPF
	data_CPF.predic(v=0)#restriction = 'non-negative')
	data_CPF.summary( )

	# ******** OLF PARAFAC analysis ******** #
	# Predic data_OFL
	data_OFL.predic(v=0)#restriction = 'non-negative')
	data_OFL.summary( )

###################################
# Step 6 || Save data 			  #
###################################
print('(6) Saving results...')
# eg.
#path = './cases/test/savetest/'
#for i, n in enumerate(Da):
#	np.savetxt(fname='{}sample{}.txt'.format(path, i) , X=Da[i,:],)
#print(Da.shape) # conteins all the data. So you can save Da
#print('...')



