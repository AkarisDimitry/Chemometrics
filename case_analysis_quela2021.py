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
	from analisys import MCR, PARAFAC, GPV, COMPLEXITY
	from alig import FAPV32, COW
	from data import DATA
	from alig import FAPV21, ICOSHIFT, MCR_ICOSHIFT
	from simulation import  SIMULATION
except: 
	print('WARNNING :: SIMULATION.py :: can NOT correctly load chemometric libraries !')

def load():
	# Chemometrics and Intelligent Laboratory Systems
	###################################
	# Step 2 || Load data 			  #
	###################################
	print('(2a) Loading spectra...')
	data_OFL = DATA()
	data_CPF = DATA()
	factors = ['OFL', 'CPF']
	# ******** LOAD EXPERIMENTAL SPECTRAS DATA ******** #
	espectra_OFlm = 'files/case_000/ExperimentalInput/experimental spectra/OFL1Bm.txt'
	espectra_OFlx = 'files/case_000/ExperimentalInput/experimental spectra/OFL1Bx.txt'
	espectra_OFlm = np.loadtxt(espectra_OFlm)
	espectra_OFlx = np.loadtxt(espectra_OFlx)

	espectra_CIP1Am = 'files/case_000/ExperimentalInput/experimental spectra/CIP1Am.txt'
	espectra_CIP1Ax = 'files/case_000/ExperimentalInput/experimental spectra/CIP1Ax.txt'
	espectra_CIP1Am = np.loadtxt(espectra_CIP1Am)
	espectra_CIP1Ax = np.loadtxt(espectra_CIP1Ax)

	espectra_DAN1Am = 'files/case_000/ExperimentalInput/experimental spectra/DAN1Am.txt'
	espectra_DAN1Ax = 'files/case_000/ExperimentalInput/experimental spectra/DAN1Ax.txt'
	espectra_DAN1Am = np.loadtxt(espectra_DAN1Am)
	espectra_DAN1Ax = np.loadtxt(espectra_DAN1Ax)

	espectra_ENO1Bm = 'files/case_000/ExperimentalInput/experimental spectra/ENO1Bm.txt'
	espectra_ENO1Bx = 'files/case_000/ExperimentalInput/experimental spectra/ENO1Bx.txt'
	espectra_ENO1Bm = np.loadtxt(espectra_ENO1Bm)
	espectra_ENO1Bx = np.loadtxt(espectra_ENO1Bx)

	espectra_MAR1Bm = 'files/case_000/ExperimentalInput/experimental spectra/MAR1Bm.txt'
	espectra_MAR1Bx = 'files/case_000/ExperimentalInput/experimental spectra/MAR1Bx.txt'
	espectra_MAR1Bm = np.loadtxt(espectra_MAR1Bm)
	espectra_MAR1Bx = np.loadtxt(espectra_MAR1Bx)



	all_espectras = [	espectra_OFlm, espectra_OFlx, 
						espectra_CIP1Am, espectra_CIP1Ax,
						espectra_DAN1Am, espectra_DAN1Ax, 
						espectra_ENO1Bm, espectra_ENO1Bx,
						espectra_MAR1Bm, espectra_MAR1Bx]

	# ******** PLOT DATA ******** #
	#for n in all_espectras:	plt.plot( n[:,0], n[:,1] )
	#plt.xlabel('longuitud de onda')
	#plt.ylabel('Signal')
	#plt.title('espectra')
	#plt.show()

	# ******** LOAD concentration ******** #
	print('(2b) Loading {} concentrations...'.format( ', '.join(factors) ) )
	# ******** LOAD OFL DATA ******** #
	print(' **** loading OFL ')
	Ycall_file_OFl = 'files/case_000/ExperimentalInput/ResponceValues/OLF/Y' # file path
	Xcall_file_OFl = 'files/case_000/OFL.npy' # file path
	data_OFL.load_Yfrom_file(Ycall_file_OFl)
	data_OFL.load_Xfrom_npfile(Xcall_file_OFl)
	data_OFL.transpose_order_channels( channels_info = [0,1,-1,1] )


	# - data information - #
	data_OFL.Ni = 1
	data_OFL.Na = 1
	data_OFL.Nc = 5
	data_OFL.N = 5+7+8 
	data_OFL.Nt = 7+8
	data_OFL.a = 2

	# ******** LOAD CPF DATA ******** #
	Ycall_file_CPF = 'files/case_000/Y_CPF.npy' # file path
	Ycall_file_CPF = 'files/case_000/ExperimentalInput/ResponceValues/CPF/Y' # file path
	Xcall_file_CPF = 'files/case_000/CPF.npy' # file path
	data_CPF.load_Yfrom_file(Ycall_file_CPF)
	data_CPF.load_Xfrom_npfile(Xcall_file_CPF)
	data_CPF.transpose_order_channels( channels_info = [0,1,-1,1] )

	# - data information - #
	data_CPF.Ni = 1 # variable
	data_CPF.Na = 1 
	data_CPF.Nc = 6
	data_CPF.N = 6+7+8 
	data_CPF.Nt = 7+8
	data_CPF.a = 2

	return data_OFL, data_CPF

def estimation(data_OFL, data_CPF):
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
		pure_vectors_OFL = [ Nloadings[1], Nloadings[2], Nloadings[3]]

		#np.savetxt( fname='pure_vectors_mode1.dat', X=pure_vectors[0] )
		#np.savetxt( fname='pure_vectors_mode2.dat', X=pure_vectors[1] )
		#np.savetxt( fname='pure_vectors_mode3.dat', X=pure_vectors[2] )

		pure_vectors_mode1 = pure_vectors[0]
		pure_vectors_mode2 = pure_vectors[1]
		pure_vectors_mode3 = pure_vectors[2]
		
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
		data_OFL.S = [ pure_vectors_mode1[1:,:], pure_vectors_mode2[1:,:], pure_vectors_mode3[1:,:]   ]
		# ******** PLOT pure vectors ******** #
		#plt.figure(1), plt.plot(pure_vectors_mode1.T)
		#plt.figure(2), plt.plot(pure_vectors_mode2.T)
		#plt.figure(3), plt.plot(pure_vectors_mode3.T)
		#plt.show()

	return data_OFL, data_CPF, [pure_vectors_mode1, pure_vectors_mode2, pure_vectors_mode3]

def aling(data_OFL, data_CPF, pure_vectors_mode, model_aling='FAPV'):
	###################################
	# Step 3 || Aling  data			  #
	###################################
	print('(3) Data alignement...')
	model_aling = model_aling # eg.'FAPV'

	if model_aling == 'ICOSHIFT': 
		# ******** CPF FAPV analysis ******** #
		MCR_ICOSHIFT_CPF = MCR_ICOSHIFT()
		data_CPF.inject_data(MCR_ICOSHIFT_CPF)
		data_CPF.aling = MCR_ICOSHIFT()
		Da, a, b = MCR_ICOSHIFT_CPF.aling()
		data_CPF.D, data_CPF.X = Da, Da

		# ******** OFL FAPV analysis ******** #
		MCR_ICOSHIFT_OFL = MCR_ICOSHIFT()
		data_OFL.inject_data(MCR_ICOSHIFT_OFL)
		data_OFL.aling = MCR_ICOSHIFT()
		Da, a, b = MCR_ICOSHIFT_OFL.aling()
		data_OFL.D, data_OFL.X = Da, Da
		

	if model_aling == 'FAPV' or model_aling == 'FAPV32': 
		# ******** CPF FAPV analysis ******** #
		global FAPV_CPF
		FAPV_CPF = FAPV32()
		data_CPF.inject_data(FAPV_CPF)
		data_CPF.aling = FAPV32()

		FAPV_CPF.S = [ pure_vectors_mode[0][:2,:], pure_vectors_mode[2][:2,:], pure_vectors_mode[1][:2,:]  ]

		# --- BEST RESULT --- #
		Da, a, b = FAPV_CPF.aling(area={'mode':'quela', 'mu_range':np.arange(153, 156,0.1) , 
										'sigma_range':np.arange(0.5, 5, 0.4), 'test shift':0, 'functional':'gaussian'}, 
						shape='original', non_negativity=True, 
						linear_adjust=True, SD={'mode':'constant'},	interference_elimination=False, 
						#include_diff = {'mode': 'direc diference', 'components':'test', 'afects':'test'},
						save=True, v=1 )

		data_CPF.D = Da
		data_CPF.X = Da
		
		# ******** OFL FAPV analysis ******** #
		global FAPV_OFL
		FAPV_OFL = FAPV32()
		data_OFL.inject_data(FAPV_OFL)
		data_OFL.aling = FAPV32()
		
		pure_vectors_mode[0] = np.roll(pure_vectors_mode[0][:,:], 1, axis=0)
		pure_vectors_mode[1] = np.roll(pure_vectors_mode[1][:,:], 1, axis=0)
		pure_vectors_mode[2] = np.roll(pure_vectors_mode[2][:,:], 1, axis=0)
		FAPV_OFL.D[5:,:,:,:] *= 0.75
		FAPV_OFL.S = [ pure_vectors_mode[0], pure_vectors_mode[2], pure_vectors_mode[1]  ]
		Da, a, b = FAPV_OFL.aling(area={'mode':'quela', 'mu_range':np.arange(125, 142, 0.3) , 
										'sigma_range':np.arange(2, 6, 0.3), 'test shift':16, 'functional':'gaussian'}, 
						shape='original', non_negativity=True, 
						linear_adjust=True, SD={'mode':'constant'},	interference_elimination=False,  
						#include_diff = {'mode': 'direc diference', 'components':'test', 'afects':'test'},
						save=True, v=1 )

		data_OFL.D = Da
		data_OFL.X = Da
		data_OFL.Ni = 0

		#FAPV_CPF.complexity_analysis()
		#FAPV_CPF.plot_alignment()
		#FAPV_OFL.complexity_analysis()
		#FAPV_OFL.plot_alignment()
		#plt.show()

	return data_OFL, data_CPF

def train(data_OFL, data_CPF, model):
	###################################
	# Step 4 || Train model			  #
	###################################
	print('(4) Trainning model...')


	if model == 'parafac': #parafac
		# ******** CPF PARAFAC analysis ******** #
		model_CPF = PARAFAC()
		data_CPF.inject_data(model_CPF)
		data_CPF.model = model_CPF
		CPF_s, CPF_Nloadings, CPF_model_mse = data_CPF.train( constraints={'non-negativity':'all', 'Null interferents':0}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']
		data_CPF.model.plot_loadings()
		plt.show()

		# ******** OLF PARAFAC analysis ******** #
		model_OFL = PARAFAC()
		data_OFL.Ni = 1
		data_OFL.Na = 1
		data_OFL.inject_data(model_OFL)
		data_OFL.model = model_OFL
		OLF_s, OLF_Nloadings, OLF_model_mse = data_OFL.train( constraints={'non-negativity':'all'}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']
		data_OFL.model.plot_loadings()
		plt.show()

	if model == 'MCR': #MCR
		# ******** CPF PARAFAC analysis ******** #
		model_CPF = MCR()
		data_CPF.inject_data(model_CPF)
		model_CPF.compress_3WAY()
		data_CPF.model = model_CPF

		CPF_S, CPF_C = data_CPF.train( constraints={'non-negativity':'all', 'Normalization':0}, ) # restriction=['non-negativity']

		# ******** OLF PARAFAC analysis ******** #
		model_OFL = MCR()
		data_OFL.inject_data(model_OFL)
		model_OFL.compress_3WAY()
		data_OFL.model = model_OFL

		OFL_S, OFL_C = data_OFL.train( constraints={'non-negativity':'all', 'Null interferents':0}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']
	return data_OFL, data_CPF

def predic(data_OFL, data_CPF, model):
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
		data_OFL.model.plot_45()
		data_OFL.model.plot_loadings()
		data_OFL.model.plot_convergence()
		plt.show()

	if model == 'MCR': #MCR
		# ******** CPF PARAFAC analysis ******** #
		# Predic CPF
		data_CPF.predic(v=0)
		data_CPF.summary( )

		# ******** OLF PARAFAC analysis ******** #
		# Predic data_OFL
		data_OFL.predic(v=0)#restriction = 'non-negative')
		data_OFL.summary( )

	return data_OFL, data_CPF

def save(data_OFL, data_CPF, path):
	###################################
	# Step 6 || Save data 			  #
	###################################
	print('(6) Saving results...')
	save = False
	if save:
		path = '{}/OFL'.format(path)
		data_OFL.export_4D_data( path=path ) 
		path = '{}/CPF'.format(path)
		data_CPF.export_4D_data( path=path ) 

def merge_OFL_CPF(data_OFL, data_CPF, path, save=True,):
	# === generate DATA obj === #
	data_merge = DATA()

	# === inject_data === #
	data_OFL.inject_data(data_merge)

	# === Set X data === #
	data_merge.D = np.concatenate( (data_OFL.D[:5,:,:,:], 
									data_CPF.D[:6,:,:,:], 
									data_OFL.D[5:,:,:,:]+data_CPF.D[6:,:,:,:]), axis=0 )
	data_merge.X = data_merge.D

	# === Set Y data === #
	data_merge.Y = np.zeros((5+6+7+8,2))
	data_merge.Y[:5,0] = data_OFL.Y[:5,0]
	data_merge.Y[5:5+6,1] = data_CPF.Y[:6,0]
	data_merge.Y[5+6:,0] = data_CPF.Y[6:,0]
	data_merge.Y[5+6:,1] = data_OFL.Y[5:,0]

	# === Basic data === #
	data_merge.Ni = 3 # variable
	data_merge.Na = 2
	data_merge.Nc = 6+5
	data_merge.N = 5+6+7+8 
	data_merge.Nt = 7+8
	data_merge.a = 2
	#if save:	data_merge.export_4D_data( path='{}/CPF+OLF+interferentsNONlinear'.format(path) ) 

	return data_merge

# ----------- PROFILER ----------- #
class Profiler(object): 

    def __init__(self, enabled=False, contextstr=None, fraction=1.0,
                 sort_by='time', parent=None, logger=None):
        self.enabled = enabled

        self.contextstr = contextstr or str(self.__class__)

        if fraction > 1.0 or fraction < 0.0:
            fraction = 1.0

        self.fraction = fraction
        self.sort_by = sort_by #cumulative

        self.parent = parent
        self.logger = logger

        self.stream = StringIO()
        self.profiler = cProfile.Profile()

    def __enter__(self, *args):

        if not self.enabled:
            return self

        # Start profiling.
        self.stream.write("\nprofile: {}: enter\n".format(self.contextstr))
        self.profiler.enable()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if not self.enabled:
            return False

        self.profiler.disable()

        sort_by = self.sort_by
        ps = pstats.Stats(self.profiler, stream=self.stream).sort_stats(sort_by)
        ps.print_stats(self.fraction)

        self.stream.write("\nprofile: {}: exit\n".format(self.contextstr))
        # save #
        ps.dump_stats(filename='profiling.prof')
		# gprof2dot -f pstats profiling.prof | dot -Tpng -o output.png && eog output.png

        return False

    def get_profile_data(self):

        value = self.stream.getvalue()
        if self.logger is not None:
            self.logger.info("%s", value)

        return value

    def save_profile_data(self):

        value = self.stream.getvalue()
        if self.logger is not None:
            self.logger.info("%s", value)

        return value

def main():
	print('open is assigned to %r' % open)
	with Profiler(enabled=True, contextstr="test") as pr:
		model = 'parafac'
		data_OFL, data_CPF = load()
		

		data_OFL, data_CPF, pure_vectors_mode = estimation(data_OFL, data_CPF)
		data_OFL, data_CPF = aling(data_OFL, data_CPF, pure_vectors_mode, model_aling='FAPV32')

		complexity = COMPLEXITY()
		FAPV_CPF.inject_data(  complexity )
		complexity.complexity_analysis()
		sadf
		# === Complexity analysis === FAPV_OFL
		# Cal-Cal samples warping 	0.9344434181514619
		# All samples warping 		0.7907079098473594
		# Cal-Test samples warping 	0.6426037725532717
		# complexity_overlap 		0.82

		# === Complexity analysis === FAPV_CPF
		# Cal-Cal samples warping 	0.9313692548826873
		# All samples warping 		0.7627384080407357
		# Cal-Test samples warping 	0.6069685101567605
		# complexity_overlap 		0.86

		for n in range(21):
			plt.plot( FAPV_CPF.L[n,0,:]/np.linalg.norm(FAPV_CPF.L[n,0,:]) ) 
		plt.figure(2), plt.plot( data_OFL.S[0].T )
		plt.figure(3), plt.plot( data_OFL.S[1].T )
		plt.show()
		

		data_OFL, data_CPF = train(data_OFL, data_CPF, model)
		data_OFL, data_CPF = predic(data_OFL, data_CPF, model)


		

		# === guardar datasets === #
		#save(data_OFL, data_CPF, path='/home/akaris/Documents/code/Chemometrics/files/case_000/ORDERED/DataSet/ICOSHIFT')

		# === convinar datasets === #
		#data_merge = merge_OFL_CPF(data_OFL, data_CPF, 
		#	path='/home/akaris/Documents/code/Chemometrics/files/case_000/ORDERED/DataSet/FAPV32')

	#stats = pstats.Stats(pr)
	#stats.sort_stats(pstats.SortKey.TIME)
	#stats.print_stats()
	#print(pr.get_profile_data())
	#python -m cProfile -o program.prof my_program.py

'''
data = DATA()
#data.load_folther( '/home/akaris/Documents/code/Chemometrics/files/case_000/ORDERED/DataSet/FAPV32/CPF+OLF+interferentsALL', data_shape='2D' )
data.load_folther( '/home/akaris/Documents/code/Chemometrics/files/case_000/ORDERED/DataSet/Order/CPF', data_shape='3D' )

model = MCR()
data.inject_data(model)
data.model = model

CPF_s, CPF_Nloadings   = data.train( constraints={'non-negativity':'all', 'Null interferents':0}, ) # restriction=['non-negativity'] ) # restriction=['non-negativity']
data.predic()
data.model.summary()
data.model.plot_loadings()
plt.show()

data.get_complexity()

data.export_4D_data(path='/home/akaris/Documents/code/Chemometrics/files/case_000/ORDERED/DataSet/FAPV32/test')
errpr
'''

from io import StringIO
import cProfile
import pstats
main()

