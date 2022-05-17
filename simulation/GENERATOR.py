# *** warning supresion *** #
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries
import numpy as np #pip3 install numpy

try:
	import matplotlib.pyplot as plt
except: 
	print('WARNNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: ( pip3 install matplotlib )')

import time

class SAMPLE(object):
	def __init__(self, X=None, D=None, Y=None, y=None, S=None, A=None, a=None,
				f=None, N=None, Nc=None, Nv=None, Nt=None, Na=None, Ni=None, 
				L=None, La=None, Ldim=None, ways=None,
				model=None, aling_model=None):
		if not X == None: self.X = np.array(X)
		else: self.X = None

		if not Y == None: self.Y = np.array(Y)
		else: self.Y = None

		self.D = D
		
		self.y = y
		self.S = S

		self.L = L
		self.loadings = self.L
		self.La = La
		self.Ldim = Ldim # list of loadings dimentions

		self.a = a 		# INT-8 || total number off well aligned ways  
		self.A = A 		# INT-8 || Total cumulated area

		self.N = N 		# INT-8 || dataset length 
		self.Nc = Nc 	# INT-8 || number of training sample
		self.Nv = Nv 	# INT-8 || number of validation sample
		self.Nt = Nt 	# INT-8 || number of test-samples 

		self.f = f 		# INT-8 || total number of factors   
		self.Na = Na 	# INT-8 || total number of analites 
		self.Ni = Ni 	# INT-8 || total number of interferents 

		self.ways = ways # toltal data tensor order

		self.model = model
		self.aling_model = aling_model

		self.pure_vectors = None
		self.pure_tensor = None

		self.colors = [ '#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',] 

		if self.f == None and self.Na != None and self.Ni != None: self.f = self.Na + self.Ni
	
	def generate(self, X=None, D=None, Y=None, y=None, factors=None, N=None, Ni=None, 
				functional=None, deformations=None, warping=None, sd=None, overlaping=None,
				dimentions=None, ways=None, pure_vectors=None,
				noise_type=None, intencity=None, 
				store=1, verv=1 ):
		# ways 			:	INT		:	number of ways		  		
		#								eg. ways = 3							( default: ways=None )
		# dimentions	:	VEC	 	:	dimentions of each way 		
		#								eg. dimentions = [50,60,40]				( default: dimentions=None )
		# N 			: 	VEC  	:	[Nc, Nv, Nt] number of calibration(Nc) validation(Nv) and test (Nt) samples	
		#								eg. N = [6,7,5]							( default: N=None )
		# factors		:	INT  	:	Number of calibrated factors 		
		#								eg. factors = 3 						( default: f=None )
		# interferentes :	INT 	:	Number of uncalibrated factors (interferents) 		
		#								eg. interferentes = 0					( default: Ni=None )
		# functional 	: 	STR 	:	Functional to describe each channel 		
		#								eg. functional='Gaussian'				( default: functional=None )
		# deformations 	:	MAT 	:	[mu, sd, Ng] x factors  mean value, sd and number of peaks of each channel 		
		#								eg. deformations=[[0,0,0],[0,0,0],[0,0,0]]	( default: deformations=None )
		# warping 		:	MAT 	:	[slide, compression] x factors miss alignement intencity 	of each analite	
		#								eg. warping=[[0,0],[0,0],[0,0]]			( default: warping=None )
		# sd 			:	VEC 	:	standar desviation of each signal 		
		#								eg. sd=[5,5]							( default: sd=None)
		# overlaping 	: 	VEC 	:	overlappint of each signal 		
		#								eg. overlaping=[0.1,0.1]				( default: overlaping=None )
		# pure_vectors 	:	TENSOR 	:	pure vectors of each analite  		
		#								eg. pure_vectors 						( default: pure_vectors=None )
		# noise_type 	:	STR 	:	type of noise  		
		#								eg. noise_type	 						( default: noise_type='random' )
		# intencity 	:	FLOAT 	:	noise intencity  		
		#								eg. intencity	 						( default: intencity=0.001 )

		# --- Pure vectors generation --- #
		if not type(pure_vectors) is np.ndarray: 
			if verv > 1: print(' -(1)- generationg pure_vectors')
			pure_vectors = self.generator_pure_vectors(ways=ways, dimentions=dimentions, N=N, 					# general shape
								factors=factors, interferentes=Ni, functional='Gaussian',									# ingredientes
								deformations=deformations, warping=warping, sd=sd, overlaping=overlaping)

		else:
			if verv > 1: print(' -(1)- reading pure_vectors')

		# --- Experiment generation --- #
		if not type(Y) is np.ndarray: 
			if verv > 1: print(' -(2)- generationg eigenvalues (Y)')
			Y = self.experiment_set(N=N, factors=factors, interferentes=Ni)

		else: 
			if verv > 1: print(' -(2)- reading eigenvalues (Y)')

		# --- Data structuration --- #
		if verv > 1: print(' -(3)- Estructurting data (X shape)')
		if not type(X) is np.ndarray: 
			X = self.data_estructuration(pure_vectors=pure_vectors, eigenvalues=Y, 
						ways=ways, dimentions=dimentions, N=N,factors=factors, interferentes=Ni)

		# --- Add noise --- #
		if verv > 1: print(' -(4)- Adding noise')
		X, noise = self.noise_add(data=X, noise_type=noise_type, intencity=intencity) 

		if store: 
			if verv > 1: print(' -(5)- Store data')
			self.X, self.D  = X, X
			self.Y = Y
			self.noise = noise

			self.f, self.Na, self.Ni = factors, factors, Ni
			self.Nc, self.Nv, self.Nt = N[0], N[1], N[2]
			self.N = N[0]+N[2]
		
		return X, Y, [N, factors, Ni]

	def gaussian(self, n,s,m):return (1.0/(s*(2*np.pi)**0.5)) * np.e**(-0.5*((np.arange(0, n, dtype=np.float32)-m)/s)**2)

	def lorentz(self, n,a,m):return (a*1.0/(np.pi*((np.arange(0, n, dtype=np.float32)-m))**2+a**2))

	def generator_pure_vectors(self, ways=None, dimentions=None, N=None, 					# general shape
								factors=None, interferentes=None, functional=None,			# ingredientes
								deformations=None, warping=None, sd=None, overlaping=None, 	# no idealidad
								store=1, verv=1):	
		# ----> [orden, n*de calibrado+n*de test, n*analitos+n*interferentes, n*de sensores]
		#	dimentions 			:	VEC 		: numero de sensores del o-esimo canal - recomendated len = (Na*6*s)
		#	ways 				:	INT 		: number of WAYS  
		# 	N					:	VEC[3] 		: [Nc, Nv, Nt] 
		#										Nc: number of calibration samples
		#										Nv: number of validation samples
		#										Nt: number of test samples
		#	factors				:	INT 		: total number of factors
		#	interferentes 		:	INT 		: total number of interferentes
		# 	Functional 			:	text 		: functional base 
		# 	deformations 		:	MAT[N, 3] 	: Introduce variations respect from the original Functional form		
		#			[0] : VAR : 0-1 mu of each gaussians respect sd
		#			[1] : VAR : 0-1 sd f each gaussians respect sd
		#			[2] : VAR : number of gaussians
		# 	warping				:	MAT[No, 2] 	: variability berween samples
		#			[0] : VAR : 0-1 shifting between samples respect sd
		#			[1] : VAR : 0-1 stretching between samples respect sd
		# 	sd 					:	VEC[w] 		: SD standar desviation of functions in sensors units
		# 	overlaping			:	VEC[w] 		: overlaping between adjasent analites
		#			[0] : VAR : 0-1 parametro de superpocion de picos consecutivos		
		if type(dimentions) == list:  	  dimentions = np.array(dimentions)
		if type(warping) 	== list:  	     warping = np.array(warping)
		if type(deformations) == list:  deformations = np.array(deformations)

		Nu = N[0]+N[2] # tatal number of samples #
		# ---- Generate pure_vectors structure ---- #
		pure_vectors = [ np.zeros((Nu, factors+interferentes, dimentions[n])) for n in range(ways) ] 	

		# ---- Parametros stocasticos de warping ----
		warp_var_space = np.random.rand(Nu, factors+interferentes, ways, 2)*2-1 			# warpinng in variable space #
		warp_var_space[:,:,:,0] = warp_var_space[:,:,:,0]*warping[:,0]*sd
		warp_var_space[:,:,:,1] = warp_var_space[:,:,:,1]*warping[:,1]*sd

		# ---- Parametros stocasticos de deformacion (not warping) ----
		deformations_var_space = np.random.rand(factors+interferentes, int(np.max(deformations[:,2])), ways, 2)*2-1 			# warpinng in variable space #
		deformations_var_space[:,:,:,0] = deformations_var_space[:,:,:,0]*deformations[:,0]*sd
		deformations_var_space[:,:,:,1] = deformations_var_space[:,:,:,1]*deformations[:,1]*sd*5

		for w in range(ways): 									# - Iteracion a1 : entre todos los orden de analisis -
			for n in range(Nu): 								# - Iteracion a2 : entre todas las muestras en analisis -	
				for f in range(factors):			# - Iteracion a3 : entre todos los analitos -
					for g in range( int(deformations[w,2]) ):	# - Iteracion a4 : entre todos los picos que inducen a las deformaciones -

						SD_i = sd[w] + deformations_var_space[f, g, w, 1] + warp_var_space[n, f, w, 1]
						# SD_i = A + B + C 
						# * A : sd 							 			: sd[w]
						# *	B : deformation from gaussian 				: deformations_var_space[f, g, w, 0]
						# *	C : shifting betwin samples 				: warp_var_space[n, f, w, 0]
						MU_i = 3*sd[w] + 6*sd[w]*(1-overlaping[w])*f +20+ deformations_var_space[f, g, w, 0] + warp_var_space[n, f, w, 0] 
						# MU_i = A + B + C + D + E
						# * A : 3 sd separation from origin 			: 3 SD 
						# *	B : 6 SD separation with next factor 		: 6*sd[w]*(1-overlaping[w])*f
						# 		more overlaping increase channel superposition
						# *	C : constant separation from origin 		: 20
						# *	D : deformation from gaussian 			: deformations_var_space[f, g, w, 0]
						# *	E : shifting betwin samples 				: warp_var_space[n, f, w, 0]

						if functional == 'Lorentz': 	pure_vectors[w][n,f,:] += self.gaussian(dimentions[w], abs(SD_i), MU_i)/int(deformations[w,2])
						elif functional == 'Gaussian': 	pure_vectors[w][n,f,:] += self.gaussian(dimentions[w], abs(SD_i), MU_i)/int(deformations[w,2])

			for n in range(N[0], N[0]+N[2]): 					# - Iteracion a2 : entre todas las muestras en analisis - 					(INTERFERENTES)
				for f in range(factors, factors+interferentes):			# - Iteracion a3 : entre todos los analitos -						(INTERFERENTES)
					for g in range( int(deformations[w,2]) ):	# - Iteracion a4 : entre todos los picos que inducen a las deformaciones - 	(INTERFERENTES)

						SD_i = sd[w] + deformations_var_space[f, g, w, 1] + warp_var_space[n, f, w, 1]
						MU_i = 3*sd[w] + 6*sd[w]*(1-overlaping[w])*(f-factors+0.5)+20 + deformations_var_space[f, g, w, 0] + warp_var_space[n, f, w, 0]

						if functional == 'Gaussian': pure_vectors[w][n,f,:] += self.gaussian(dimentions[w], abs(SD_i), MU_i)/int(deformations[w,2])
						elif functional == 'Lorentz': pure_vectors[w][n,f,:] += self.gaussian(dimentions[w], abs(SD_i), MU_i)/int(deformations[w,2])

		if store == 1: # -- store posible usefull data in class SAMPLE -- #
			self.pure_vectors = pure_vectors

		return pure_vectors # --> (O, Ncal+Ntest, Na+Ni, Ln ) -- [orden, n*de calibrado+n*de test, n*analitos+n*interferentes, n*de sensores]

	def experiment_set(self, N=None, factors=None, interferentes=None, experiment='Random', error=None, store=True, min_cc=1):
		# 	Ncal			:	VAR 	: Numero de muestras de calibrado
		#	Ntest			:	VAR 	: Numero de muestras test
		#	Na 				:	VAR 	: Numero de analitos
		# 	Ni 				:	VAR 	: Numero de interferentes
		# 	experiment		:	STR 	: Tipo de experimento
		# 	error 			:	VAR 	: Error aleatorio considerado sobre las cc del set de calibrado.
		# 	min_cc 			:	FLOAT 	: minimun cc value 
		min_cc = min_cc if type(min_cc) in [int, float] else 0.0
		Nu = N[0]+N[2]	# tatal number of samples # 

		Y = np.random.rand(Nu, factors+interferentes) + min_cc
		#	data 			: 	Matrix	:	Matrix completa de datos
		if experiment == 'Clasic' or experiment == 'default':
		#	  --- Na ----   -- Ni --			
		#	[ 1 0 0 ... 0 | 0 ... 0 ] -
		#	[ 0 1 0 ... 0 | 0 ... 0 ] |
		#	[ 0 0 1 ... 0 | 0 ... 0 ] |
		#	[ . . . ... 1 | 0 ... 0 ] |
		# 	[ R R R ... R | 0 ... 0 ] Ncal
		#	[ R R R ... R | 0 ... 0 ] |
		#	[ . . . ... . | 0 ... 0 ] |
		#	[ R R R ... R | 0 ... 0 ] -
		#	[ ----------------------]
		#	[ R R R ... R | R ... R ] -
		#	[ R R R ... R | R ... R ] |
		#	[ . . . ... . | . ... . ] Ntest
		#	[ R R R ... R | R ... R ] |
		#	[ R R R ... R | R ... R ] -
			Y[:N[0], factors:] = 0
			Y[:factors, :factors] = np.eye( factors )

		if experiment == 'Random' or experiment == 'random':
		#	  --- Na ----   -- Ni --			
		#	[ 1 0 0 ... 0 | 0 ... 0 ] -
		#	[ 0 1 0 ... 0 | 0 ... 0 ] |
		#	[ 0 0 1 ... 0 | 0 ... 0 ] |
		#	[ . . . ... 1 | 0 ... 0 ] |
		# 	[ R R R ... R | 0 ... 0 ] Ncal
		#	[ R R R ... R | 0 ... 0 ] |
		#	[ . . . ... . | 0 ... 0 ] |
		#	[ R R R ... R | 0 ... 0 ] -
		#	[ ----------------------]
		#	[ R R R ... R | R ... R ] -
		#	[ R R R ... R | R ... R ] |
		#	[ . . . ... . | . ... . ] Ntest
		#	[ R R R ... R | R ... R ] |
		#	[ R R R ... R | R ... R ] -
			pass

		if store == 1: # -- store posible usefull data in class SAMPLE -- #
			self.Y = Y

		return Y


	def data_estructuration(self, pure_vectors=None, eigenvalues=None, 
								ways=None, dimentions=None, N=None, 					# general shape
								factors=None, interferentes=None,
								store = 1 ): 
		#  ---> data  			: 	TENSOR 		: Contiene toda la informacion estructurada del sistema. [n*de calibrado+n*de test, n*de sensores**orden]
		# pure_vectors 			:	TENSOR		:	Tensor con toda la informacion del sistema 
		#										[orden, n*de calibrado+n*de test, n*analitos+n*interferentes, n*de sensores]
		# pure_vectors 			:	Matrix		:	Tensor con toda las concentraciones del set de calibrado y test 
		#										[n*de calibrado+n*de test, n*analitos+n*interferentes]
		#	pure_vectors[w][n,f,v]
		#	dimentions 			:	VEC 		: numero de sensores del o-esimo canal - recomendated len = (Na*6*s)
		#	ways 				:	INT 		: number of WAYS 
		# 	N					:	VEC[3] 		: [Nc, Nv, Nt] 
		#										Nc: number of calibration samples
		#										Nv: number of validation samples
		#										Nt: number of test samples
		#	factors				:	INT 		: total number of factors
		#	interferentes 		:	INT 		: total number of interferentes
		def npzeros( *args): return np.zeros(args)

		Nu = N[0]+N[2]	# tatal number of samples #
		data = npzeros( Nu, *dimentions ) # ! (CHANGE in python 2.7)
		pure_tensor = npzeros( factors+interferentes, Nu, *dimentions ) # ! (CHANGE in python 2.7)

		for n in range(Nu):
			for f in range(factors+interferentes):
				T = pure_vectors[0][n,f,:]
				for w in range(1,ways):
					T = np.tensordot(T, pure_vectors[w][n,f,:], axes=0)
				pure_tensor[f, n, :] += T 
				data[n, :] += T * eigenvalues[n, f]

		if store == 1: # -- store posible usefull data in class SAMPLE -- #
			self.X = np.array( data ) 
			self.D = np.array( data ) 
			self.pure_tensor = pure_tensor

		return np.array( data ) #  ---> data  			: 	TENSOR 		: Contiene toda la informacion estructurada del sistema. 

	def noise_add(self, data=None, noise_type='random', intencity=1, store=1):  # -- ADD noise -- #
		# 	data			:	TENSOR 			: TENSOR to add noise
		#	type			:	STR 			: noise structure
		#	intencity 		: 	float32 		: Noise intencity
		if not type(data) is np.ndarray: 
			if data == None: data = self.X

		if noise_type == 'random': 
			noise = np.random.randn( *data.shape ) * intencity
			data += noise

		if store == 1: # -- store posible usefull data in class SAMPLE -- #
			self.X = data
			self.D = data
			self.noise = noise
		return data, noise

	# ***** INYEcT ***** #
	def inyect_data(self, obj):
		try: obj.__dict__ = self.__dict__.copy() 
		except: print(' ERROR  :: code X :: GENERATOR.inyect_data() :: can not inject data into {}'.format(str(obj)))

# ---- ---- ---- ---- ---- eg. usage ---- ---- ---- ---- ---- #
# SD_i = sd[w] + deformations_var_space[f, g, w, 1] + warp_var_space[n, f, w, 1]
# MU_i = 3*sd[w] + 6*sd[w]*(1-overlaping[w])*f +20+ deformations_var_space[f, g, w, 0] + warp_var_space[n, f, w, 0] 

'''
# === EG. superposition === #
def gaussian(n,s,m):return (1.0/(s*(2*np.pi)**0.5)) * np.e**(-0.5*((np.arange(0, n, dtype=np.float32)-m)/s)**2)
sd = 60
N = 1000
w = 0
overlaping = 0.55

SD_i = sd + w
MU_i = 3*sd + 6*sd*(1-overlaping)*0 + 200 + w 
plt.plot( gaussian(N, SD_i, MU_i) )

SD_i = sd + w
MU_i = 3*sd + 6*sd*(1-overlaping)*1 + 200 + w 
plt.plot( gaussian(N, SD_i, MU_i) )

plt.show()
'''

'''
# - (1) define usefull libs - # 
from analisys import MCR, PARAFAC, GPV
from alig import FPAV
from data import DATA

# - (2) make class instance - #
sample = SAMPLE()

# - (3) define problem parameters - #
ways=3					# number of ways #
dimentions=[50,60,40] 	# dimentionality of each way #
N=[6,7,5] 				# [trainnig samples, validation samples, test samples] #

factors=3 				# number of calibrated factors #
interferentes=0 		# number of interferents #

functional='Gaussian' 						# functional basee form of pure vectors # 
deformations=[[0.0,0.0,1],[0,0,1],[0,0,1]] 	# deformations #
warping=[[0.0,0.0],[0,0],[0,0]] 			# warping #
sd=[2,2,2] 										# SD of pure vectors #
overlaping=[0,0,0] 								# overlapping of pure vectors # 

noise_type='random'		# noise structure #
intencity=0.001 		# noise intensity #

# - (4) ganerate problem - #
X, Y, metadata = sample.generate(factors=factors, N=N, Ni=interferentes, 
								functional=functional, deformations=deformations, warping=warping, sd=sd, overlaping=overlaping,
								dimentions=dimentions, ways=ways, noise_type=noise_type, intencity=intencity)

plt.matshow(X[6,:,:,35])
plt.show()

# - (4) ganerate problem - #
data_O = DATA()

# - (5) try to solve problem (GPV)- #
data_O.Ni = interferentes
data_O.Na = factors
data_O.Nc = N[0]
data_O.N =  N[0]+N[2]
data_O.Nt = N[2]
data_O.a = 2
data_O.Y = Y
data_O.X = X
data_O.D = X

PARAFAC = GPV()
data_O.inyect_data(PARAFAC)
data_O.model = PARAFAC
data_O.train( )
data_O.predic()

plt.figure(211), plt.plot(data_O.model.L[1].T)
plt.figure(212), plt.plot(data_O.model.L[0].T)
plt.figure(213), plt.plot(data_O.model.L[2].T)
#plt.show()
'''