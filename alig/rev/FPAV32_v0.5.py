# *** warning supresion
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries
import numpy as np #pip3 install numpy
from functools import reduce

# *** graph libraries
try:
	import matplotlib.pyplot as plt #pip3 install matplotlib
except: 
	print('WARNNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: ( pip3 install matplotlib )')
	
class FAPV32(object):
	def __init__(self, X=None, D=None, L=None, La=None, f=None, A=None, a=1,
				Y=None, y=None, S=None,	N=None, Nc=None, Na=None, Ni=None, Nt=None ):

		self.D = D 		# Complete data set #
		self.X = X 		# Complete data set #
		
		self.Da = None 	# Complete well aligned data set #

		self.Y = Y 		# Complete response values #
		self.y = y 		# Complete response values #
		
		self.S = S 		# well aligned ways (initial estimations) #

		
		self.L = L 		# original estimated loadings #
		self.La = La 	# well aligned loadings #

		self.a = a 		# number of well aligned ways #
		self.A = A 		# estimated areas values #

		self.f = f 		# number of factors #

		self.N = N  	# total number of sample N = Nc + Nt + Nv #
		self.Nc = Nc    # total number of calibration samples #
		self.Nt = Nt 	# total number of calibration samples #

		self.Na = Na 	# total number of calibrated factors #
		self.Ni = Ni 	# total number of uncalibrated factors #

		self.sigma_range = None 	# sigma exploration space #
		self.mu_range = None 		# mu exploration space #

		# color palette #
		self.colors = [ '#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',] 

	def aling(self, D=None, S=None, Nc=None, Na=None, Ni=None, Nt=None, a=None,
					area={'mode':'sum'}, shape='gaussian', functional='gaussian', non_negativity=True, 
					linear_adjust=True, SD={'mode':'constant'},	
					calibration_interference=False, interference_elimination=True,
					save=True, v=1
					): 
		# -- variable setter for aling_FAPVXX -- #
		# checking variable integrity #
		if v > 0: print('Checking variable integrity.')
		
		D = D if type(D) is np.ndarray else self.D
		S = S if type(S) is list else self.S

		Nc = Nc if type(Nc) in [np.ndarray, int, float] else self.Nc
		Na = Na if type(Na) in [np.ndarray, int, float] else self.Na
		Ni = Ni if type(Ni) in [np.ndarray, int, float] else self.Ni
		a  = a  if type(a ) in [np.ndarray, int, float] else self.a
		
		non_negativity = non_negativity if type(non_negativity) == bool else True 
		linear_adjust = linear_adjust if type(linear_adjust) == bool else False

		if save:
			if v > 0: print('Store actual data in obj instace.')
			# if save == True then save all actual data #
			self.D = D
			self.S = S

			self.Nc = Nc
			self.Na = Na
			self.Ni = Ni
			self.a  = a

		if self.a == 2: 
			if v > 0: print('aling_FAPV21 mode.')
			# 3 order, 2 way, 1 aligned channel, 1 non aligned channel  
			self.aling_FAPV32(	D=D, S=S, Nc=Nc, Na=Na, Ni=Ni, Nt=Nt, 
								area=area, shape=shape, non_negativity=non_negativity, linear_adjust=linear_adjust, SD=SD,
								interference_elimination=interference_elimination,
								save=save)
		else:	
			print('WARNNING :: FAPV32.aling() :: There must be exactly 2 NON well aligned channel. You can correct this with: self.a = 2 or FAPV32.a = 2')
			print('WARNNING :: FAPV32.aling() :: FAP32 can detect this and assume self.a = 2. But it is recommended to properly correct this.')
			
			if v > 0: print('aling_FAPV21 mode.')
			# 3 order, 2 way, 1 aligned channel, 1 non aligned channel  
			self.aling_FAPV32(	D=D, S=S, Nc=Nc, Na=Na, Ni=Ni, Nt=Nt, 
								area=area, shape=shape, functional=functional, non_negativity=non_negativity, linear_adjust=linear_adjust, SD=SD,
								calibration_interference=calibration_interference, interference_elimination=interference_elimination,
								save=save)
		return self.Da, self.La, self.L # 
	
	def aling_FAPV32(self, D=None, X=None,S=None, Nc=None, Na=None, Ni=None, Nt=None, a=None, shape='gaussian', functional='functional',
					area={'mode':'sum'}, non_negativity=True, linear_adjust=True, SD={'mode':'constant'},
					calibration_interference=False, interference_elimination=False,
					save=True, v=1, 
					): 		# FPAV for 4 order, 3 way, 2 aligned channel, 1non aligned channel 

		# ----- Alignment algorithm for 3rd order data with 2 aligned channels ----- #
		# 
		# D 				:	N-MAT 		:		N-array/Matrix non-aligned data D => [N,l1,l2]
		# S 				:	N-MAT 		:		N-array/Matrix well aligned channels  S => [[self.f,l1],[self.f,l2]] 
		# Nc				:	INT 		: 		number of calibration samples.
		# Na				:	INT 		: 		number of analites.
		# Ni				:	INT 		: 		number of interferences.
		# Nt				:	INT 		: 		number of test samples.
		# a					:	INT 		: 		number of well aligned channels.
		# shape				:	str 		: 		How to preserve the functional shape. 	{'gaussian', 'original'}
		# area 				:   str 		: 		area preserving mode 					{'gaussian', 'max', 'sum'}
		# non_negativity 	:	BOOL  		: 		NON-negativity constrain
		# linear_adjust 	:	BOOL  		: 		Linear fit. (not implemented yet)
		# mu_range			:	N-array 	: 		this parameter consists of the exploration range of mu (of the functions of the functional space)
		#									eg: mu_range=np.arange(150, 300, 3)
		# sigma_range 		: N-array		: 		this parameter consists of the exploration range of sigma (of the functions of the functional space)
		#									eg: sigma_range=np.arange(3, 30, 1)
		# save 				: 	BOOL		:		this save the
		# v 				: 	BOOL		: 		verboisity
		# ------------------------------------------------------------------------------- #
		# self.D 		:	N-MAT 		:		N-array/Matrix non-aligned data D => [N,l1,l2]
		# self.S 		:	N-MAT 		:		N-array/Matrix well aligned channels  S => [[self.f,l1],[self.f,l2]] 
		# self.Nc		:	INT 		: 		number of calibration samples.
		# self.Na		:	INT 		: 		number of analites.
		# self.Ni		:	INT 		: 		number of interferentes.
		# ------------------------------------------------------------------------------- #
		# self.S  	: [L1, L2] | L1=[f,l1], L2=[f,l2] 
		# self.f  	: numero de factores = Numero de analitos(Na) + Numero de interferentes(Ni)
		# self.N  	: Numero de muestras de calibrado(Nc) + Numero de muestras test(Nt)
		# l1 		: number of variables of channel 1
		# l2 		: number of variables of channel 2
		# l3 		: number of variables of channel 3

		# --- Define variables --- # 
		if v >= 1: print( ('0- Setting variables') )

		D = D if type(D) is np.ndarray else self.D
		X = X if type(X) is np.ndarray else self.X

		if type(D) is np.ndarray and type(X) is np.ndarray and not (X==D).all(): D = X
		S = S if type(S) is list else self.S

		Nc = Nc if type(Nc) in [np.ndarray, int, float] else self.Nc
		Na = Na if type(Na) in [np.ndarray, int, float] else self.Na
		Ni = Ni if type(Ni) in [np.ndarray, int, float] else self.Ni
		a  = a  if type(a ) in [np.ndarray, int, float] else 2

		non_negativity = non_negativity if type(non_negativity) == bool else True 
		linear_adjust = linear_adjust if type(linear_adjust) == bool else False

		N, l1, l2, l3  =  D.shape
		self.N = N if save else self.N 
		Nt = Nt if type(Nt) in [int, float, np.ndarray] else N-Nc
		Nt, f = N-Nc, Na+Ni

		# --- STEPS --- #
		# (1) estimation of well aligned channels (commonly spectral channels)
		# (2) Estimate non-spectral channels
		# (3) Align non-spectral channels
		# (4) Rebuild the samples
		# (5) Post processing

			# ------ ------ CALIBRATION SAMPLE ------ ------ #
		# --- Z --- #
		if v >= 1: print( ('1- Initializing matrix') )
		Ze = np.zeros( (Na, l1, l2) )
		for n in range(Na):	Ze[n,:,:] = np.tensordot( S[0][n,:], S[1][n,:] , axes=0)
		Z = Ze[:,0,:]
		for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

	  	# --- Y --- #
		Ye = D[0,:,:,:]
		for n in range(1, Nc): Ye = np.concatenate( (Ye ,  D[n,:,:,:]), axis=2)
		Y = Ye[0,:,:]
		for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

	  	# --- X(Lc) --- #
		Lc=np.dot(Y.T, np.linalg.pinv(Z) )
		L = np.zeros( (Nc, Na, l3) )
		for nc in range(Nc): L[nc,:,:] = Lc[nc*l3:(nc+1)*l3,:].T

		# ------ AREAS ------ #
		if area['mode'] == 'sum':
			A = np.sum( L, axis=2 ) #

		elif area['mode'] == 'max':
			A = np.max( L, axis=2 )  

		elif area['mode']=='gaussian_coeficient':
			#  evaluate proyection coef and gaussian parameters 
			coef = np.zeros((Nc+Nt, Na+Ni))
			Gcoef = np.zeros((Nc+Nt, Na+Ni, 2))
			for nc in range(Nc):
				for na in range(Na):
					CC = self.G( L[nc,na,:], mu_range=area['mu_range'], sigma_range=area['sigma_range'])
					coef[nc, na] = CC[1]
					Gcoef[nc, na, :] = CC[2], CC[3] 
					if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
			A = coef
		
		if area['mode']=='gaussian_coeficient': 
			#  evaluate all eigen functions (this step is not necesary ) 
			all_gauss_aling = np.zeros((Nc+Nt, Na+Ni, l2))
			for na in range(Na):
				for nc in range(Nc): 
					all_gauss_aling[nc, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l2) * A[nc, na] 

			all_gauss = np.zeros((Nc+Nt, Na+Ni, l2))
			for na in range(Na):
				for nc in range(Nc): 
					all_gauss[nc, na, :] = self.gaussian(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l2) * A[nc, na] 
		
		elif area['mode'] == 'quela':
			A = self.quela( 'calibration', area=area )

		# ------ FUNCTIONAL ALIGNMENT ------ # Estimacion del tensor de datos alineado (self.Da)
		if v >= 1: print( ('3- Functional alignement (calibration samples)') )
		Lmax, Amax, Da = np.zeros((Na)), np.argmax(A, axis=0), np.zeros((N, l1, l2, l3))

		for na in range(Na): Lmax[na] = np.argmax(L, axis=2)[Amax[na]][na]

		if v >= 1: print( ('4- Recostructing data (calibration samples)') )

		if shape == 'original': # retains the original shape  
			L_mean = np.mean(L, axis=0) # mean loading estimation for all samples
			for na in range(Na):
				Ma = np.tensordot(S[0][na, :], L_mean[na, :] , axes=0) 	# inner product (spectra, normalized eigen function with discrete partition)
				for nc in range(Nc): Da[nc,:,:] += Ma*A[nc, na] 		# eigen matrix * eigen value

			# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
			La = np.zeros( (N, f, l2) )
			for na in range(Na):
				for nc in range(Nc): La[nc, na, :] = L_mean[na, :]*A[nc, na] # eigen vector * eigen value

		elif shape == 'gaussian': # gaussian shape for eigen function  
			# - standard deviation estimation - #
			if SD['mode'] == 'mean': 								sigma = np.mean(Gcoef[:, :, 1], axis=0)
			if SD['mode'] == 'constant' and 'value' in SD: 			sigma = [SD['value'] ]*(Na+Ni)
			elif SD['mode'] == 'constant' and not 'value' in SD: 	sigma = [float(l2)/f/5 ]*(Na+Ni)

			for na in range(Na):
				Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), self.gaussian(Lmax[na], sigma[na], l3) , axes=0)
				for nc in range(Nc): Da[nc,:,:,:] += Ma*A[nc, na]

			# --- Estimation of originally NON-aligned channels (self.L) but already aligned (self.La).
			La = np.zeros( (N, f, l3) )
			for na in range(Na):
				for nc in range(Nc): La[nc, na, :] = self.gaussian(Lmax[na], sigma[na], l3)*A[nc, na]

	  	#  ------ ------ TEST SAMPLES ------ ------ #
		if Nt > 0:
			# --- FUNCTIONAL ALIGNMENT --- # 
			if v >= 1: print( ('3b- Functional alignement (test samples)') )
			# --- Z --- #
			Ze = np.zeros( (f, l1, l2) )
			for n in range(f):	Ze[n,:,:] = np.tensordot( S[0][n,:], S[1][n,:] , axes=0)
			Z = Ze[:,0,:]
			for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

			# --- Y --- #
			Ye = D[Nc,:,:,:]
			for n in range(1,Nt): Ye = np.concatenate( (Ye ,  D[Nc+n,:,:,:]), axis=2)
			Y = Ye[0,:,:]
			for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

			# --- X(Lc) --- #
			Lc=np.dot(Y.T, np.linalg.pinv(Z) )
			L = np.zeros( (Nt, f, l3) )
			for nt in range(Nt): L[nt,:,:] = Lc[nt*l3:(nt+1)*l3,:].T 

			# ------ AREAS ------ #
			if area['mode'] == 'sum':
				A = np.sum( L, axis=2 ) 

			if area['mode'] == 'max':
				A = np.max( L, axis=2 )  

			if area['mode']=='gaussian_coeficient':
				#  evaluate projection coef and Gaussian parameters
				coef = np.zeros((Nt, f))
				Gcoef = np.zeros((Nt, f, 2))
				for nc in range(Nt):
					for na in range(f):
						CC = self.G( L[nc,na,:], mu_range=area['mu_range'], sigma_range=area['sigma_range'] )
						coef[nc, na] = CC[1]
						Gcoef[nc, na, :] = CC[2], CC[3] 
						if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
				A = coef

			if area['mode']=='gaussian_coeficient':
				#  evaluate all eigen functions (this step is not necessary ) 
				for na in range(Na+Ni):
					for nt in range(Nt): 
						all_gauss_aling[Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l2) * A[nt, na] 

				for na in range(Na+Ni):
					for nt in range(Nt): 
						all_gauss[Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l2) * A[nt, na] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			if area['mode']=='quela':
				A = self.quela( 'test', area=area )

			if interference_elimination == True:	A[:,Na: ] = 0

			if linear_adjust:
				pass # not implemented

			# --- FUNCTIONAL ALIGNMENT --- # Estimation of aligned data tensor (self.Da)
			if v >= 1: print( ('4b- Reconstructing data (test samples)') )

			if shape == 'original': # retains the original shape  
				L_mean = np.mean(L, axis=0) # mean loading estimation for all samples
				# L.shape = (Nc, Na, l2)
				# L_mean.shape = (Na, l2)
				for na in range(Na):
					Ma = np.tensordot( S[0][na, :], L_mean[na, :] , axes=0)	# inner product (spectra, normalized eigen function with discrete partition)
					for nt in range(Nt): Da[Nc+nt,:,:] += Ma*A[nt, na]		# eigen matrix * eigen value

				# --- Estimation of channels originally NOT aligned (self.L) but already aligned (self.La).
				for na in range(Na):
					for nt in range(Nt): 
						La[nt+Nc, na, :] = L_mean[na, :]*A[nt, na] # eigen vector * eigen value 


			elif shape == 'gaussian': # Gaussian shape for eigen function  
				# --- ALINEADO FUNCIONAL --- # Estimacion del tensor de datos alineado (self.Da)
				for na in range(Na):
					Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), self.gaussian(Lmax[na], sigma[na], l3) , axes=0)
					for nt in range(Nt): Da[Nc+nt,:,:,:] += Ma*A[nt, na]

				# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
				for na in range(Na):
					for nt in range(Nt): La[nt+Nc, na, :] = self.gaussian(Lmax[na], sigma[na], l3)*A[nt, na]


		# Devolvemos (0-self.Da)Tensor de datos alineado (1-self.La)Perfiles cromatograficos alineados (2-self.L)Estimacion original de los perfiles cromatograficos
		if save: 
			print('Saving results...')
			try:
				self.all_gauss_aling = all_gauss_aling
				self.all_gauss = all_gauss
			except:
				self.all_gauss_aling = None
				self.all_gauss = None
				
			self.A = A
			self.La = La
			self.Da = Da
			self.L = L
			self.Lc=np.dot(np.array(reduce(lambda x, y:  np.concatenate( (x ,  y), axis=1), D[:,:,:] )).T, np.linalg.pinv(S[0][:, :]) )

		print('Successful alignment.')

		return Da, La, L 
		# return self.Da, self.La, self.L
		# ---------------------------------------------------- #
		# self.Da 		:	ndarray  :  well aligned data 
		# self.La 		:	ndarray  :  well aligned loadings   shape [Nt+Nc, Na, l2]
		# self.L 		:	ndarray  :  stimated loadings		shape [Nt+Nc, Na, l2]
		# ---------------------------------------------------- #
		#	Nt 			: 	 INT 	:	number of test samples 
		#	Nc 			: 	 INT 	:	number of calibration samples 
		#	Na 			: 	 INT 	:	number analites (calibrated factors) 
		#	Ni 			: 	 INT 	:	number non interfers (non-calibrated factors) 
		#	l2 			: 	 INT 	:	number variables of the non-aligned way
		# ---------------------------------------------------- #
	def quela(self, mode, area, functional='gaussian', v=False):
		Nc = self.Nc
		Nt = self.Nt
		Na = self.Na+1
		f = self.Na + self.Ni
		N, l1, l2, l3 = self.D.shape

		if mode == 'calibration':			
			# *** *** *** CALIBRATION *** *** *** #	
			# --- Z --- #
			Ze = np.zeros( (Na, l1, l2) )
			for n in range(Na):	Ze[n,:,:] = np.tensordot( self.S[0][n,:], self.S[1][n,:] , axes=0)
			Z = Ze[:,0,:]
			for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

			# --- Y --- #
			Ye = self.D[0,:,:,:]
			for n in range(1, Nc): Ye = np.concatenate( (Ye ,  self.D[n,:,:,:]), axis=2)
			Y = Ye[0,:,:]
			for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

			# --- X(Lc) --- #
			Lc=np.dot(Y.T, np.linalg.pinv(Z) )
			L = np.zeros( (Nc, Na, l3) )
			for nc in range(Nc): L[nc,:,:] = Lc[nc*l3:(nc+1)*l3,:].T

			coef = np.zeros((Nc, f))
			Gcoef = np.zeros((Nc, f, 2))
			for nc in range(Nc):
				for na in range(f):
					CC = self.G( L[nc,na,:], mu_range=area['mu_range'], sigma_range=area['sigma_range'] )
					coef[nc, na] = CC[1]
					Gcoef[nc, na, :] = CC[2], CC[3] 
					if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
				
				plt.plot( L[nc,0,:].T, 'r') # !!!!!!!!!!!!!!!!!!!!!!!!!!!
				print( Gcoef[nc, 0, :] )# !!!!!!!!!!!!!!!!!!!!!!!!!!!
				plt.plot( self.gaussian(Gcoef[nc, 0, 0] , Gcoef[nc, 0, 1] , l3)*coef[nc, 0], 'k' ) # !!!!!!!!!!!!!!!!!!!!!!!!!!!
			plt.show()

			A = coef

		if mode == 'test':
			# *** *** *** TEST *** *** *** #
			# --- Z --- #
			Ze = np.zeros( (f, l1, l2) )
			for n in range(f):	Ze[n,:,:] = np.tensordot( self.S[0][n,:], self.S[1][n,:] , axes=0)
			Z = Ze[:,0,:]
			for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

			# --- Y --- #
			Ye = self.D[Nc,:,:,:]
			for n in range(1,Nt): Ye = np.concatenate( (Ye ,  self.D[Nc+n,:,:,:]), axis=2)
			Y = Ye[0,:,:]
			for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

			# --- X(Lc) --- #
			Lc=np.dot(Y.T, np.linalg.pinv(Z) )
			L = np.zeros( (Nt, f, l3) )
			for nt in range(Nt): L[nt,:,:] = Lc[nt*l3:(nt+1)*l3,:].T 

			coef = np.zeros((Nt, f))
			Gcoef = np.zeros((Nt, f, 2))
			for nc in range(Nt):
				for na in range(f):
					CC = self.G( L[nc,na,:], mu_range=area['mu_range']+area['test shift'], sigma_range=area['sigma_range'] )
					coef[nc, na] = CC[1]
					Gcoef[nc, na, :] = CC[2], CC[3] 
					if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
				
				plt.plot( L[nc,0,:].T, 'r')
				print( Gcoef[nc, 0, 1] )
				plt.plot( self.gaussian(Gcoef[nc, 0, 0] , Gcoef[nc, 0, 1] , l3)*coef[nc, 0], 'k' )
			plt.show()

			A = coef

		return A

	def gaussian(self, mu, sigma, n, ):
		# return Gaussian vector with n dimension G(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mean value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# ---------------------------------------------------- #
		f = np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)
		return f/np.linalg.norm(f)

	def Lorentz(self,n,a,m):
		return (a*1.0/(np.pi*((np.arange(0, n, dtype=np.float32)-m))**2+a**2))

	def Rayleigh(self,mu, sigma, n, ):
		# return Rayleigh distribution vector with n dimension R(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mode value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# ---------------------------------------------------- #
		x = np.arange(n) - mu + sigma # x + desire_mean - actual_mean  
		f = x/sigma**2 * np.e**(-1.0/2 * (x/sigma)**2)
		f[f<0]=0
		return f/np.linalg.norm(f)

	def GIG(self,mu, sigma, n, a=2, b=1, p=-1):
		'''
		In probability theory and statistics, the generalized inverse Gaussian 
		distribution (GIG) is a three-parameter family of continuous probability 
		istributions with probability density function
		# return generalized inverse Gaussian distribution vector with n dimension
		 R(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mode value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# ---------------------------------------------------- #
		'''

		x = (np.arange(n))*5/n* sigma
		f = x**(p-1) * np.e**(-(a*x+b/x)/2)
		f[f<0]=0
		f = np.nan_to_num(f, copy=True, nan=0.0, posinf=None, neginf=None)

		x = (np.arange(n)+np.argmax(f)-mu)*5/n * sigma
		x[x<0] = 0

		f = x**(p-1) * np.e**(-(a*x+b/x)/2)
		f[f<0]=0
		f = np.nan_to_num(f, copy=True, nan=0.0, posinf=None, neginf=None)

		return f/np.linalg.norm(f)

	def plot(self,):
		color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', ]
		for n in range(2):
			plt.plot(self.L[:,n,:].T, '-o', color=color[n])
		plt.show()

	def plot_spectra(self,):
		color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', ]
		for n in range(2):
			plt.figure(1), plt.plot(self.S[0][n,:].T, '-o', color=color[n])
			plt.figure(2), plt.plot(self.S[1][n,:].T, '-o', color=color[n])
		plt.show()

	def summary(self,):
		print(' *** Sensors ***')
		for i, n in enumerate(self.D.shape):		
			if i == 0: print('  *  Loaded data has '+str(n)+' samples')
			elif i != 3: print('  *  (aligned) Channel '+str(i)+' has '+str(n)+' sensors')
			elif i == 3: print('  *  (NON-aligned) Channel '+str(i)+' has '+str(n)+' sensors')


	# ***** INJEcT ***** #
	def inject_data(self, obj, force=False, replace=True):
		if force:
			try: obj.__dict__ = self.__dict__.copy() 
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
		elif not replace :
			try: 	obj.__dict__ =  { key:value if not key in obj.__dict__ else obj.__dict__[key] for (key,value) in list(obj.__dict__.items())+list(self.__dict__.items()) }
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
		elif replace :
			try: 	obj.__dict__ =  { key:value if not key in self.__dict__ else self.__dict__[key] for (key,value) in list(obj.__dict__.items())+list(self.__dict__.items()) }
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
	
		return None

	# ***** import from obj ***** #
	def import_data_obj(self, obj, force=False, replace=True):
		if force:
			try: self.__dict__ = obj.__dict__.copy() 
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
		elif not replace :
			try: 	self.__dict__ =  { key:value if not key in self.__dict__ else self.__dict__[key] for (key,value) in list(obj.__dict__.items())+list(self.__dict__.items()) }
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
		elif replace :
			try: 	self.__dict__ =  { key:value if not key in obj.__dict__ else obj.__dict__[key] for (key,value) in list(obj.__dict__.items())+list(self.__dict__.items()) }
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
	
		return None

	# ***** import from dict ***** #
	def import_data_dict(self, dictionary, force=False, replace=True):
		if force:
			try: self.__dict__ = dictionary.copy() 
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
		elif not replace :
			try: 	self.__dict__ =  { key:value if not key in self.__dict__ else self.__dict__[key] for (key,value) in list(dictionary.items())+list(self.__dict__.items()) }
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
		elif replace :
			try: 	self.__dict__ =  { key:value if not key in dictionary else dictionary[key] for (key,value) in list(dictionary.items())+list(self.__dict__.items()) }
			except: print(' ERROR  :: code X :: DATA.inject_data() :: can not inject data into {}'.format(str(obj)))
	
		return None

	def G(self, data=None, mu_range=np.arange(160, 280, 3), 
				sigma_range=np.arange(3, 30, 2), normalized_data=False, functional='gaussian' ):
		if not type(data) is np.ndarray:		 	data =self.X

		if type(mu_range) is list:					mu_range = np.array(mu_range)
		elif not type(mu_range) is np.ndarray: 		mu_range = range(0, data.shape[0])

		if type(sigma_range) is list:				sigma_range = np.array(sigma_range)
		elif not type(sigma_range) is np.ndarray: 	sigma_range = range(1, data.shape[0])
		
		n = data.shape[0] 

		# generate functional data base #
		if   functional=='gaussian':	base = np.array([ self.gaussian(mu, sigma, n) for mu in mu_range for sigma in sigma_range ])
		elif functional=='GIG':		base = np.array([ self.GIG(mu, sigma, n) for mu in mu_range for sigma in sigma_range ])
		elif functional=='Ray':		base = np.array([ self.Rayleigh(mu, sigma, n) for mu in mu_range for sigma in sigma_range ])
		base_parameters_list = np.array([ [mu, sigma] for mu in mu_range for sigma in sigma_range ]) 

		# Evaluate functional proy #
		if normalized_data:
			proy = base.dot( data/np.linalg.norm(data) )
			max_arg   = np.argmax(proy)
			proy = base.dot( data )
			max_value = np.max(proy)
		else:
			proy = base.dot( data )
			max_arg   = np.argmax(proy)
			max_value = np.max(proy)

		return max_arg, max_value, base_parameters_list[max_arg][0], base_parameters_list[max_arg][1]

	def G_vector_descompose(self, data, G_max, v=False):
		L1 = data.shape[0]
		coef = np.zeros((G_max, 4))
		resto = data
		for g in range(G_max):
			if v: print('Compleate {}%'.format(100*g/G_max) )
			coef[g, :] = G( resto )
			resto = resto - gaussian(coef[g, 2], coef[g, 3], L1)*coef[g, 1]
		return coef

	def G_matrix_descompose(self, data, G_max, v=False):
		L1, L2 = data.shape

		coef = np.zeros(( L1, G_max, 4))
		for l1 in range(L1):
			if v: print('Compleate {}%'.format(100*l1/L1) )
			coef[l1, :, :] = G_vector_descompose( data=data[l1, :], G_max=G_max )

		return coef

	def G_tensor_descompose(self, data, G_max, v=True):
		N, L1, L2 = data.shape

		coef = np.zeros((N, 2, G_max, 4))
		for n in range(N):
			if v: print('Compleate {}%'.format(100*n/N) )
			coef[n, :, :, :] = G_matrix_descompose( data=data[n, :, :], G_max=G_max )

		return coef



