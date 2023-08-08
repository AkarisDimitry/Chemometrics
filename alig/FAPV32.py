# FAPV32
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
					area={'mode':'sum'}, shape='gaussian', non_negativity=True, 
					linear_adjust=True, SD={'mode':'constant'},	
					calibration_interference=False, interference_elimination=False, include_diff={'mode': False},
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
								interference_elimination=interference_elimination, include_diff=include_diff,
								save=save)
		else:	
			print('WARNNING :: FAPV32.aling() :: There must be exactly 2 NON well aligned channel. You can correct this with: self.a = 2 or FAPV32.a = 2')
			print('WARNNING :: FAPV32.aling() :: FAP32 can detect this and assume self.a = 2. But it is recommended to properly correct this.')
			
			if v > 0: print('aling_FAPV21 mode.')
			# 3 order, 2 way, 1 aligned channel, 1 non aligned channel  
			self.aling_FAPV32(	D=D, S=S, Nc=Nc, Na=Na, Ni=Ni, Nt=Nt, 
								area=area, shape=shape, non_negativity=non_negativity, linear_adjust=linear_adjust, SD=SD,
								calibration_interference=calibration_interference, interference_elimination=interference_elimination, include_diff=include_diff,
								save=save)
		return self.Da, self.La, self.L # 
	
	def aling_FAPV32(self, D=None, X=None,S=None, Nc=None, Na=None, Ni=None, Nt=None, a=None, shape='gaussian', 
					area={'mode':'sum'}, non_negativity=True, linear_adjust=True, SD={'mode':'constant'},
					calibration_interference=False, interference_elimination=False, include_diff={'mode':False, }, 
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

		elif area['mode']=='proyection coeficient':
			#  evaluate proyection coef and gaussian parameters 
			coef = np.zeros((Nc+Nt, Na+Ni))
			Gcoef = np.zeros((Nc+Nt, Na+Ni, 2))
			for nc in range(Nc):
				for na in range(Na):
					CC = self.G( L[nc,na,:], mu_range=area['mu_range'], sigma_range=area['sigma_range'], functional=area['functional'])
					coef[nc, na] = CC[1]
					Gcoef[nc, na, :] = CC[2], CC[3] 
					if v >= 1: print('Gcoef: {:>10.3f} mead:{:.3f} sd:{:.3f} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
			A = coef
		
			#  evaluate all eigen functions (this step is not necesary ) 
			all_gauss_aling = np.zeros((Nc+Nt, Na+Ni, l3))

			for na in range(Na):
				for nc in range(Nc): 
					if area['functional'] == 'gaussian':
						all_gauss_aling[nc, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nc, na] 
					if area['functional'] == 'GIG':
						all_gauss_aling[nc, na, :] = self.GIG(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nc, na] 
					if area['functional'] == 'Rayleigh':
						all_gauss_aling[nc, na, :] = self.Rayleigh(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nc, na] 
					
			all_gauss = np.zeros((Nc+Nt, Na+Ni, l3))
			for na in range(Na):
				for nc in range(Nc): 
					if area['functional'] == 'gaussian':
						all_gauss[nc, na, :] = self.gaussian(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l3) * A[nc, na] 
					if area['functional'] == 'GIG':
						all_gauss[nc, na, :] = self.GIG(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l3) * A[nc, na] 
					if area['functional'] == 'Rayleigh':
						all_gauss[nc, na, :] = self.Rayleigh(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l3) * A[nc, na] 
					
		elif area['mode'] == 'quela':
			A, Gcoef = self.quela( 'calibration', area=area )
			
			#  evaluate all eigen functions (this step is not necesary ) 
			all_gauss_aling = np.zeros((Nc+Nt, Na+Ni, l3))
			for na in range(Na):
				for nc in range(Nc): 
					if area['functional'] == 'gaussian':
						all_gauss_aling[nc, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nc, na] 
					if area['functional'] == 'GIG':
						all_gauss_aling[nc, na, :] = self.GIG(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nc, na] 
					if area['functional'] == 'Rayleigh':
						all_gauss_aling[nc, na, :] = self.Rayleigh(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nc, na] 
					
			all_gauss = np.zeros((Nc+Nt, Na+Ni, l3))
			for na in range(Na):
				for nc in range(Nc): 
					if area['functional'] == 'gaussian':
						all_gauss[nc, na, :] = self.gaussian(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l3) * A[nc, na] 
					if area['functional'] == 'GIG':
						all_gauss[nc, na, :] = self.GIG(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l3) * A[nc, na] 
					if area['functional'] == 'Rayleigh':
						all_gauss[nc, na, :] = self.Rayleigh(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l3) * A[nc, na] 
					
		# ------ FUNCTIONAL ALIGNMENT ------ # Estimacion del tensor de datos alineado (self.Da)
		if v >= 1: print( ('3- Functional alignement (calibration samples)') )
		Lmax, Amax, Da, De = np.zeros((Na)), np.argmax(A, axis=0), np.zeros((N, l1, l2, l3)), np.zeros((N, l1, l2, l3))

		for na in range(Na): Lmax[na] = np.argmax(L, axis=2)[Amax[na]][na]

		if v >= 1: print( ('4- Recostructing data (calibration samples)') )

		if shape == 'original': # retains the original shape  
			L_mean_cal = np.mean(L, axis=0) # mean loading estimation for all samples
			for na in range(Na):
				Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), L_mean_cal[na, :]/np.linalg.norm(L_mean_cal[na, :]) , axes=0)
				for nc in range(Nc): Da[nc,:,:,:] += Ma*A[nc, na] 		# eigen matrix * eigen value

			# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
			La = np.zeros( (N, f, l3) )
			for na in range(Na):
				for nc in range(Nc): La[nc, na, :] = L_mean_cal[na, :]/np.linalg.norm(L_mean_cal[na, :])*A[nc, na] # eigen vector * eigen value

		elif shape == 'gaussian': # gaussian shape for eigen function  
			# - standard deviation estimation - #
			if SD['mode'] == 'mean': 								sigma = np.mean(Gcoef[:, :, 1], axis=0)
			if SD['mode'] == 'constant' and 'value' in SD: 			sigma = [SD['value'] ]*(Na+Ni)
			elif SD['mode'] == 'constant' and not 'value' in SD: 	sigma = [float(l3)/f/5 ]*(Na+Ni)

			for na in range(Na):
				Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), self.gaussian(Lmax[na], sigma[na], l3) , axes=0)
				for nc in range(Nc): Da[nc,:,:,:] += Ma*A[nc, na]

			# --- Estimation of originally NON-aligned channels (self.L) but already aligned (self.La).
			La = np.zeros( (N, f, l3) )
			for na in range(Na):
				for nc in range(Nc): La[nc, na, :] = self.gaussian(Lmax[na], sigma[na], l3)*A[nc, na]

		elif shape == 'GIG': # gaussian shape for eigen function  
			# - standard deviation estimation - #
			if SD['mode'] == 'mean': 								sigma = np.mean(Gcoef[:, :, 1], axis=0)
			if SD['mode'] == 'constant' and 'value' in SD: 			sigma = [SD['value'] ]*(Na+Ni)
			elif SD['mode'] == 'constant' and not 'value' in SD: 	sigma = [float(l3)/f/5 ]*(Na+Ni)

			for na in range(Na):
				Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), self.GIG(Lmax[na], sigma[na], l3) , axes=0)
				for nc in range(Nc): Da[nc,:,:,:] += Ma*A[nc, na]

			# --- Estimation of originally NON-aligned channels (self.L) but already aligned (self.La).
			La = np.zeros( (N, f, l3) )
			for na in range(Na):
				for nc in range(Nc): La[nc, na, :] = self.GIG(Lmax[na], sigma[na], l3)*A[nc, na]



	  	#  ------ ------ TEST SAMPLES ------ ------ # ------ ------ # ------ ------ # ------ ------ # ------ ------ #
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
			Lt=np.dot(Y.T, np.linalg.pinv(Z) )
			L = np.zeros( (Nt, f, l3) )
			for nt in range(Nt): L[nt,:,:] = Lt[nt*l3:(nt+1)*l3,:].T 

			# ------ AREAS ------ #
			if area['mode'] == 'sum':
				A = np.sum( L, axis=2 ) 

			if area['mode'] == 'max':
				A = np.max( L, axis=2 )  

			if area['mode']=='proyection coeficient':
				#  evaluate projection coef and Gaussian parameters
				coef = np.zeros((Nt, f))
				Gcoef = np.zeros((Nt, f, 2))
				for nc in range(Nt):
					for na in range(f):
						CC = self.G( L[nc,na,:], mu_range=area['mu_range'], sigma_range=area['sigma_range'], functional=area['functional'] )
						coef[nc, na] = CC[1]
						Gcoef[nc, na, :] = CC[2], CC[3] 
						if v >= 1: print('Gcoef: {:>10.3f} mead:{:.3f} sd:{:.3f} :: sample: {} factor: {:.3f} '.format(CC[1], CC[2], CC[3], nc, na))
				A = coef

				#  evaluate all eigen functions (this step is not necessary ) 
				for na in range(Na+Ni):
					for nt in range(Nt): 
						#all_gauss_aling[Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						if area['functional'] == 'gaussian':
							all_gauss_aling[Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						if area['functional'] == 'GIG':
							all_gauss_aling[Nc+nt, na, :] = self.GIG(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						if area['functional'] == 'Rayleigh':
							all_gauss_aling[Nc+nt, na, :] = self.Rayleigh(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						
				for na in range(Na+Ni):
					for nt in range(Nt): 
						#all_gauss[Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na] 
						if area['functional'] == 'gaussian':
							all_gauss[Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na]
						if area['functional'] == 'GIG':
							all_gauss[Nc+nt, na, :] = self.GIG(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na]
						if area['functional'] == 'Rayleigh':
							all_gauss[Nc+nt, na, :] = self.Rayleigh(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na]
						
			if area['mode']=='quela':
				A, Gcoef = self.quela( 'test', area=area )

				#  evaluate all eigen functions (this step is not necessary ) 
				for na in range(Na+Ni):
					for nt in range(Nt): 
						#all_gauss_aling[Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						if area['functional'] == 'gaussian':
							all_gauss_aling[Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						if area['functional'] == 'GIG':
							all_gauss_aling[Nc+nt, na, :] = self.GIG(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						if area['functional'] == 'Rayleigh':
							all_gauss_aling[Nc+nt, na, :] = self.Rayleigh(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l3) * A[nt, na] 
						
				for na in range(Na+Ni):
					for nt in range(Nt): 
						#all_gauss[Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na] 
						if area['functional'] == 'gaussian':
							all_gauss[Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na]
						if area['functional'] == 'GIG':
							all_gauss[Nc+nt, na, :] = self.GIG(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na]
						if area['functional'] == 'Rayleigh':
							all_gauss[Nc+nt, na, :] = self.Rayleigh(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l3) * A[nt, na]
						
			# eliminate the presence of interferents
			if interference_elimination == True:	A[:,Na: ] = 0

			# linear adjustmente
			if linear_adjust:
				pass # not implemented

			# --- FUNCTIONAL ALIGNMENT --- # Estimation of aligned data tensor (self.Da)
			if v >= 1: print( ('4b- Reconstructing data (test samples)') )

			if shape == 'original': # retains the original shape  
				L_mean_test = np.mean(L, axis=0) # mean loading estimation for all samples
				# L.shape = (Nt, Na, l3)
				# L_mean.shape = (Na, l3)
				# it is usually better to use L_mean_cal 
				for na in range(Na):
					Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), L_mean_cal[na, :]/np.linalg.norm(L_mean_cal[na, :]) , axes=0)
					for nt in range(Nt): Da[Nc+nt,:,:,:] += Ma*A[nt, na]		# eigen matrix * eigen value

				# --- Estimation of channels originally NOT aligned (self.L) but already aligned (self.La).
				for na in range(Na):
					for nt in range(Nt): 
						La[nt+Nc, na, :] = L_mean_cal[na, :]/np.linalg.norm(L_mean_cal[na, :])*A[nt, na] # eigen vector * eigen value 

			elif shape == 'gaussian': # Gaussian shape for eigen function  
				# --- ALINEADO FUNCIONAL --- # Estimacion del tensor de datos alineado (self.Da)
				for na in range(Na):
					Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), self.gaussian(Lmax[na], sigma[na], l3) , axes=0)
					for nt in range(Nt): Da[Nc+nt,:,:,:] += Ma*A[nt, na]

				# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
				for na in range(Na):
					for nt in range(Nt): La[nt+Nc, na, :] = self.gaussian(Lmax[na], sigma[na], l3)*A[nt, na]

			elif shape == 'GIG': # Gaussian shape for eigen function  
				# --- ALINEADO FUNCIONAL --- # Estimacion del tensor de datos alineado (self.Da)
				for na in range(Na):
					Ma = np.tensordot( np.tensordot(S[0][na, :], S[1][na, :], axes=0), self.GIG(Lmax[na], sigma[na], l3) , axes=0)
					for nt in range(Nt): Da[Nc+nt,:,:,:] += Ma*A[nt, na]

				# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
				for na in range(Na):
					for nt in range(Nt): La[nt+Nc, na, :] = self.GIG(Lmax[na], sigma[na], l3)*A[nt, na]

			# --- Diference  estimation vs  nominal  --- # 	
			# ( could be use to recontructed proyections or estimate ERROR)
			if 'mode' in include_diff and include_diff['mode'] == 'direc diference':
				tensor_distance = np.zeros_like( Da ) # in tensor space ( it is not a distance ) 
				tensor_distance[:] = D - Da
				tensor_distance[tensor_distance < 0] = 0
				Da = Da+tensor_distance

			elif 'mode' in include_diff and include_diff['mode'] == 'mean diference':
				tensor_distance = np.zeros_like( Da ) # in tensor space ( it is not a distance ) 
				tensor_distance[:] = D - Da

				if 'components' in include_diff and include_diff['components'] == 'all':
					tensor_mean_distance = np.mean( tensor_distance[:,:,:,:], axis=0 )		
					for nt in range(Nt): Da[Nc+nt,:,:,:] += tensor_mean_distance
				elif 'components' in include_diff and include_diff['components'] == 'test':
					tensor_mean_distance = np.mean( tensor_distance[Nc:,:,:,:], axis=0 )
				elif 'components' in include_diff and include_diff['components'] == 'calibration':
					tensor_mean_distance = np.mean( tensor_distance[:Nc,:,:,:], axis=0 )

				if 'afects' in include_diff and include_diff['afects'] == 'all':
					for n in range(Nt+Nc): Da[n,:,:,:] += tensor_mean_distance
				elif 'afects' in include_diff and include_diff['afects'] == 'test':
					for nt in range(Nt): Da[Nc+nt,:,:,:] += tensor_mean_distance
				elif 'afects' in include_diff and include_diff['afects'] == 'calibration':
					for nc in range(Nc): Da[nc,:,:,:] += tensor_mean_distance

			# --- Fully recontructed proyections --- # 
			# --- Z --- #
			Ze = np.zeros( (f, l1, l2) )
			for n in range(f):	Ze[n,:,:] = np.tensordot( S[0][n,:], S[1][n,:] , axes=0)
			Z = Ze[:,0,:]
			for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

			# --- Y --- #
			Ye = D[0,:,:,:]
			for n in range(1,Nc+Nt): Ye = np.concatenate( (Ye ,  D[n,:,:,:]), axis=2)
			Y = Ye[0,:,:]
			for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

			# --- X(Lc) --- #
			Lc=np.dot(Y.T, np.linalg.pinv(Z) )
			L = np.zeros( (Nt+Nc, f, l3) )
			for n in range(Nt+Nc): L[n,:,:] = Lc[n*l3:(n+1)*l3,:].T 


		# Devolvemos (0-self.Da)Tensor de datos alineado (1-self.La)Perfiles cromatograficos alineados (2-self.L)Estimacion original de los perfiles cromatograficos
		if save: 
			print('Saving results...')
			try:
				self.all_gauss_aling = all_gauss_aling
				self.all_gauss = all_gauss
			except:
				self.all_gauss_aling = None
				self.all_gauss = None

			try:
				self.L_mean_cal = L_mean_cal
				self.L_mean_test = L_mean_test
			except: 
				self.L_mean_cal = None
				self.L_mean_test = None

			self.A = A
			self.La = La
			self.Da = Da
			self.L = L
			self.Lc=Lc # np.dot(np.array(reduce(lambda x, y:  np.concatenate( (x ,  y), axis=1), D[:,:,:] )).T, np.linalg.pinv(S[0][:, :]) )

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
	def quela(self, mode, area, functional='gaussian', v=1, plot=False):
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
					if v >= 1: print('Gcoef: {:>10.3f} mead:{:.3f} sd:{:.3f} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))

				if plot:
					plt.plot( L[nc,0,:].T, 'r') 
					plt.plot( self.gaussian(Gcoef[nc, 0, 0] , Gcoef[nc, 0, 1] , l3)*coef[nc, 0], 'k' ) 
			if plot: plt.show() # plot partial results from 

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
					if v >= 1: print('Gcoef: {:>10.3f} mead:{:.3f} sd:{:.3f} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
				
				if plot:
					plt.plot( L[nc,0,:].T, 'r')
					plt.plot( self.gaussian(Gcoef[nc, 0, 0] , Gcoef[nc, 0, 1] , l3)*coef[nc, 0], 'k' )
			if plot: plt.show()

			A = coef
		return A, Gcoef

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

	def plot_alignment(self, Na=None, Ni=None, colors=None):
		Na = Na if type(Na) in [np.ndarray, int, float] else self.Na
		Ni = Ni if type(Ni) in [np.ndarray, int, float] else self.Ni
		try: self.colors
		except: self.colors = None

		colors = self.colors if type(colors) == type(None) else colors
		colors = [  (0.8, 0.4, 0.4), 
					(0.4, 0.8, 0.4), 
					(0.4, 0.4, 0.8), 	 ] if type(colors) == type(None) else colors


		for f in range( Na + Ni ):
			fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
			# ============ Estimated Non aligned channel  ============ #
			ax1.plot( [0,0], color=(0.3,0.3,0.3),  label='component:{}, estimated self-vector'.format(f+1) )
			ax1.plot( self.L[:,f,:].T, color=(0.3,0.3,0.3), alpha=0.4, lw=2 )

			ax1.plot( [0,0], color=colors[f], label='component:{}, estimated self-function'.format(f+1) )
			ax1.plot( self.all_gauss[:,f,:].T, color=colors[f], alpha=0.6, lw=2 )
			#ax1.set_xlabel('Non-aligned channel - Sensors')
			ax1.set_ylabel('Intensity')
			ax1.grid(linestyle=':', linewidth=0.7)
			ax1.set_title('Estimated Non aligned channel (grey) vs Functional estimacion (color)')
			ax1.legend()

			# ============ Estimated Non aligned channel  ============ #
			ax2.plot( [0,0], color=(0.3,0.3,0.3),  label='component:{}, NON-aligned self-function'.format(f+1) )
			ax2.plot( self.all_gauss[:,f,:].T, color=(0.3,0.3,0.3), alpha=0.4, lw=2 )

			ax2.plot( [0,0], color=colors[f], label='component:{}, WELL-aligned self-function'.format(f+1) )
			ax2.plot( self.all_gauss_aling[:,f,:].T, color=colors[f], alpha=0.6, lw=2 )
			#ax2.set_xlabel('Originally Non-aligned channel - Sensors')
			ax2.set_ylabel('Intensity')
			ax2.grid(linestyle=':', linewidth=0.7)
			ax2.set_title('NON-aligned self-function (grey) vs WELL-aligned self-function (color)')
			ax2.legend()

			# ============ Estimated Non aligned channel  ============ #
			ax3.plot( [0,0], color=(0.3,0.3,0.3),  label='component:{}, estimated error per sensors'.format(f+1) )
			ax3.plot( self.L[:,f,:].T - self.all_gauss[:,f,:].T, color=(0.3,0.3,0.3), alpha=0.4, lw=2 )

			#ax3.set_xlabel('Non-aligned channel - Sensors')
			ax3.set_ylabel('Intensity')
			ax3.grid(linestyle=':', linewidth=0.7)
			ax3.set_title('Estimation absulute error per sensors (grey)')
			ax3.legend()

		return None

	def complexity_analysis( self, 	D=None, X=None, S=None, Nc=None, Na=None, Ni=None, a=None,
									save=True, v=True, include_test=True, compress_nonalifned_channel=False ):

		def self_vector_estimation():
			Ze = np.zeros( (Na, l1, l2) )
			for n in range(Na):	Ze[n,:,:] = np.tensordot( S[0][n,:], S[1][n,:] , axes=0)
			Z = Ze[:,0,:]
			for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

		  	# --- Y --- #
			Ye = D[0,:,:,:]
			for n in range(1, N): Ye = np.concatenate( (Ye ,  D[n,:,:,:]), axis=2)
			Y = Ye[0,:,:]
			for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

		  	# --- X(Lc) --- #
			Lc=np.dot(Y.T, np.linalg.pinv(Z) )
			L = np.zeros( (N, Na, l3) )
			for n in range(N): L[n,:,:] = Lc[n*l3:(n+1)*l3,:].T

			return L

		if v: print('=== Complexity analysis ===')

		D = D if type(D) is np.ndarray else self.D
		X = X if type(X) is np.ndarray else self.X

		if type(D) is np.ndarray and type(X) is np.ndarray and not (X==D).all(): D = X
		S = S if type(S) is list else self.S

		Nc = Nc if type(Nc) in [np.ndarray, int, float] else self.Nc
		Na = Na if type(Na) in [np.ndarray, int, float] else self.Na
		Ni = Ni if type(Ni) in [np.ndarray, int, float] else self.Ni
		a  = a  if type(a ) in [np.ndarray, int, float] else 2

		N, l1, l2, l3  =  D.shape
		N = Nc if not include_test else N

		L = L if type(self.L) == np.ndarray and self.L.shape == np.zeros((N, Na, l3)) else self_vector_estimation() # np.zeros( (N, Na, l3) )
		Da = np.zeros((N, l1, l2, l3))

		complexity_warping = np.zeros( (N, N, Na) )
		for f in range(Na):
			var = []
			for n1 in range(N):
				tensor_fn1 = np.tensordot( np.tensordot(S[0][f, :], S[1][f, :], axes=0), L[n1 ,f, :]/np.linalg.norm(L[n1 ,f, :]) , axes=0)
				if compress_nonalifned_channel: tensor_fn1 = np.sum(tensor_fn1, axis=-1)
				tensor_fn1 = tensor_fn1/np.linalg.norm(tensor_fn1)

				for n2 in range(N):
					tensor_fn2 = np.tensordot( np.tensordot(S[0][f, :], S[1][f, :], axes=0), L[n2 ,f, :]/np.linalg.norm(L[n2 ,f, :]) , axes=0)
					if compress_nonalifned_channel: tensor_fn2 = np.sum(tensor_fn2, axis=-1)
					tensor_fn2 = tensor_fn2/np.linalg.norm(tensor_fn2)

					complexity_warping[n1, n2, f] =  np.sum(tensor_fn1*tensor_fn2)

					#if v: print( 'sample {} sample {} factor {} {}'.format(n1, n2, f,  np.sum(tensor_fn1*tensor_fn2)  ))
		

		complexity_overlap = np.zeros( (Na, Na, N) )
		for n in range(N):
			for f1 in range(Na):
				tensor_f1n = np.tensordot( np.tensordot(S[0][f1, :], S[1][f1, :], axes=0), L[n ,f1, :]/np.linalg.norm(L[n ,f1, :]) , axes=0)
				tensor_f1n = tensor_f1n/np.linalg.norm(tensor_f1n)
				for f2 in range(Na):
					tensor_f2n = np.tensordot( np.tensordot(S[0][f2, :], S[1][f2, :], axes=0), L[n ,f2, :]/np.linalg.norm(L[n ,f2, :]) , axes=0)
					tensor_f2n = tensor_f2n/np.linalg.norm(tensor_f2n)

					complexity_overlap[f1, f2, n] = np.sum(tensor_f1n*tensor_f2n)
					#if v: print( 'sample {} factor {} factor {} {}'.format(n, f1, f2, np.sum(tensor_f2n*tensor_f2n) ))
		
		print( 'Cal-Cal samples overlapping', np.mean(complexity_warping[:Nc, :Nc, 0]) )
		print( 'All samples overlapping',np.mean(complexity_warping[:, :, 0]) )
		print( 'Cal-Test samples overlapping',np.mean(complexity_warping[14:, :14, 0]) )

		'''
		# residual overlaping estimation #
		data = np.zeros((N))
		for n in range(N):
			diff1 = np.array(self.L[n,0,:].T - self.all_gauss[n,0,:].T) 
			diff1[diff1<0] = 0
			diff1 = diff1/np.linalg.norm(diff1)
			#print(np.linalg.norm(diff1))
			diff2 = np.array(self.L[n,1,:].T - self.all_gauss[n,1,:].T)
			diff2[diff2<0] = 0
			diff2 = diff2/np.linalg.norm(diff2)

			vector = self.all_gauss[n,1,:]/np.linalg.norm(self.all_gauss[n,1,:])

			data[n] =  np.sum(vector * (diff1+diff2)/2 )  
 		
			plt.figure(12), plt.plot(vector, 'b', alpha=0.4)
			plt.figure(12), plt.plot(diff1, c=(1,0,0), alpha=0.4)
			plt.figure(12), plt.plot(diff2, c=(0.7,0.7,0), alpha=0.4)

		#print(data, np.mean(data))
		plt.figure(22), plt.plot(data, 'r', alpha=0.7)
		plt.matshow( complexity_warping[:,:,0] )
		plt.show()
		'''

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

