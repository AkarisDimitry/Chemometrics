# *** warning supresion
import warnings
warnings.filterwarnings("ignore")
# *** numeric libraries
import numpy as np
# *** graph libraries
import matplotlib.pyplot as plt

class FAPV21(object):
	def __init__(self, X=None, D=None, L=None, La=None, f=None, A=None, a=1,
				Y=None, y=None, S=None,	N=None, Nc=None, Na=None, Ni=None, Nt=None ):

		self.D = D 		# Complete data set #
		self.X = X 		# Complete data set #
		
		self.Y = Y 		# Complete response values #
		self.y = y 		# Complete response values #
		
		self.S = S 		# well aligned ways (initial stimations) #

		
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
					area='gaussian_coeficient', non_negativity=True, linear_adjust=True,
					mu_range=None, sigma_range=None, save=True, v=1
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

		mu_range = np.array(mu_range) if type(mu_range) is list else self.mu_range
		mu_range = mu_range if type(mu_range) is np.ndarray else self.mu_range

		sigma_range = np.array(sigma_range) if type(sigma_range) is np.ndarray else self.sigma_range
		sigma_range = sigma_range if type(sigma_range) is np.ndarray else self.sigma_range

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

			self.mu_range = mu_range 
			self.sigma_range = sigma_range

		if self.a == 1: 
			if v > 0: print('aling_FAPV21 mode.')
			# 3 order, 2 way, 1 aligned channel, 1 non aligned channel  
			self.aling_FAPV21(	D=D, S=S, Nc=Nc, Na=Na, Ni=Ni, Nt=Nt,
								area=area, non_negativity=non_negativity, linear_adjust=linear_adjust,
								mu_range=mu_range, sigma_range=sigma_range, save=save)

		return self.Da, self.La, self.L # 
		
	def aling_FAPV21(self, D=None, S=None, Nc=None, Na=None, Ni=None, Nt=None, v=1, 
					area='gaussian_coeficient', non_negativity=True, linear_adjust=True,
					mu_range=None, sigma_range=None, save=True
					): 		# FPAV for 3 order, 2 way, 1 aligned channel, 1non aligned channel 
		# ----- Alignment algorithm for 3rd order data with 2 aligned channels ----- #
		# 
		# D 		:	N-MAT 		:		N-array/Matrix non-aligned data D => [N,l1,l2]
		# S 		:	N-MAT 		:		N-array/Matrix well aligned channels  S => [[self.f,l1],[self.f,l2]] 
		# Nc		:	INT 		: 		number of calibration samples.
		# Na		:	INT 		: 		number of analites.
		# Ni		:	INT 		: 		number of interferentes.
		# v 		: 	BOOL		: 		verboisity
		# area 		:   str 		: 		area preserving mode {'gaussian', 'max', 'sum'}
		# non_negativity :	BOOL  	: 		NON-negativity constrain
		# mu_range	:	N-array 	: 		this parameter consists of the exploration range of mu (of the functions of the functional space)
		#									eg: mu_range=np.arange(150, 300, 3)
		# sigma_range : N-array		; 		this parameter consists of the exploration range of sigma (of the functions of the functional space)
		#									eg: sigma_range=np.arange(3, 30, 1)
		# save 		: 	BOOL		:		this save the
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
		if not (X==D).all(): D = X

		S = S if type(S) is list else self.S

		Nc = Nc if type(Nc) in [np.ndarray, int, float] else self.Nc
		Na = Na if type(Na) in [np.ndarray, int, float] else self.Na
		Ni = Ni if type(Ni) in [np.ndarray, int, float] else self.Ni
		a  = a  if type(a ) in [np.ndarray, int, float] else self.a

		mu_range = np.array(mu_range) if type(mu_range) is list else self.mu_range
		mu_range = mu_range if type(mu_range) is np.ndarray else self.mu_range

		sigma_range = np.array(sigma_range) if type(sigma_range) is np.ndarray else self.sigma_range
		sigma_range = sigma_range if type(sigma_range) is np.ndarray else self.sigma_range

		non_negativity = non_negativity if type(non_negativity) == bool else True 
		linear_adjust = linear_adjust if type(linear_adjust) == bool else False

		N, l1, l2  =  D.shape
		self.N = N if save else self.N 
		Nt = Nt if type(Nt) in [int, float, np.ndarray] else N-Nc
		Nt, f = N-Nc, Na+Ni

		# --- STEPS --- #
		# (0) estimation of well aligned channels (commonly spectral channels)
		# (1) Estimate non-spectral channels
		# (2) Align non-spectral channels
		# (3) Rebuild the samples
		# (4) Postprocessing

			# ------ ------ MUESTRAS DE CALIBRADO ------ ------ #
		# --- Z --- #
		if v >= 1: print( ('1- Inicializing matrix') )
		Ze = S[0][:Na, :]
		Z = Ze

	  	# --- Y --- #
		Ye = D[0,:,:]
		for n in range(1, Nc): Ye = np.concatenate( (Ye ,  D[n,:,:]), axis=1)
		Y = Ye

	  	# --- X(Lc) --- #
		if v >= 1: print( ('2- Estimation of non-aligned channel') )
		Lc=np.dot(Y.T, np.linalg.pinv(Z) )

		if non_negativity: 	Lc = np.where(Lc>0, Lc, 0) 			

		L = np.zeros( (Nc, Na, l2) )
		for nc in range(Nc): L[nc,:,:] = Lc[nc*l2:(nc+1)*l2,:].T

		# ------ AREAS ------ #
		if area == 'sum':
			A = np.sum( L, axis=2 ) # [N, self.f]

		if area == 'max':
			A = np.max( L, axis=2 )  

		if area=='gaussian_coeficient':
			#  evaluate proyection coef and gaussian parameters 
			coef = np.zeros((Nc+Nt, Na+Ni))
			Gcoef = np.zeros((Nc+Nt, Na+Ni, 2))
			for nc in range(Nc):
				for na in range(Na):
					CC = self.G( L[nc,na,:], mu_range=mu_range, sigma_range=sigma_range)
					coef[nc, na] = CC[1]
					Gcoef[nc, na, :] = CC[2], CC[3] 
					if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
			A = coef

		if area=='gaussian_coeficient':
			#  evaluate all eigen functions (this step is not necesary ) 
			all_gauss_aling = np.zeros((Nc+Nt, Na+Ni, l2))
			for na in range(Na):
				for nc in range(Nc): 
					all_gauss_aling[nc, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l2) * A[nc, na] 

			all_gauss = np.zeros((Nc+Nt, Na+Ni, l2))
			for na in range(Na):
				for nc in range(Nc): 
					all_gauss[nc, na, :] = self.gaussian(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l2) * A[nc, na] 

			# ------ FUNCTIONAL ALIGNMENT ------ # Estimacion del tensor de datos alineado (self.Da)
		if v >= 1: print( ('3- Functional alignement (calibration samples)') )
		Lmax, Amax, Da, sigma = np.zeros((Na)), np.argmax(A, axis=0), np.zeros((N, l1, l2)), l2/f/30
		for na in range(Na): Lmax[na] = np.argmax(L, axis=2)[Amax[na]][na]

		if v >= 1: print( ('4- Recostructing data (calibration samples)') )
		for na in range(Na):
			Ma = np.tensordot(S[0][na, :], self.gaussian(Lmax[na], sigma, l2) , axes=0)
			for nc in range(Nc): Da[nc,:,:] += Ma*A[nc, na]

		# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		La = np.zeros( (N, f, l2) )
		for na in range(Na):
			for nc in range(Nc): La[nc, na, :] = self.gaussian(Lmax[na], sigma, l2)*A[nc, na]

	  	#  ------ ------ TEST SAMPLES ------ ------ #
		if Nt > 0:
			if v >= 1: print( ('3b- Functional alignement (test samples)') )
			# --- Z --- #
			Ze = S[0]
			Z = Ze

			# --- Y --- #
			Ye = D[Nc,:,:]
			for n in range(1,Nt): Ye = np.concatenate( (Ye ,  D[Nc+n,:,:]), axis=1)
			Y = Ye

			# --- X(Lc) --- #
			Lc= np.dot(Y.T, np.linalg.pinv(Z) )
			if non_negativity: 	Lc = np.where(Lc>0, Lc, 0) 

			L = np.zeros( (Nt, f, l2) )
			for nt in range(Nt): L[nt,:,:] = Lc[nt*l2:(nt+1)*l2,:].T 
		
			# ------ AREAS ------ #
			if area == 'sum':
				A = np.sum( L, axis=2 ) 

			if area == 'max':
				A = np.max( L, axis=2 )  

			if area=='gaussian_coeficient':
				#  evaluate proyection coef and gaussian parameters
				coef = np.zeros((Nt, f))
				Gcoef = np.zeros((Nt, f, 2))
				for nc in range(Nt):
					for na in range(f):
						CC = self.G( L[nc,na,:], mu_range=mu_range, sigma_range=sigma_range )
						coef[nc, na] = CC[1]
						Gcoef[nc, na, :] = CC[2], CC[3] 
						if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
				A = coef

			if linear_adjust:
				pass # not implemented

			if area=='gaussian_coeficient':
				#  evaluate all eigen functions (this step is not necesary ) 
				for na in range(Na+Ni):
					for nt in range(Nt): 
						all_gauss_aling[Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l2) * A[nt, na] 

				for na in range(Na+Ni):
					for nt in range(Nt): 
						all_gauss[Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l2) * A[nt, na] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			# --- FUNCTIONAL ALIGNMENT --- # Estimation of aligned data tensor (self.Da)
			if v >= 1: print( ('4b- Recostructing data (test samples)') )
			for na in range(Na):
				Ma = np.tensordot( S[0][na, :], self.gaussian(Lmax[na], sigma, l2) , axes=0)
				for nt in range(Nt): Da[Nc+nt,:,:] += Ma*A[nt, na]

		# --- Estimation of channels originally NOT aligned (self.L) but already aligned (self.La).
		for na in range(Na):
			for nt in range(Nt): 
				La[nt+Nc, na, :] = self.gaussian(Lmax[na], sigma, l2)*A[nt, na]
				La[nt+Nc, na, :] = self.gaussian(Gcoef[nt, na, 0], sigma, l2)*A[nt, na]

		if save: 
			print('Saving results...')
			self.all_gauss_aling = all_gauss_aling
			self.all_gauss = all_gauss
			self.A = A
			self.La = La
			self.Da = Da
			self.L = L

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
		
	def gaussian(self, mu, sigma, n, ):
			return np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2) / (np.linalg.norm(np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)))

	def Lorentz(self,n,a,m):
		return (a*1.0/(np.pi*((np.arange(0, n, dtype=np.float32)-m))**2+a**2))

	def plot(self, names):
		color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', ]

		fig_all, ax_all = [], []

		for n in range(self.Na):
			fig, ax = plt.subplots()
			fig_all.append(fig)
			ax_all.append(ax)

		for n in range(self.Na):
			fig, ax = fig_all[n], ax_all[n]
			
			dat = self.all_gauss[:,n,:]
			color = [ 0.7 + 0.3*np.max(dat[n,:])/np.max(dat) for n in range(dat.shape[0])]
			ax.plot( np.linspace(0,6.5,391), dat[0,:], color=(color[0], 0.5, 0.3 ), lw=1.5, alpha=0.6, label=names[n] )
			for n in range(1, dat.shape[0]):
				ax.plot( np.linspace(0,6.5,391), dat[n,:], color=(color[n], 0.5, 0.3 ), lw=1.5, alpha=0.6 )

			ax.set_xlabel('Elution time (min)', fontsize=12)
			ax.set_ylabel('Absorbance (a.u.)', fontsize=12)

			ax.spines["right"].set_linewidth(2)
			ax.spines["right"].set_color("#333333")
			ax.spines["left"].set_linewidth(2)
			ax.spines["left"].set_color("#333333")
			ax.spines["top"].set_linewidth(2)
			ax.spines["top"].set_color("#333333")
			ax.spines["bottom"].set_linewidth(2)
			ax.spines["bottom"].set_color("#333333")

			legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
			legend.get_frame().set_facecolor('#FFFFFF')

	def plot_spectra(self, names=None, colors=None):
		try:
			if   colors == None:			color = self.colors
			elif colors == 'vainilla':		color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', ]
			else:							color = colors
		except:		print('ERROR :: code 002c FAPV21.plot_spectra() :: Can not set colors')

		try:		fig, ax = plt.subplots()
		except:		print('ERROR :: code 002c FAPV21.plot_spectra() :: Can not make axes')

		try:
			for n in range(self.Na):
				if names != None:	ax.plot( self.S[0][n,:].T, '-o', color=color[n], lw=1.5, alpha=0.6, ms=3, label=names[n])
				else:				ax.plot( self.S[0][n,:].T, '-o', color=color[n], lw=1.5, alpha=0.6, ms=3 )
		except:		print('ERROR :: code 002c FAPV21.plot_spectra() :: Can not PLOT spectra')

		try:
			ax.set_xlabel('variable', fontsize=12);		ax.set_ylabel('Absorbance (a.u.)', fontsize=12)
			ax.spines["right"].set_linewidth(2); 		ax.spines["right"].set_color("#333333")
			ax.spines["left"].set_linewidth(2); 		ax.spines["left"].set_color("#333333")
			ax.spines["top"].set_linewidth(2); 			ax.spines["top"].set_color("#333333")
			ax.spines["bottom"].set_linewidth(2); 		ax.spines["bottom"].set_color("#333333")
		except:		print('ERROR :: code 002c FAPV21.plot_spectra() :: Can not configure axis')


		try:
			if names != None:
				legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
				legend.get_frame().set_facecolor('#FFFFFF')
		except:		print('ERROR :: code 002c FAPV21.plot_spectra() :: Can not configure legend')

	def summary(self,):
		print(' *** Sensors ***')
		for i, n in enumerate(self.D.shape):		
			if i == 0: print('  *  Loaded data has '+str(n)+' samples')
			elif i != 3: print('  *  (aligned) Channel '+str(i)+' has '+str(n)+' sensors')
			elif i == 3: print('  *  (NON-aligned) Channel '+str(i)+' has '+str(n)+' sensors')

	def load_fromDATA(self, data=None):
		try: self.__dict__ = data.__dict__.copy() 
		except: pass

	def G(self, data=None, mu_range=np.arange(160, 280, 3), sigma_range=np.arange(3, 30, 2) ):
		if not type(data) is np.ndarray:		 	data =self.X

		if type(mu_range) is list:					mu_range = np.array(mu_range)
		elif not type(mu_range) is np.ndarray: 		mu_range = range(0, data.shape[0])

		if type(sigma_range) is list:				sigma_range = np.array(sigma_range)
		elif not type(sigma_range) is np.ndarray: 	sigma_range = range(1, data.shape[0])
		
		n = data.shape[0]

		base = np.array([ self.gaussian(mu, sigma, n) for mu in mu_range for sigma in sigma_range ])
		base_parameters_list = np.array([ [mu, sigma] for mu in mu_range for sigma in sigma_range ]) 

		proy = base.dot(data)
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

'''
eg.
# - data information - #
data.Ni = int 	#
data.Na = int 	#-
data.Nc = int
data.N = int 
data.Nt = int  
data.a = int
data.Y = int 

# - alig FPAV - #
FAPV21 = FAPV21()
data_O.inyect_data(FPAV21)
data_O.aling_model = FPAV21

FPAV21.S = [np.array([data1[:,1], data2[:,1]])]
Da, a, b = FPAV21.aling( area='gaussian_coeficient', mu_range=np.arange(150, 300, 3), sigma_range=np.arange(3, 30, 1))
data_O.D = Da
data_O.X = Da
'''

