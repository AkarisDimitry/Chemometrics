# *** warning supresion *** #
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries *** #
import numpy as np #pip3 install numpy

# *** graph libraries *** #
try:
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pl
except: 
	print('WAENNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: ( pip3 install matplotlib )')
	
class MCR(object): # generador de datos
	def __init__(self, X=None, D=None, S=None, 
					C=None, A=None,
					Y=None, y=None,	
					N=None, Nc=None, Nt=None,
					f=None, Na=None, Ni=None):

		try: self.X = np.array(X)	
		except: print('WARNING :: code 400 MCR.MCR() :: can not load X data in to self.X')
		try: self.D = np.array(D)
		except: print('WARNING :: code 400 MCR.MCR() :: can not load X data in to self.D')

		if type(self.X) is np.ndarray and type(self.D) is np.ndarray and  self.X.shape != self.D.shape:
			self.D = self.X
			print('WARNING :: code 401 MCR.MCR() :: self.X.shape({}) != self.D.shape({})'.format(self.X.shape, self.D.shape) ) 
			print('WARNING :: making self.D = self.X')

		self.Dc = None # - This is a compress self.D in to 1 matrix - # (MCR can only process matrix)
			
		self.S = np.array(S)
		self.C = np.array(C)

		self.A = A

		self.Y = Y
		self.y = y

		self.MSE = None

		self.N  = N
		self.Nc = Nc
		self.Nt = Nt

		self.f  = f
		self.Na = Na
		self.Ni = Ni

		self.L = []
		self.loadings = [] 	 		# loadings 
		self.Nloadings = []  		# normalized loadings 
		self.model_mse = None 		# MSE error 

		self.constraints = None

		self.font = {
				'family': 'serif',
        		'color':  (0.2, 0.2, 0.2, 1.0),
        		'weight': 'normal',
        		'size': 16,
        		}

	def compress_2WAY(self, X=None, D=None, save=True):
		'''
		Fold X or D 3-th tensor into a 2nd order (1-WAY) matrix
		X.shape = (N0, n1, n2) 
			(1)	X 	:	array 	: 	Tensor con todos los datos.
			(2)	D 	:	array 	: 	Tensor con todos los datos.
		 	(3) save:	bool 	: 	save = {False, True}
		return Dc
			(0) Dc 	:	array 	: 	folded tensor 
			Dc.shape = (N0*n2, n1)
		'''

		# step 0 | variable setter		
		if   type(X) is list: 							X = np.array(X)
		elif type(X) is np.ndarray:						pass
		elif type(self.X) is np.ndarray:				X = np.array(self.X)

		if type(X) == type(None) and type(D) is list: 	X = np.array(D)
		elif type(X) is np.ndarray:						pass

		# step 1 | compresion 
		Dc =  X[0,:,:] 
		for n in range(1, X.shape[0]): Dc = np.concatenate( (Dc,  X[n,:,:]), axis = 1)

		# step 2 | save
		if save: self.Dc = Dc
		else: pass
		
		return Dc

	def compress_3WAY(self, X=None, D=None, save=True):
		'''
		Fold X or D 4-th order tensor into a 2nd order (1-WAY) matrix
		X.shape = (N0, n1, n2, n3) 
			(1)	X 	:	array 	: 	Tensor con todos los datos.
			(2)	D 	:	array 	: 	Tensor con todos los datos.
		 	(3) save:	bool 	: 	save = {False, True}
		return Dc
			(0) Dc 	:	array 	: 	folded tensor 
			Dc.shape = (N0*n3, n1.n2)
		'''

		# step 0 | variable setter
		if   type(X) is list: 							X = np.array(X)
		elif type(X) is np.ndarray:						pass
		elif type(self.X) is np.ndarray:				X = np.array(self.X)

		if type(X) == type(None) and type(D) is list: 	X = np.array(D)
		elif type(X) is np.ndarray:						pass

		N = X.shape[0]

		# step 1 | compresion 
		De =  X[0,:,:,:] 
		for n in range(1, N):		De = np.concatenate( (De ,  X[n,:,:,:]), axis = 2 )
		Dc =  De[0,:,:]
		for n in range(1, De.shape[0]):	Dc = np.concatenate( (Dc ,  De[n,:,:]), axis = 0)

		# step 2 | save 
		if save: self.Dc = Dc
		else: pass
		
		return Dc

	def compress_nWAY(self, X=None, D=None, save=True):
		'''
		Fold X or D n-th order tensor into a 2nd order (1-WAY) matrix
		X.shape = (N0, n_1, n_2, n_3 ... n_n) 
			(1)	X 	:	array 	: 	Tensor con todos los datos.
			(2)	D 	:	array 	: 	Tensor con todos los datos.
		 	(3) save:	bool 	: 	save = {False, True}
		return Dc
			(0) Dc 	:	array 	: 	folded tensor 
			Dc.shape = (N0*n_n, n_1.n_2 ... )
		'''

		# step 0 | variable setter
		if   type(X) is list: 							X = np.array(X)
		elif type(X) is np.ndarray:						pass
		elif type(self.X) is np.ndarray:				X = np.array(self.X)

		if type(X) == type(None) and type(D) is list: 	X = np.array(D)
		elif type(X) is np.ndarray:						pass

		order = len(X.shape) # - Total data order - #
	
		N = X.shape[0]

		# step 1 | compresion 
		De =  X[0,:] # Calibracion busco espectros limpios.
		for n in range(1, N):				De = np.concatenate( (De ,  X[n,:]), axis = (order-2))
		Dc =  De[0,:]
		for n in range(1, De.shape[0]):		Dc = np.concatenate( (Dc ,  De[n,:]), axis = 0)

		# step 2 | save 
		if save: self.Dc = Dc
		else: pass
		
		return Dc

	def train(self, Dc=None, S=None, constraints={'Normalization':['S'], 'non-negativity':['all']}, 
					inicializacion='random', max_iter=200, save=True):
		# MCR cannon. Datos de tercer  orden. 2 canales NO alineados y 1 canal alineado. 
		# D = [Nc, l1, l2] ; l1 : alineados  	l2 l3 : NO alineados
		# 	(1) D			:	MAT 	: Matrix con todos los datos. 										X = [l1, l2]
		# 	(2) S			:	MAT 	: vector con las estimaciones de los espectros - canal alineado. 	S = [f, l1]
		# 	(3) max_iter	:	INT 	: Numero maximo de iteraciones

		# --- step (0) Check tensor integrity --- Tensor len(shape) must be 2
		Dc = self.Dc if type(Dc) == type(None) else Dc
		if type(Dc) == type(None): 	
			if type(self.D) == type(None): 	
				print('ERROR :: Null extended tensor :: Dc = None')
			elif len(self.D.shape) == 3:
				Dc = self.compress_2WAY()
			elif len(self.D.shape) == 4:
				Dc = self.compress_3WAY()
			elif len(self.D.shape) > 4:
				Dc = self.compress_nWAY()

		# --- Step (1) initial stimation of matrix S and C ---
		# init Dn(l) ---->  D1(l) = S | D2(l) = C
		if type(S) is np.ndarray: # readed from init
			pass
		elif type(self.S) is np.ndarray: # read from obj
			S = self.S
		elif not type(self.S) is np.ndarray and type(self.Na) is int and type(self.Ni) is int: # random init (mode 1)
			S = np.random.rand(Dc.shape[0], self.Na+self.Ni)
		else:
			S = np.random.rand(Dc.shape[0], 1) # random init (mode 2)
		
		# ---- Yf ---- Y1 = D | Y2 = D.T
		C = np.dot(Dc.T, np.linalg.pinv(S).T)

		# --- Step (2) alternating least squares (MCR-ALS) ---
		for i in range(max_iter):
			# ---- Zn ---- Z1 = C | Z2 = S		------> en este caso particular no requiere operaciones adicionales
			# ---- Yf ---- Y1 = D | Y2 = D.T 	------> en este caso particular no requiere operaciones adicionales
			S=np.dot(Dc, np.linalg.pinv(C).T);	
			S, C = self.restrictions(S, C, constraints) # restrictions are apply here 
			
			C=np.dot(Dc.T, np.linalg.pinv(S).T)
			S, C = self.restrictions(S, C, constraints) # restrictions are apply here 

		if save:	# --- SAVE actual trainning in class MCR --- #
			self.S = S
			self.C = C
			self.L = [S, C]
			self.Dc = Dc
			self.constraints = constraints

		return [S, self.C]

	def restrictions(self, S, C, constraints):
		for key, value in constraints.items():
			#if key == 'no camlibration interferents':
			#	pass

			if key == 'non-negativity':
				if (type(value) == str and value == 'S') or (type(value) == list and 'S' in value):
					S = np.where(S>0, S, 0) # -- Channel 1 -- #  # model : L1 = D Z1**-1 and NON-negativity
				if (type(value) == str and value == 'all') or (type(value) == list and len(value)>0 and value[0] == 'all' ):
					S = np.where(S>0, S, 0) # -- Channel 1 -- #  # model : L1 = D Z1**-1 and NON-negativity
			if key == 'non-negativity S' and type(value) is bool and value == True:
				S = np.where(S>0, S, 0) # -- Channel 1 -- #  # model : L1 = D Z1**-1 and NON-negativity

			if key == 'non-negativity':
				if (type(value) == str and value == 'C') or (type(value) == list and 'C' in value):
					C = np.where(C>0, C, 0) # -- Channel 2 -- #  # model : L1 = D Z1**-1 and NON-negativity
				if (type(value) == str and value == 'all') or (type(value) == list and len(value)>0 and value[0] == 'all' ):
					C = np.where(C>0, C, 0) # -- Channel 2 -- #  # model : L1 = D Z1**-1 and NON-negativity
			if key == 'non-negativity C' and type(value) is bool and value == True:
				C = np.where(C>0, C, 0) # -- Channel 2 -- #  # model : L1 = D Z1**-1 and NON-negativity
			
			if key == 'fixed S': # -- fixed S  -- #
				l1, l2 = value.shape
				S[:l1, :l2] = value

			if key == 'fixed C': # -- fixed C  -- #
				l1, l2 = value.shape
				C[:l1, :l2] = value

			if key == 'Normalization S': # -- S Normalization -- #
				for i in range(S.shape[1]): S[:,i] =  S[:,i]/np.linalg.norm(S[:,i]) 

			if key == 'Normalization C': # -- C Normalization -- #
				for i in range(C.shape[1]): C[:,i] =  C[:,i]/np.linalg.norm(C[:,i]) 

			if key == 'Normalization': # -- Normalization -- #
				if type(value) == list:
					if 'S' in value: 
						for i in range(S.shape[1]): S[:,i] =  S[:,i]/np.linalg.norm(S[:,i]) 
					if 'C' in value: 
						for i in range(C.shape[1]): C[:,i] =  C[:,i]/np.linalg.norm(C[:,i])
				elif type(value) == bool and value == True:
					for i in range(S.shape[1]): S[:,i] =  S[:,i]/np.linalg.norm(S[:,i]) 
					for i in range(C.shape[1]): C[:,i] =  C[:,i]/np.linalg.norm(C[:,i])

			if key == 'Null interferents': # -- Normalization -- #
				C[:self.D.shape[-1]*self.Nc,self.Na:] = 1

		return S, C

	def predic(self, Nc=None, Na=None, Ni=None, N=None, f=None, save=True, v=1):
		if not type(N) is np.ndarray and not type(N) is int and N == None and (type(self.N) is np.ndarray or type(self.N) is int): N = self.N
		else: print('ERROR :: code 050 :: MCR.predic() :: can NOT get N({}) or self.N({})'.format(N, self.N))
		if not type(Nc) is np.ndarray and not type(Nc) is int and Nc == None and (type(self.Nc) is np.ndarray or type(self.Nc) is int): Nc = self.Nc
		else: print('ERROR :: code 050 :: MCR.predic() :: can NOT get Nc({}) or self.Nc({})'.format(Nc, self.Nc))

		if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
		else: print('ERROR :: code 050 :: MCR.predic() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
		if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
		else: print('ERROR :: code 050 :: MCR.predic() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		else: print('ERROR :: code 050 :: MCR.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		else: print('ERROR :: code 050 :: MCR.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if not type(self.y) is np.ndarray or not self.y.shape == (self.f, self.X.shape[0]): 
			if type(self.y) is np.ndarray:	print('WARNING :: code 1000 MCR.predic() :: Previuos data store in self.y will be erased. ')
			self.y = np.zeros((self.f, self.X.shape[0]))

		if v >= 1:	print(' (1) Predic tensor coeficients.')

		Cm = np.zeros((self.N, self.Na+self.Ni, self.X.shape[-1] ))
		for n in range(self.N):	Cm[n,:,:] = self.C[n*self.X.shape[-1]:(n+1)*self.X.shape[-1], :].T
		if save: self.Cm = Cm
		self.A = np.sum( Cm, axis=2 ).T

		if v >= 2:	
			print(' \t\t * coeficients : {}'.format(self.A.shape))
			for n in range(self.A.shape[1]):
				vec_str = '\t {} '.format( int(n+1) )
				for m in range(self.A.shape[0]): 	
					vec_str += '\t\t {:<.3f}'.format(self.A[m][n])
					if v >= 3: vec_str += '\t\t {:<.3f}'.format(self.A[m][n]/np.linalg.norm(self.A[m,:]))

		if v >= 1:	print(' (2) Predic calibration, validation and test samples.')
		if self.Na > 1: self.MSE = np.ones((self.Na))*99**3
		else: self.MSE = np.array([99**3])
		ref_index = np.zeros((self.Na))

		for n in range(self.Na):
			for m in range(self.Na+self.Ni):
				try:
					z = np.polyfit( self.A[int(m), :self.Nc] , self.Y[:self.Nc, int(n)], 1)
					self.MCR_prediction = self.A[int(m), :]*z[0]+z[1]

					predic = self.MCR_prediction
					MSE_nm = np.sum(( predic[:self.Nc] - self.Y[:self.Nc, int(n)])**2)
					if self.MSE[n] > MSE_nm:
						ref_index[n], self.y[n, :], self.MSE[n] = m, predic, MSE_nm
						self.pseudounivariate_parameters = z
				except: pass
				
		if v >= 1:	
			self.summary()

		return self.y

	def plot_loadings(self, loading_plot=None, fig=None, ext='compress', factor_name=None, nomalized=False):
		# ******** INIT ******** # 
		if type(self.L) is list: 
			for i, loading in enumerate(self.L):
				if not type(loading) is np.ndarray and not type(loading) is list :
					print('ERROR :: code 050 :: MCR.plot_loadings() :: can NOT plot loading {}'.format(i))
		try:
			if   type(loading_plot) is np.ndarray and len(loading_plot.shape) == 1:	pass
			elif type(loading_plot) is int: 		loading_plot = [loading_plot]
			elif type(loading_plot) is list: 		pass 
			elif not type(loading_plot) is list: 	loading_plot = [0, 1]
			
			try:
				if fig == None:	fig = plt.figure()
				else: fig=fig
			except:
				print('ERROR :: code 020b MCR.plot_loadings() :: can NOT generate fig obj')
			
			try: 	 # ******** define axis  ********
				axis_list = []
				for i, n in enumerate(loading_plot): 
					axis_list.append(fig.add_subplot( int(len(loading_plot))*100 + int(10) + (i+1) ))
					
					axis_list[i].set_xlabel('X', fontdict=self.font)
					axis_list[i].set_ylabel('Y', fontdict=self.font)
					axis_list[i].set_title('loading {}'.format(i+1) , fontdict=self.font)

					axis_list[i].spines["right"].set_linewidth(2)
					axis_list[i].spines["right"].set_color("#333333")
					axis_list[i].spines["left"].set_linewidth(2)
					axis_list[i].spines["left"].set_color("#333333")
					axis_list[i].spines["top"].set_linewidth(2)
					axis_list[i].spines["top"].set_color("#333333")
					axis_list[i].spines["bottom"].set_linewidth(2)
					axis_list[i].spines["bottom"].set_color("#333333")

					if factor_name != None:
						legend = axis_list[i].legend(shadow=True, fontsize='large')
						legend.get_frame().set_facecolor('#FFFFFF')

			except:	print('ERROR :: code 020c MCR.plot_loadings() :: can NOT generate axis, maybe fig argument is NOT what except ')

			# ******** plot normalized cromatografic vectors ********
			if nomalized: 
				for i, n in enumerate(loading_plot):  # --- iteration in each i-WAY --- # 
					try:
						for j, m in enumerate(self.L[n].T): 	# --- iteration in each j-FACTOR --- # 
							if n < 1: 							#  PLOT well aling ways  #
								self.L[i][:,j] = self.L[i][:,j] / np.linalg.norm(self.L[i][:,j], ord=None) # Frobenius norm
							elif n >= 1: 						#  PLOT well aling ways  #
								self.L[i][:,j] = self.L[i][:,j] / np.linalg.norm(self.L[i][:,j], ord=None) # Frobenius norm
					except: print('ERROR :: code 020c MCR.plot_loadings() :: can NOT normalize {} WAY {} factor '.format(i,j) )	

			# ******** Compress or NONaugmented ********
			if ext == 'compress' or ext == 'non-augmented':
				Cm = np.zeros((self.N, self.Na+self.Ni, self.X.shape[-1] ))
				for n in range(self.N):	Cm[n,:,:] = self.C[n*self.X.shape[-1]:(n+1)*self.X.shape[-1], :].T

			if factor_name == None:		factor_name = [ None for j, m in enumerate(self.L[0].T) ] 

			try: 
				# ***** ***** PLOT ***** ***** #
				for i, n in enumerate(loading_plot):  # --- iteration in each i-WAY --- # 
					try:
						for j, m in enumerate(self.L[n].T): 	# --- iteration in each j-FACTOR --- # 
							if n < 1: 					#  PLOT well aling ways  #
								axis_list[i].plot(m, color=self.colors[j], linestyle='-', linewidth=1.5,
											marker='o', markersize=0, alpha=0.7, label=factor_name[j])

							elif ext == 'augmented':		#  PLOT NON aling ways  #
								loading_max, loading_min = np.max(self.L[n]), np.min(self.L[n])
								for l in range(1, int(self.X.shape[0]) ): # --- iteration in each l-SAMPLE --- #
									x1, x2 = l*int(self.X.shape[n+1]), l*int(self.X.shape[n+1]) 
									y1, y2 = loading_max, loading_min
									axis_list[i].plot([x1, x2], [y1, y2], color='#444444', linestyle='--', linewidth=1.5, alpha=0.7)	#  PLOT sample separation lines  #

								axis_list[i].plot(m.T, color=self.colors[j], linestyle='-', linewidth=1.5, 			#  PLOT augmented self vector  #
											marker='o', markersize=0, alpha=0.7, label=factor_name[j])	

							elif ext == 'compress' or ext == 'non-augmented':		#  PLOT NON aling ways  #
								loading_max, loading_min = np.max(self.L[n]), np.min(self.L[n])
								color_rgb = list( float(int(self.colors[j][1:][i:i+2], 16))/255  for i in (0, 2, 4) )
								for sample in range(Cm.shape[0]):
									color = tuple([ color_rgb[0] * np.max(Cm[sample,j,:])/np.max(Cm[:,j,:]),
													color_rgb[1] * np.max(Cm[sample,j,:])/np.max(Cm[:,j,:]),
													color_rgb[2] * np.max(Cm[sample,j,:])/np.max(Cm[:,j,:]) ])
									axis_list[i].plot(Cm[sample,j,:].T, color=color, linestyle='-', linewidth=1.5, 			#  PLOT augmented self vector  #
												marker='o', markersize=0, alpha=0.6, label=factor_name[j])	

							else:
								pass
					except:		print('WARNING :: code 050 :: MCR.plot_loadings() :: can NOT plot loadings prediction')
			except:	print('ERROR :: code 020c MCR.plot_loadings() :: can NOT plot loading ')
		except: print('ERROR :: code 020c MCR.plot_loadings() :: can NOT plot loadings ')	


	def plot_45(self, Nc=None, Na=None, Ni=None, fig=None, color=None, factor_plot=None, f=None, error_plot=None):
		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		else: print('ERROR :: code 050 :: GPV.plot_45() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		else: print('ERROR :: code 050 :: GPV.plot_45() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if error_plot == None: error_plot = 'sample'
		else: print('ERROR :: code 050 :: GPV.plot_45() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		

		if type(factor_plot) == int:  	factor_plot = [factor_plot]
		elif type(factor_plot) == list: pass
		elif factor_plot == None:		factor_plot = range(f)
		else: factor_plot = range(f)

		try:
			for i in factor_plot: 

				try: # ** make new figure ** #
					if fig == None:	fig = plt.figure()
					else: fig=fig
				except:
					print('ERROR :: code 020b DATA.plot_3d() :: can NOT generate fig obj')

				try: # ** make new axis ** #	 
					ax1 = fig.add_subplot(211)
					if error_plot == 'sample': 	ax2 = fig.add_subplot(212) # ** plot sample error ** #	
					if error_plot == 'cc': 		ax2 = fig.add_subplot(212, sharex=ax1) # ** plot cc error ** #	
				except:	print('ERROR :: code 020c DATA.plot_3d() :: can NOT generate axis, maybe fig argument is NOT what except ')
				
				# ***** PLOT 1 ***** #
				x1, x2 = np.min(self.Y[:self.Nc,i]), np.max(self.Y[:self.Nc,i])
				ax1.plot([x1, x2], [x1, x2], '--', color='#222222', lw=2)

				try:		ax1.plot(self.Y[:self.Nc,i], self.y[i,:self.Nc], 'o', color=self.colors[i], marker='o')
				except:		print('WARNING :: code 050 :: MCR.plot_45() :: can NOT plot CALIBRATION prediction 45')
				
				try:		ax1.plot(self.Y[self.Nc:,i], self.y[i,self.Nc:], 'o', color=self.colors[i], marker='^')
				except:		print('WARNING :: code 050 :: MCR.plot_45() :: can NOT plot TEST prediction 45')

				ax1.set_xlabel('Y (cc.)')
				ax1.set_ylabel('y (estimation cc.)')
				ax1.set_title('Analite {}'.format(i+1))

				# ***** PLOT 2 ***** #
				try:
					if error_plot == 'sample': # ** plot sample error ** #	
						ax2.plot([0, self.Nc-1], [0, 0], '--', color='#333333', lw=2, alpha=0.8)
						ax2.plot( range(self.Nc), self.y[i,:self.Nc]-self.Y[:self.Nc,i], 'o', color=self.colors[i], marker='o')
						#ax2.plot( range(self.Y[self.Nc:,i].shape[0]), self.y[i,self.Nc:]-self.Y[self.Nc:,i], 'o', color=self.colors[i], marker='^')
						ax2.set_xlabel('Y (cc.)')
						ax2.set_ylabel('Error')
						ax2.set_title('Analite {}'.format(i+1))
				except: print('ERROR :: code 050 :: MCR.plot_45() :: can NOT plot45 (plot 2 error)')
			
				try:
					if error_plot == 'cc': # ** plot cc error ** #	
						ax2.plot([x1, x2], [0, 0], '--', color='#333333', lw=2)
						ax2.plot(self.Y[:self.Nc,i], self.y[i,:self.Nc]-self.Y[:self.Nc,i], 'o', color=self.colors[i], marker='o')
						ax2.plot(self.Y[self.Nc:,i], self.y[i,self.Nc:]-self.Y[self.Nc:,i], 'o', color=self.colors[i], marker='^')
						ax2.set_xlabel('Y (cc.)')
						ax2.set_ylabel('Error')
						ax2.set_title('Analite {}'.format(i+1))
				except: print('ERROR :: code 050 :: MCR.plot_45() :: can NOT plot45 (plot 2 error)')
		except:	print('ERROR :: code 050 :: MCR.plot_45() :: can NOT plot 45')


	def steal_data(self, obj):
		try: self.__dict__ = obj.__dict__.copy() 
		except: print('ERROR :: code 051 :: MCR.steal_data() :: can steal data')

	def summary(self, X=None, Y=None , Nc=None, Na=None, Ni=None, N=None):
		try: 
			if not X == None: self.X = X
		except: pass
		try:
			if not Y == None: self.Y = Y
		except: pass
		try:
			if not N == None: self.N = N
		except: pass
		try:
			if not Nc == None: self.Nc = Nc
		except: pass
		try:
			if not Na == None: self.Na = Na
		except: pass
		try:
			if not Ni == None: self.Ni = Ni
		except: pass

		print(' \t\t * Predic : {}'.format(self.A.shape))
		for a in range(self.A.shape[0]): 
			print(' {} Factor {} {}'.format('*'*15, int(a+1), '*'*15) )
			print('Sample \t\t\t Y \t\t\t y(stimation) \t\t\t Error \t\t\t e% \t\t\t ')
			for n in range(self.A.shape[1]):
				if self.Nc == n:
					print(' ---------------- '*5)
				
				if self.Nc > n:
					try:
						print('Cal{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
														(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )
					except: print(f'WARNNIGN :: MCR.summary() :: Missidata Y {self.Y.shape} y {self.y.shape}')
				else:
					try:
						print('Test{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
															(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )
					except: print(f'WARNNIGN :: MCR.summary() :: Missidata Y {self.Y.shape} y {self.y.shape}')
		return None











		