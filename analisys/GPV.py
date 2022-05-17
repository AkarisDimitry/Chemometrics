# *** numeric libraries
import numpy as np #pip3 install numpy

import logging, operator

# *** graph libraries
try:
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pl
except: 
	print('WARNNING :: PARAFAC.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: (pip3 install matplotlib)')

class GPV(object): # generador de datos
	def __init__(self, X=None, D=None, L=None, f=None, a=None, A=None,
				Y=None, y=None,	N=None, Nc=None, Na=None, Ni=None):
		self.X = np.array(X)

		self.D = np.array(D)

		self.a = a
		self.A = A 	# tensor coeficient self.A.shape = (f, N) #
		self.L = L

		self.Cm = None

		self.Y = np.array(Y)
		self.y = np.array(y)
		self.MSE = None

		self.N  = N
		self.Nc = Nc

		self.Na = Na
		self.Ni = Ni

		self.loadings = [] 	 	# loadings 
		self.Nloadings = []  	# normalized loadings 
		self.tensors = []
		self.model_mse = None 		# MSE error
		self.pseudounivariate_parameters = None

		self.colors = None
		self.restriction = []

		if f == None and self.Na != None and self.Ni != None:  self.f = self.Na + self.Ni
		else: self.f = f

	def isnum(self, n):
		# ------------------ Define if n is or not a number ------------------ # 
		# n     :   VAR     :   VAR to check if it is a numerical VAR
		# return :  BOOL    : True/False
		try: float(n); return True
		except: return False

	def loading_stimation(self, x=None, nfactors=None, max_iter=3, inicializacion='random', restriction='None'):

		if not type(x) is np.ndarray and type(self.X) is np.ndarray: x = self.X
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get x or self.X')

		if not self.isnum(nfactors):
			if self.isnum(self.f): 
				nfactors = int(self.f)
			elif  self.isnum(self.Na) and self.isnum(self.Ni):
				nfactors = int(self.Na + self.Ni)
				print('WARNING :: code 110A PARAFAC.loading_stimation() :: Setting to default value self.f = int(self.Na + self.Ni) ')
			else: 
				print('WARNING :: code 110A PARAFAC.loading_stimation() :: Setting to default value self.f = 1 ')
				nfactors = 1

		self.loadings, self.model_mse = self.parafac_base(x, nfactors, max_iter, inicializacion, restriction)
		s, self.Nloadings = self.normalized_loadings(self.loadings)
		return self.Nloadings
		 
	def train(self, X=None, a=None, L=None, f=None, restriction=None, max_iter=4, progressbar=None):
		if not X == None: self.X = X
		if not a == None: self.a = a
		if not L == None: self.L = L
		if not f == None: self.f = self.Na + self.Ni
		if restriction == None: restriction = [0]

		# MCR datos de tercer orden orden. 1 canal NO alienado y 2 canales alineados.
		# X = [Nc, l1, l2, l3] ; l1 l2: alineados  	l3 : NO alineados
		# 	(1) X			:	TENSOR 	: tensor con todos los datos.  X = [Nc, l1, l2, l3]
		# 	(2) a			:	INT 	: Numero de canales alineados
		# 	(3) f			:	MAT 	: numero de factores
		# 	(4) max_iter	:	INT 	: Numero maximo de iteraciones

		self.O, Op = len( self.X.shape ), len( self.X.shape ) - 1 # O: DATA order ; Op: numero de operaciones por iteracion
		self.L, Y = [], [] 

		# ---- THE model ----
		#  Z x = Y 
		# self.a <= m; 	Zc[Na x Nc.l1.l=!m...lo], 	xc[lm x Na], 		Yc[lm x Nc.l1.l=!m...lo]
		# self.a > m ; 	Zc[Na x l1.l=!m...lo], 		xc[Nc.lm x Na], 	Yc[Nc.lm x l1.l=!m...lo]

		#  ----  Y  ----  # 
		for n in range(1, self.O):
			if n <= self.a: # Case :: n <= self.a --> for all aligned channels
				Yr = np.rollaxis(self.X, n)
				Y.append( Yr.reshape(Yr.shape[0], -1)) # Yw = [lw, long]

			else: # case :: n > self.a --> for all NON-aligned channels
				Yr = np.rollaxis(self.X, n, 1)
				for l in range(self.O-2-self.a): Yr = np.sum( Yr, axis=self.a+2)
				Y.append( Yr.reshape(Yr.shape[0]*Yr.shape[1], -1) )# Yw = [lw, long]

		#  ----  Y  ----  # 
		#for w in range(1, ways):
		#	if w <= aling_modes: # Case ::  for all aligned channels
		#		Yr = np.rollaxis(X, w)
		#		Y[w] = Yr.reshape(Yr.shape[0], -1).T # Yw = [lw, long]
		#
		#	else: # case :: w > a --> for all NON-aligned channels
		#		Yr = np.rollaxis(X, w, 1) # Yr = [s, lw, ...]
		#		for l in range(O-2-a): Yr = np.sum( Yr, axis=a+2)
		#		Y[w] = Yr.reshape(Yr.shape[0]*Yr.shape[1], -1).T # Yw = [lw, long]

		#Y = np.array(Y)
		#Yr = np.random.rand( f, self.X.shape[0] )
		#for l in range(self.O-1-self.a): Yr = np.sum( Yr, axis=self.a+1)
		#for m in range(self.a):
		#	Yc = Yr[:,0]
		#	for l in range(1, Yr.shape[2]): Yc = np.concatenate( (Yc ,  Yr[:,l]), axis = 1)
		#	Yr = Yc

		#  ----  X  ----  #
		for n in range(1, self.O):
			if n <= self.a:	self.L.append( np.random.rand( self.f, self.X.shape[n] ) )	# Case n <= self.a --> aligned channels
			else:	self.L.append( np.random.rand( self.f, self.X.shape[n]*self.X.shape[0] ) ) # case n > self.a --> NON-aligned channels

		for i in range(max_iter): # # ** Iterative model ** # Convergencia

			#  ----  Z  ----  # 
			for n in range(Op): # -- Channel n -- #  Actualizacion del n-esimo cannal
				if n <= self.a-1: # ALIGNED chennels
					Zlf = []
					for fn in range(self.f): # -- Factor fn -- #
						Zlc = []
						for Nc in range( self.X.shape[0] ): # -- Sample Nc -- #
							if n == 0 and 1 <= self.a-1:  	Zp = self.L[1][fn,:]; Vinit = 2 
							elif n == 0 and 1 > self.a-1: 	Zp = self.L[1][fn,Nc*self.X.shape[1+1]:(Nc+1)*self.X.shape[1+1]]; Vinit = 2
							else: 							Zp = self.L[0][fn,:]; Vinit = 1

							for ln in range(Vinit, Op ): # Iteracion entre los canales que componen self.a Z
								if ln <= self.a-1 and ln != n:	Zp = np.tensordot(Zp, self.L[ln][fn,:], axes=0);
								elif ln > self.a-1:	Zp = np.tensordot(Zp, self.L[ln][fn, Nc*self.X.shape[ln+1]:(Nc+1)*self.X.shape[ln+1]], axes=0)
							Zlc.append( Zp )
						Zlf.append( Zlc )
					Zlf = np.array( Zlf )
					for m in range(self.O-2):
						Z = Zlf[:,:,0]
						for l in range(1, Zlf.shape[2]): Z = np.concatenate( (Z ,  Zlf[:,:,l]), axis = 1)
						Zlf = Z

				elif self.a > 0: # NON ALIGNED channels
					Zlf = []
					for fn in range(self.f): # -- Factor fn -- #
						Zp = self.L[0][fn,:]
						for ln in range(1, self.a ): Zp = np.tensordot(Zp, self.L[ln][fn,:], axes=0);# Iteracion entre los canales que componen a Z
						Zlf.append( Zp )
					Z = np.array( Zlf )
					for m in range(self.a-1): # Iteracion entre los cannales alineados #
						Ze = Z[:,:,0]
						for l in range(1, Z.shape[2]): Ze = np.concatenate( (Ze ,  Z[:,:,l]), axis = 1)
						Z = Ze

				#  ----  Resolution of the model  ----  # Xn(self.L) = Y.Zn**-1
				self.L[n]=np.dot( Y[n], np.linalg.pinv(Z) ).T;	self.L[n] = np.where(self.L[n]>0, self.L[n], 0)

				if restriction[0] != 0: 
					self.restrictions(restriction=restriction)

				if n < self.a: #  ----  NORMALIZATION  ---- 
					for i in range(self.f): 
						self.L[n][i,:] =  self.L[n][i,:]/np.linalg.norm(self.L[n][i,:])

			#Lc=np.dot(Yc, np.linalg.pinv(Zc) ).T;	Lc = np.where(Lc>0, Lc, 0)

		# --- Tensor spacec estimation --- # 
		self.tensors = np.ones(self.f, *self.X.shape)  # self.tensors[N-factor, M-sample, *all ways*] ! (CHANGE in python 2.7)
		for n in range(self.N): 		# iteration in samples #
			for m in range(self.f):		# iteration in factors #
				T = self.L[0][m, :]		# T is m,n tensor 	   #
				for o in range(1,Op):
					if o < self.a:	T = np.tensordot( T, self.L[o][m, :] , axes=0 )
					else:			T = np.tensordot( T, self.L[o][m, n*self.X.shape[o+1]:(n+1)*self.X.shape[o+1]] , axes=0 )

				self.tensors[m,n,:] = T

		return self.L


	def train_old(self, X=None, a=None, L=None, f=None, restriction=None, max_iter=4, progressbar=None):
		if not X == None: self.X = X
		if not a == None: self.a = a
		if not L == None: self.L = L
		if not f == None: self.f = self.Na + self.Ni
		if restriction == None: restriction = [0]

		# MCR datos de tercer orden orden. 1 canal NO alienado y 2 canales alineados.
		# X = [Nc, l1, l2, l3] ; l1 l2: alineados  	l3 : NO alineados
		# 	(1) X			:	TENSOR 	: tensor con todos los datos.  X = [Nc, l1, l2, l3]
		# 	(2) a			:	INT 	: Numero de canales alineados
		# 	(3) f			:	MAT 	: numero de factores
		# 	(4) max_iter	:	INT 	: Numero maximo de iteraciones

		self.O, Op = len( self.X.shape ), len( self.X.shape ) - 1 # O: DATA order ; Op: numero de operaciones por iteracion
		self.L, Y = [], [] 

		# ---- THE model ----
		#  Z x = Y 
		# self.a <= m; 	Zc[Na x Nc.l1.l=!m...lo], 	xc[lm x Na], 		Yc[lm x Nc.l1.l=!m...lo]
		# self.a > m ; 	Zc[Na x l1.l=!m...lo], 		xc[Nc.lm x Na], 	Yc[Nc.lm x l1.l=!m...lo]

		#  ----  Y  ----  # 
		for n in range(1, self.O):
			if n <= self.a: # Case :: n <= self.a --> for all aligned channels
				Yr = np.rollaxis(self.X, n)
				Y.append( Yr.reshape(Yr.shape[0], -1)) # Yw = [lw, long]

			else: # case :: n > self.a --> for all NON-aligned channels
				Yr = np.rollaxis(self.X, n, 1)
				for l in range(self.O-2-self.a): Yr = np.sum( Yr, axis=self.a+2)
				Y.append( Yr.reshape(Yr.shape[0]*Yr.shape[1], -1) )# Yw = [lw, long]

		#Y = np.array(Y)
		#Yr = np.random.rand( f, self.X.shape[0] )
		#for l in range(self.O-1-self.a): Yr = np.sum( Yr, axis=self.a+1)
		#for m in range(self.a):
		#	Yc = Yr[:,0]
		#	for l in range(1, Yr.shape[2]): Yc = np.concatenate( (Yc ,  Yr[:,l]), axis = 1)
		#	Yr = Yc

		#  ----  X  ----  #
		for n in range(1, self.O):
			if n <= self.a:	self.L.append( np.random.rand( self.f, self.X.shape[n] ) )	# Case n <= self.a --> aligned channels
			else:	self.L.append( np.random.rand( self.f, self.X.shape[n]*self.X.shape[0] ) ) # case n > self.a --> NON-aligned channels

		for i in range(max_iter): # # ** Iterative model ** # Convergencia

			#  ----  Z  ----  # 
			for n in range(Op): # -- Channel n -- #  Actualizacion del n-esimo cannal
				if n <= self.a-1: # ALIGNED chennels
					Zlf = []
					for fn in range(self.f): # -- Factor fn -- #
						Zlc = []
						for Nc in range( self.X.shape[0] ): # -- Sample Nc -- #
							if n == 0 and 1 <= self.a-1:  Zp = self.L[1][fn,:]; Vinit = 2 
							elif n == 0 and 1 > self.a-1: Zp = self.L[1][fn,Nc*self.X.shape[1+1]:(Nc+1)*self.X.shape[1+1]]; Vinit = 2
							else: 		Zp = self.L[0][fn,:]; Vinit = 1
							for ln in range(Vinit, Op ): # Iteracion entre los canales que componen self.a Z
								if ln <= self.a-1 and ln != n:	Zp = np.tensordot(Zp, self.L[ln][fn,:], axes=0);
								elif ln > self.a-1:	Zp = np.tensordot(Zp, self.L[ln][fn, Nc*self.X.shape[ln+1]:(Nc+1)*self.X.shape[ln+1]], axes=0)
							Zlc.append( Zp )
						Zlf.append( Zlc )
					Zlf = np.array( Zlf )
					for m in range(self.O-2):
						Z = Zlf[:,:,0]
						for l in range(1, Zlf.shape[2]): Z = np.concatenate( (Z ,  Zlf[:,:,l]), axis = 1)
						Zlf = Z

				elif self.a > 0: # NON ALIGNED channels
					Zlf = []
					for fn in range(self.f): # -- Factor fn -- #
						Zp = self.L[0][fn,:]
						for ln in range(1, self.a ): Zp = np.tensordot(Zp, self.L[ln][fn,:], axes=0);# Iteracion entre los canales que componen a Z
						Zlf.append( Zp )
					Z = np.array( Zlf )
					for m in range(self.a-1): # Iteracion entre los cannales alineados #
						Ze = Z[:,:,0]
						for l in range(1, Z.shape[2]): Ze = np.concatenate( (Ze ,  Z[:,:,l]), axis = 1)
						Z = Ze

				#  ----  Resolution of the model  ----  # Xn(self.L) = Y.Zn**-1
				self.L[n]=np.dot( Y[n], np.linalg.pinv(Z) ).T;	self.L[n] = np.where(self.L[n]>0, self.L[n], 0)

				if restriction[0] != 0: 
					self.restrictions(restriction=restriction)

				if n < self.a: #  ----  NORMALIZATION  ---- 
					for i in range(self.f): 
						self.L[n][i,:] =  self.L[n][i,:]/np.linalg.norm(self.L[n][i,:])

			#Lc=np.dot(Yc, np.linalg.pinv(Zc) ).T;	Lc = np.where(Lc>0, Lc, 0)

		# --- Tensor spacec estimation --- # 
		self.tensors = np.ones(self.f, *self.X.shape)  # self.tensors[N-factor, M-sample, *all ways*] ! (CHANGE in python 2.7)
		for n in range(self.N): 		# iteration in samples #
			for m in range(self.f):		# iteration in factors #
				T = self.L[0][m, :]		# T is m,n tensor 	   #
				for o in range(1,Op):
					if o < self.a:	T = np.tensordot( T, self.L[o][m, :] , axes=0 )
					else:			T = np.tensordot( T, self.L[o][m, n*self.X.shape[o+1]:(n+1)*self.X.shape[o+1]] , axes=0 )

				self.tensors[m,n,:] = T

		return self.L


	def restrictions(self, loadings=None, restriction=None):
		if restriction[0] == 'non-negative': # non-negativity restriction
			for i, n in enumerate(loadings): loadings[i] = np.where(loadings[i]>0, loadings[i], 0)

		if restriction[0] == 'test':
			self.L[0] = restriction[1]

		if restriction[0] == 'unimodalidad':
			N = restriction[2]
			L = restriction[3] 
			F = restriction[1]
			for f in range(F):
				for n in range(N):
					Lvec =  loadings[f, n*L:(n+1)*L]
					arg_max = np.argmax(Lvec)
					left_max = Lvec[arg_max]
					rigth_max = Lvec[arg_max]
					for l_max in range(arg_max-2):
						if Lvec[arg_max-1-l_max] < Lvec[arg_max-2-l_max]:
							Lvec[arg_max-2-l_max] = left_max
						else:
							left_max = Lvec[arg_max-1-l_max] 

					for r_max in range(L - arg_max-2):
						if Lvec[arg_max+r_max] < Lvec[arg_max+r_max+1]:
							Lvec[arg_max+r_max+1] = rigth_max
						else:
							rigth_max = Lvec[arg_max+r_max+1]

					loadings[f, n*L:(n+1)*L] = Lvec
		


		return loadings

	def pseudo_univariate(self, D=None, Nc=None):
		if not type(N) is np.ndarray and not type(N) is int and N == None and (type(self.N) is np.ndarray or type(self.N) is int): N = self.N
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get N({}) or self.N({})'.format(N, self.N))
		if not type(Nc) is np.ndarray and not type(Nc) is int and Nc == None and (type(self.Nc) is np.ndarray or type(self.Nc) is int): Nc = self.Nc
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get Nc({}) or self.Nc({})'.format(Nc, self.Nc))

		if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
		if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if not type(self.y) is np.ndarray or not self.y.shape == (self.f, self.X.shape[0]): 
			if type(self.y) is np.ndarray:	print('WARNING :: code 1000 GPV.predic() :: Previuos data store in self.y will be erased. ')
			self.y = np.zeros((self.f, self.X.shape[0]))

		if v >= 1:	print(' (1) Predic tensor coeficients.')
		self.A = self.tensors
		for ax in range(len(self.tensors.shape)-2):
			self.A = np.sum(self.A, axis=2)
		self.A = self.A ** (1/(len(self.X.shape)-1-self.a) )

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
			for m in range(self.f):
				z = np.polyfit( self.A[int(m), :self.Nc] , self.Y[:self.Nc, int(n)], 1)
				predic = self.A[int(m), :self.Nc]*z[0]+z[1]
				MSE_nm = np.sum(( predic[:self.Nc] - self.Y[:self.Nc, int(n)])**2)
				if self.MSE[n] > MSE_nm:	
					# store best fit model #
					# ref_index : 		VEC 		: best reference index 		i-factor
					# self.y 	: 		MAT 		: best prediction 			i-factor j-sample 
					# self.MSE : 		INT 		: RMS					  	i-factor
					ref_index[n], self.y[n, :], self.MSE[n] = m, predic, MSE_nm 
					self.pseudounivariate_parameters = z

		if v >= 1:	
			self.summary()

		return self.y

	def predic(self, Nc=None, Na=None, Ni=None, N=None, f=None, v=1):
		if not type(N) is np.ndarray and not type(N) is int and N == None and (type(self.N) is np.ndarray or type(self.N) is int): N = self.N
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get N({}) or self.N({})'.format(N, self.N))
		if not type(Nc) is np.ndarray and not type(Nc) is int and Nc == None and (type(self.Nc) is np.ndarray or type(self.Nc) is int): Nc = self.Nc
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get Nc({}) or self.Nc({})'.format(Nc, self.Nc))

		if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
		if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		else: print('ERROR :: code 050 :: GPV.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if not type(self.y) is np.ndarray or not self.y.shape == (self.f, self.X.shape[0]): 
			if type(self.y) is np.ndarray:	print('WARNING :: code 1000 GPV.predic() :: Previuos data store in self.y will be erased. ')
			self.y = np.zeros((self.f, self.X.shape[0]))

		if v >= 1:	print(' (1) Predic tensor coeficients.')
		self.A = self.tensors
		for ax in range(len(self.tensors.shape)-2):
			self.A = np.sum(self.A, axis=2)
		self.A = self.A ** (1/(len(self.X.shape)-1-self.a) )

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
			for m in range(self.f):
				z = np.polyfit( self.A[int(m), :self.Nc] , self.Y[:self.Nc, int(n)], 1)
				predic = self.A[int(m), :]*z[0]+z[1]
				MSE_nm = np.sum(( predic[:self.Nc] - self.Y[:self.Nc, int(n)])**2)
				if self.MSE[n] > MSE_nm:	
					ref_index[n], self.y[n, :], self.MSE[n] = m, predic, MSE_nm
					self.pseudounivariate_parameters = z

		if v >= 1:	
			self.summary()

		return self.y

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
					print('Cal{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
														(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )

				else:
					try:
						print('Test{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
														(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )
					except:
						print('Can NOT summarise result from sample {}'.format(n+1) )
		return None

	def plot_loadings(self, loading_plot=None, fig=None):
		if type(self.L) is list: 
			for i, loading in enumerate(self.L):
				if not type(loading) is np.ndarray and not type(loading) is list :
					print('ERROR :: code 050 :: GPV.plot_loadings() :: can NOT plot loading {}'.format(i))
		try:
			if  type(loading_plot) is np.ndarray and len(loading_plot.shape) == 1:	pass
			elif type(loading_plot) is int: loading_plot = [loading_plot]
			elif type(loading_plot) is list: 		pass 
			elif not type(loading_plot) is list: 	loading_plot = range( len(self.L) )
			
			try:
				if fig == None:	fig = plt.figure()
				else: fig=fig
			except:
				print('ERROR :: code 020b DATA.plot_loadings() :: can NOT generate fig obj')
			
			try: 	
				axis_list = []

				for i, n in enumerate(loading_plot): 
					axis_list.append(fig.add_subplot( int(len(loading_plot))*100 + int(10) + (i+1) ))
			except:	print('ERROR :: code 020c DATA.plot_loadings() :: can NOT generate axis, maybe fig argument is NOT what except ')

			try: 
				# ***** ***** PLOT ***** ***** #
				for i, n in enumerate(loading_plot):  # --- iteration in each i-WAY --- # 
					try:
						for j, m in enumerate(self.L[n]): 	# --- iteration in each j-FACTOR --- # 
							if n < self.a: 					#  PLOT well aling ways  #
								axis_list[i].plot(m, color=self.colors[j], linestyle='-', linewidth=1,
											marker='o', markersize=0)
							else:							#  PLOT NON aling ways  #
								loading_max, loading_min = np.max(self.L[n]), np.min(self.L[n])
								for l in range(1, int(self.X.shape[0]) ): # --- iteration in each l-SAMPLE --- #
									x1, x2 = l*int(self.X.shape[n+1]), l*int(self.X.shape[n+1]) 
									y1, y2 = loading_max, loading_min
									axis_list[i].plot([x1, x2], [y1, y2], color='#444444', linestyle='--', linewidth=1,)	#  PLOT sample separation lines  #

								axis_list[i].plot(m, color=self.colors[j], linestyle='-', linewidth=1, 			#  PLOT augmented self vector  #
											marker='o', markersize=0)	

					except:		print('WARNING :: code 050 :: GPV.plot_45() :: can NOT plot CALIBRATION prediction 45')
			except:	print('ERROR :: code 020c DATA.plot_loadings() :: can NOT plot loading ')
		except: print('ERROR :: code 020c DATA.plot_loadings() :: can NOT plot loadings ')	

	def plot_45(self, Nc=None, Na=None, Ni=None, fig=None, color=None, factor_plot=None, f=None):
		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		else: print('ERROR :: code 050 :: GPV.plot_45() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		else: print('ERROR :: code 050 :: GPV.plot_45() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if type(factor_plot) == int:  	factor_plot = [factor_plot]
		elif type(factor_plot) == list: pass
		elif factor_plot == None:		factor_plot = range(f)
		else: factor_plot = range(f)
		try:
			for i in factor_plot: 

				try:
					if fig == None:	fig = plt.figure()
					else: fig=fig
				except:
					print('ERROR :: code 020b DATA.plot_3d() :: can NOT generate fig obj')

				try: 	
					ax1 = fig.add_subplot(211)
					ax2 = fig.add_subplot(212, sharex=ax1)
				except:	print('ERROR :: code 020c DATA.plot_3d() :: can NOT generate axis, maybe fig argument is NOT what except ')
				
				# ***** PLOT 1 ***** #
				x1, x2 = np.min(self.Y[:self.Nc,i]), np.max(self.Y[:self.Nc,i])
				ax1.plot([x1, x2], [x1, x2], '--', color='#222222', lw=2)

				try:		ax1.plot(self.Y[:self.Nc,i], self.y[i,:self.Nc], 'o', color=self.colors[i], marker='o')
				except:		print('WARNING :: code 050 :: GPV.plot_45() :: can NOT plot CALIBRATION prediction 45')
				
				try:		ax1.plot(self.Y[self.Nc:,i], self.y[i,self.Nc:], 'o', color=self.colors[i], marker='^')
				except:		print('WARNING :: code 050 :: GPV.plot_45() :: can NOT plot TEST prediction 45')

				ax1.set_xlabel('Y (cc.)')
				ax1.set_ylabel('y (estimation cc.)')
				ax1.set_title('Analite {}'.format(i+1))

				# ***** PLOT 2 ***** #
				ax2.plot([x1, x2], [0, 0], '--', color='#222222', lw=2)

				ax2.plot(self.Y[:self.Nc,i], self.y[i,:self.Nc]-self.Y[:self.Nc,i], 'o', color=self.colors[i], marker='o')
				ax2.plot(self.Y[self.Nc:,i], self.y[i,self.Nc:]-self.Y[self.Nc:,i], 'o', color=self.colors[i], marker='^')
				ax2.set_xlabel('Y (cc.)')
				ax2.set_ylabel('Error')
				ax2.set_title('Analite {}'.format(i+1))
		except:	print('ERROR :: code 050 :: GPV.plot_45() :: can NOT plot 45')



