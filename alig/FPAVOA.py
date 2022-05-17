import numpy as np

# *** graph libraries
try:
	import matplotlib.pyplot as plt #pip3 install matplotlib
except: 
	print('WARNNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: ( pip3 install matplotlib )')
	
class FPAV21(object):
	def __init__(self, X=None, D=None, L=None, La=None, f=None, A=None, a=1,
				Y=None, y=None, S=None,	N=None, Nc=None, Na=None, Ni=None, Nt=None ):
		self.D = D
		self.X = X
		
		self.Y = Y
		self.y = y
		
		self.S = S

		self.L = L
		self.La = La

		self.a = a
		self.A = A

		self.f = f

		self.N = N
		self.Nc = Nc

		self.Na = Na
		self.Ni = Ni
		self.Nt = Nt

	def aling(self, D=None, S=None, Nc=None, Na=None, Ni=None, a=None):
		print(self.D.shape)
		try: 
			if not D  == None: 	self.D  = D
		except:	pass
		try: 
			if not S  == None: self.S  = S
		except:	pass
		if not Nc == None: self.Nc = Nc
		if not Na == None: self.Na = Na
		if not Ni == None: self.Ni = Ni
		if not a  == None: self.a  = a
		

		#if self.a == 2: self.aling_FAPV21(D=self.D, S=self.S, Nc=self.Nc, Na=self.Na, Ni=self.Ni)
		#elif self.a == None and len(self.D.shape) == 4: 
		
		self.aling_FAPV(D=self.D, S=self.S, Nc=self.Nc, Na=self.Na, Ni=self.Ni)

	def aling_FAPV(self, D=None, S=None, Nc=None, Na=None, Ni=None, A=None, a=None, iter=None):
		# ----- Algoritmo de alineado para datos de 3er orden con 2 canales alineados ----- #
		# 
		# self.D 		:	N-MAT 		:		N-array/Matrix non-aligned data D => [N,l1,l2]
		# self.S 		:	N-MAT 		:		N-array/Matrix estimation of well aligned channels  S => [[self.f,l1],[self.f,l2]] 
		# self.Nc		:	INT 		: 		number of calibration samples.
		# self.Na		:	INT 		: 		number of analites.
		# self.Ni		:	INT 		: 		number of interferentes.
		# self.A		:	INT 		: 		number of non-aligned channels.

		# self.S  	: [L1, L2] | L1=[f,l1], L2=[f,l2] 
		# self.f  	: factors number = number of analites(self,Na) + number of interferentes(self,Ni)
		# self.N  	: Numero de muestras de calibrado(Nc) + Numero de muestras test(Nt)
		# l1 		: number of variables of channel 1
		# l2 		: number of variables of channel 2
		# l3 		: number of variables of channel 3

		# --- Defino vaiables --- # 
		aling_modes = self.a

		D = self.D
		X = self.X

		N = self.N
		Nc = self.Nc
		Nt = self.Nt

		Na = self.Na
		Ni = self.Ni
		f = Na + Ni

		ways, modes  = int(len(D.shape)), int(len(D.shape)-1)
		Ln =  D.shape
		
		# (0) estimation of well aligned channels (commonly spectral channels)
		# (1) Estimar los canales no espectrales
		# (2) Alinear los canales no espectrales
		# (3) Reconstruir las muestras
		# (4) Postprocesar
		# pp : parameters [[numero de muestras de calibrado, numero de sensores, L[0]+L[10]], L[0] ]
		# spectra = spectral_stimation(data)
		Y = {}
		#  ----  Y  ----  # 
		for w in range(1, ways):
			if w <= aling_modes: # Case ::  for all aligned channels
				Yr = np.rollaxis(X, w)
				Y[w] = Yr.reshape(Yr.shape[0], -1).T # Yw = [lw, long]

			else: # case :: w > a --> for all NON-aligned channels
				Yr = np.rollaxis(X, w, 1) # Yr = [s, lw, ...]
				for l in range(O-2-a): Yr = np.sum( Yr, axis=a+2)
				Y[w] = Yr.reshape(Yr.shape[0]*Yr.shape[1], -1).T # Yw = [lw, long]


		#  ----  X  ----  #
		L = []
		for w in range(1, ways):
			if w <= aling_modes:	L.append( np.random.rand( self.f, self.X.shape[n] ) )	# Case n <= self.a --> aligned channels
			else:					L.append( np.random.rand( self.f, self.X.shape[n]*self.X.shape[0] ) ) # case n > self.a --> NON-aligned channels

		for i in range(max_iter): # # ** Iterative model ** # Convergencia

			#  ----  Z  ----  # 
			for mode in range(1, ways): # -- Channel n -- #  Actualizacion del n-esimo cannal
				if mode <= aling_modes: # ALIGNED chennels
				
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

		# --- Tensor spacec stimation --- # 
		self.tensors = np.ones((self.f, *self.X.shape))  # self.tensors[N-factor, M-sample, *all ways*]
		for n in range(self.N): 		# iteration in samples #
			for m in range(self.f):		# iteration in factors #
				T = self.L[0][m, :]		# T is m,n tensor 	   #
				for o in range(1,Op):
					if o < self.a:	T = np.tensordot( T, self.L[o][m, :] , axes=0 )
					else:			T = np.tensordot( T, self.L[o][m, n*self.X.shape[o+1]:(n+1)*self.X.shape[o+1]] , axes=0 )

				self.tensors[m,n,:] = T


			# ------ ------ MUESTRAS DE CALIBRADO ------ ------ #
		# --- Z --- #
		# all aligned channels ---> Z
		# Z[l1, ... , la] 
		Ze = np.zeros( [self.Na]+[n for n in Ln[1:-3]] )
		for n in range(self.Na):	Ze[n,:,:] = np.tensordot( self.S[0][n,:], self.S[1][n,:] , axes=0)
		Z = Ze[:,0,:]
		for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

	  	# --- Y --- #
		Ye = self.D[0,:,:,:]
		for n in range(1, self.Nc): Ye = np.concatenate( (Ye ,  self.D[n,:,:,:]), axis=2)
		Y = Ye[0,:,:]
		for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

	  	# --- X(Lc) --- #
		Lc=np.dot(Y.T, np.linalg.pinv(Z) )
		self.L = np.zeros( (self.Nc, self.Na, l3) )
		for nc in range(self.Nc): self.L[nc,:,:] = Lc[nc*l3:(nc+1)*l3,:].T
				# ------ AREAS ------ #
		self.A = np.sum( self.L, axis=2 ) # [N, self.f]


				# ------ ALINEADO FUNCIONAL ------ # Estimacion del tensor de datos alineado (self.Da)
		Lmax, Amax, self.Da, sigma = np.zeros((self.Na)), np.argmax(self.A, axis=0), np.zeros((self.N, l1, l2, l3)), l3/self.f/6
		for na in range(self.Na): Lmax[na] = np.argmax(self.L, axis=2)[Amax[na]][na]
		for na in range(self.Na):
			Ma = np.tensordot( np.tensordot(self.S[0][na, :], self.S[1][na, :], axes=0), self.gaussian( l3,sigma,Lmax[na]) , axes=0)
			for nc in range(self.Nc): self.Da[nc,:,:,:] += Ma*self.A[nc, na]

		# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		self.La = np.zeros( (self.N, self.f, l3) )
		for na in range(self.Na):
			for nc in range(self.Nc): self.La[nc, na, :] = self.gaussian( l3,sigma,Lmax[na])*self.A[nc, na]

	  			#  ------ ------ MUESTRAS TEST ------ ------ #
	   	# --- Z --- #
		Ze = np.zeros( (self.f, l1, l2) )
		for n in range(self.f):	Ze[n,:,:] = np.tensordot( self.S[0][n,:], self.S[1][n,:] , axes=0)
		Z = Ze[:,0,:]
		for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

	  	# --- Y --- #
		Ye = self.D[self.Nc,:,:,:]
		for n in range(1,self.Nt): Ye = np.concatenate( (Ye ,  self.D[self.Nc+n,:,:,:]), axis=2)
		Y = Ye[0,:,:]
		for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

	  	# --- X(Lc) --- #
		Lc=np.dot(Y.T, np.linalg.pinv(Z) )
		self.L = np.zeros( (self.Nt, self.f, l3) )
		for nt in range(self.Nt): self.L[nt,:,:] = Lc[nt*l3:(nt+1)*l3,:].T 

		# --- AREAS --- #
		self.A = np.sum( self.L, axis=2 ) # [self.Nt, f]

		# --- ALINEADO FUNCIONAL --- # Estimacion del tensor de datos alineado (self.Da)
		sigma = l3/self.f/6  # Amax = np.argmax(self.A, axis=0)
		for na in range(self.Na):
			Ma = np.tensordot( np.tensordot(self.S[0][na, :], self.S[1][na, :], axes=0), self.gaussian( l3,sigma,Lmax[na]) , axes=0)
			for nt in range(self.Nt): self.Da[self.Nc+nt,:,:,:] += Ma*self.A[nt, na]

		# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		for na in range(self.Na):
			for nt in range(self.Nt): self.La[nt+self.Nc, na, :] = self.gaussian( l3,sigma,Lmax[na])*self.A[nt, na]

		# Devolvemos (0-self.Da)Tensor de datos alineado (1-self.La)Perfiles cromatograficos alineados (2-self.L)Estimacion original de los perfiles cromatograficos
		print('Alineado exitoso')
		return self.Da, self.La, self.L # 
	
	def gaussian(self,n,s,m):
		return (1.0/(s*(2*np.pi)**0.5)) * np.e**(-0.5*((np.arange(0, n, dtype=np.float32)-m)/s)**2)

	def Lorentz(self,n,a,m):
		return (a*1.0/(np.pi*((np.arange(0, n, dtype=np.float32)-m))**2+a**2))

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

	def load_fromDATA(self, data=None):
		try: self.__dict__ = data.__dict__.copy() 
		except: pass

