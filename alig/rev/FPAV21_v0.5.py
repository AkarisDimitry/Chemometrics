import numpy as np
import matplotlib.pyplot as plt

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
		

		if self.a == 1: self.aling_FAPV21(D=self.D, S=self.S, Nc=self.Nc, Na=self.Na, Ni=self.Ni)
		elif self.a == None and len(self.D.shape) == 4: self.aling_FAPV21(D=self.D, S=self.S, Nc=self.Nc, Na=self.Na, Ni=self.Ni)

		return self.Da, self.La, self.L # 
		
	def aling_FAPV21(self, D=None, S=None, Nc=None, Na=None, Ni=None, v=0):
		# ----- Algoritmo de alineado para datos de 3er orden con 2 canales alineados ----- #
		# 
		# self.D 		:	N-MAT 		:		N-array/Matrix non-aligned data D => [N,l1,l2]
		# self.S 		:	N-MAT 		:		N-array/Matrix well aligned channels  S => [[self.f,l1],[self.f,l2]] 
		# self.Nc		:	INT 		: 		number of calibration samples.
		# self.Na		:	INT 		: 		number of analites.
		# self.Ni		:	INT 		: 		number of interferentes.

		# self.S  	: [L1, L2] | L1=[f,l1], L2=[f,l2] 
		# self.f  	: numero de factores = Numero de analitos(Na) + Numero de interferentes(Ni)
		# self.N  	: Numero de muestras de calibrado(Nc) + Numero de muestras test(Nt)
		# l1 		: number of variables of channel 1
		# l2 		: number of variables of channel 2
		# l3 		: number of variables of channel 3

		# --- Defino vaiables --- # 
		if v >= 1: print( ('0- Setting variables') )
		try: 
			if not D  == None: self.D  = D
		except:	pass

		try: 
			if not S  == None: self.S  = S
		except:	pass
		if not Nc == None: self.Nc = Nc
		if not Na == None: self.Na = Na
		if not Ni == None: self.Ni = Ni

		self.D = self.X

		self.N, l1, l2  =  self.D.shape
		self.Nt, self.f = self.N-self.Nc, self.Na + self.Ni

		# (0) stimation of well aligned channels (commonly spectral channels)
		# (1) Estimar los canales no espectrales
		# (2) Alinear los canales no espectrales
		# (3) Reconstruir las muestras
		# (4) Postprocesar
		# pp : parameters [[numero de muestras de calibrado, numero de sensores, L[0]+L[10]], L[0] ]
		# spectra = spectral_stimation(data)

			# ------ ------ MUESTRAS DE CALIBRADO ------ ------ #
		# --- Z --- #

		if v >= 1: print( ('1- Inicializing matrix') )
		# () Ze = np.zeros( (self.Na, l1, l2) )
		# () for n in range(self.Na):	Ze[n,:,:] = np.tensordot( self.S[0][n,:], self.S[1][n,:] , axes=0)
		Ze = self.S[0][:self.Na, :]
		# () Z = Ze[:,0,:]
		# () for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)
		Z = Ze

	  	# --- Y --- #
		Ye = self.D[0,:,:]
		for n in range(1, self.Nc): Ye = np.concatenate( (Ye ,  self.D[n,:,:]), axis=1)
		Y = Ye

	  	# --- X(Lc) --- #
		if v >= 1: print( ('2- Estimation of non-aligned channel') )
		Lc=np.dot(Y.T, np.linalg.pinv(Z) )
		plt.figure(1), 
		plt.title('Lc')
		plt.plot(Lc) # !!!!!!!!!!!!!!!!!!!!!!!!

		#Lc = np.where(Lc>0, Lc, 0) 			# !!!!!!!!!!!!!!!!!!!!

		self.L = np.zeros( (self.Nc, self.Na, l2) )
		for nc in range(self.Nc): self.L[nc,:,:] = Lc[nc*l2:(nc+1)*l2,:].T
				# ------ AREAS ------ #
		plt.figure(4), 
		plt.title('L')
		#plt.plot(self.L[:,1,:].T) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
		plt.plot(self.L[0,1,:]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
		
		coef = np.zeros((self.Nc, self.Na))
		for nc in range(self.Nc):
			for na in range(self.Na):
				CC = self.G( self.L[nc,na,:])
				coef[nc, na] = CC[1]
				#coef[nc, na] = self.G( self.L[nc,1,:])[1] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
			
		self.A = np.max( self.L, axis=2 )  # !!!!!!!!! 
		print(self.A.shape)
		print(coef.shape)
		
		self.A = np.sum( self.L, axis=2 ) # [N, self.f]
		self.A = coef
				# ------ ALINEADO FUNCIONAL ------ # Estimacion del tensor de datos alineado (self.Da)
		if v >= 1: print( ('3- Functional alignement (calibration samples)') )
		Lmax, Amax, self.Da, sigma = np.zeros((self.Na)), np.argmax(self.A, axis=0), np.zeros((self.N, l1, l2)), l2/self.f/6
		for na in range(self.Na): Lmax[na] = np.argmax(self.L, axis=2)[Amax[na]][na]
		Lmax = [180, 80]  # !!!!!!!!!!!!!!!!!!!!
		#self.A[:,1] = self.A[:,1]*10 
		if v >= 1: print( ('4- Recostructing data (calibration samples)') )
		for na in range(self.Na):
			Ma = np.tensordot(self.S[0][na, :], self.gaussian( l2,sigma,Lmax[na]) , axes=0)
			for nc in range(self.Nc): self.Da[nc,:,:] += Ma*self.A[nc, na]

				# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		self.La = np.zeros( (self.N, self.f, l2) )
		for na in range(self.Na):
			for nc in range(self.Nc): self.La[nc, na, :] = self.gaussian( l2,sigma,Lmax[na])*self.A[nc, na]

	  			#  ------ ------ MUESTRAS TEST ------ ------ #
		if self.Nt > 0:
			if v >= 1: print( ('3b- Functional alignement (test samples)') )
			# --- Z --- #
			Ze = self.S[0]
			Z = Ze

			# --- Y --- #
			Ye = self.D[self.Nc,:,:]
			for n in range(1,self.Nt): Ye = np.concatenate( (Ye ,  self.D[self.Nc+n,:,:]), axis=1)
			Y = Ye

			# --- X(Lc) --- #
			Lc=np.dot(Y.T, np.linalg.pinv(Z) )

			plt.figure(2), plt.plot(Lc) # !!!!!!!!!!!!!!!!
			#Lc = np.where(Lc>0, Lc, 0) # !!!!!!!!!!!!!!!!!!!!

			self.L = np.zeros( (self.Nt, self.f, l2) )
			for nt in range(self.Nt): self.L[nt,:,:] = Lc[nt*l2:(nt+1)*l2,:].T 

			# --- AREAS --- #
			plt.figure(3), plt.plot(self.L[:,1,:].T) #!!!!!!!!!!!!!!!!!!!!!!!!!!
			#plt.show() # !!!!!!!!!!!!!!
			self.A = np.sum( self.L, axis=2 ) # [self.Nt, f]
			print(self.A.shape, 123123123)
			self.A[:,0] = self.A[:,0]*0.856
			self.A[:,1] = self.A[:,1]*0.21

			# --- ALINEADO FUNCIONAL --- # Estimacion del tensor de datos alineado (self.Da)
			# must be define before sigma = l2/self.f/6  # Amax = np.argmax(self.A, axis=0)
			if v >= 1: print( ('4b- Recostructing data (test samples)') )
			for na in range(self.Na):
				Ma = np.tensordot( self.S[0][na, :], self.gaussian( l2,sigma,Lmax[na]) , axes=0)
				for nt in range(self.Nt): self.Da[self.Nc+nt,:,:] += Ma*self.A[nt, na]

		# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		for na in range(self.Na):
			for nt in range(self.Nt): self.La[nt+self.Nc, na, :] = self.gaussian( l2,sigma,Lmax[na])*self.A[nt, na]

		# Devolvemos (0-self.Da)Tensor de datos alineado (1-self.La)Perfiles cromatograficos alineados (2-self.L)Estimacion original de los perfiles cromatograficos
		print('Alineado exitoso')
		return self.Da, self.La, self.L # 
	
	def gaussian(self,n,s,m):
		return np.array((1.0/(s*(2*np.pi)**0.5)) * np.e**(-0.5*((np.arange(0, n, dtype=np.float32)-m)/s)**2))

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

	def G(self,data=None):
		def gaussian(mu, sigma, n, ):
			return np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2) / (np.linalg.norm(np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)))

		n = data.shape[0]
		mu_range = range(100, 350)
		sigma_range = range(1,30)
		base = np.array([ gaussian(mu, sigma, n) for mu in mu_range for sigma in sigma_range ])
		base_parameters_list = np.array([ [mu, sigma] for mu in mu_range for sigma in sigma_range ]) 

		proy = base.dot(data)
		max_arg   = np.argmax(proy)
		max_value = np.max(proy)

		return max_arg, max_value, base_parameters_list[max_arg][0], base_parameters_list[max_arg][1]
