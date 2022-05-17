import numpy as np
import matplotlib.pyplot as plt

class FPAV32(object):
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
		
		self.colors = [ '#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',] 

	def aling(self, D=None, S=None, Nc=None, Na=None, Ni=None, a=None,
					area='gaussian_coeficient', non_negativity=True, linear_adjust=True,
					mu_range=None, sigma_range=None, save=True
					): 
		# variable setter for aling_FAPVXX  
		try: 
			if not D  == None: 	self.D  = D
		except:	pass

		try: 
			if not S  == None: self.S  = S
		except:	pass

		try: 
			if save: self.mu_range = mu_range
		except: pass

		try: 
			if save: self.sigma_range = sigma_range
		except: pass

		if not Nc == None: self.Nc = Nc
		if not Na == None: self.Na = Na
		if not Ni == None: self.Ni = Ni
		if not a  == None: self.a  = a
		
		return self.Da, self.La, self.L
	
	def aling_FAPV32(self, D=None, S=None, Nc=None, Na=None, Ni=None, save=True):
		# ----- Algoritmo de alineado para datos de 3er orden con 2 canales alineados ----- #
		# 
		# self.D 		:	N-MAT 		:		N-array/Matrix de datos no alineados D => [N,l1,l2,l3]
		# self.S 		:	N-MAT 		:		N-array/Matrix perfiles de los canales alineados  S => [[self.f,l1],[self.f,l2]] 
		# self.Nc		:	INT 		: 		Numero de muestras de calibrado.
		# self.Na		:	INT 		: 		Numero analitos.
		# self.Ni		:	INT 		: 		Numero de interferentes.
		# ---------------------------------------------------------------------------------- #
		# self.S  	: [L1, L2] | L1=[f,l1], L2=[f,l2] 
		# self.f  	: numero de factores = Numero de analitos(Na) + Numero de interferentes(Ni)
		# self.N  	: Numero de muestras de calibrado(Nc) + Numero de muestras test(Nt)
		# l1 		: numero de sensores del canal 1
		# l2 		: numero de sensores del canal 2
		# l3 		: numero de sensores del canal 3

		# --- Defino vaiables --- # 
		try: 
			if not D  == None: self.D  = D
		except:	pass
		try: 
			if not S  == None: self.S  = S
		except:	pass
		if not Nc == None: self.Nc = Nc
		if not Na == None: self.Na = Na
		if not Ni == None: self.Ni = Ni

		self.N, l1, l2, l3 =  self.D.shape
		self.Nt, self.f = self.N-self.Nc, self.Na + self.Ni

			# ------ ------ MUESTRAS DE CALIBRADO ------ ------ #
		# --- Z --- #
		Ze = np.zeros( (self.Na, l1, l2) )
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
		print('Successful alignment.')

		return self.Da, self.La, self.L 
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






