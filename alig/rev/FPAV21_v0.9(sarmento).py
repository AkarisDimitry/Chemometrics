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
		

		if self.a == 1: 
			# if 3 way 
			self.aling_FAPV21(	D=self.D, S=self.S, Nc=self.Nc, Na=self.Na, Ni=self.Ni, 
								area=area, non_negativity=non_negativity, linear_adjust=linear_adjust,
								mu_range=mu_range, sigma_range=sigma_range, save=save)

		return self.Da, self.La, self.L # 
		
	def aling_FAPV21(self, D=None, S=None, Nc=None, Na=None, Ni=None, v=1, 
					area='gaussian_coeficient', non_negativity=True, linear_adjust=True,
					mu_range=None, sigma_range=None, save=True
					):
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

		if non_negativity: 	Lc = np.where(Lc>0, Lc, 0) 			

		self.L = np.zeros( (self.Nc, self.Na, l2) )
		for nc in range(self.Nc): self.L[nc,:,:] = Lc[nc*l2:(nc+1)*l2,:].T

		#plt.figure(1), plt.plot( self.L[:,0,:].T, 'b' ) # !!!!! sarmento !!!!!
		#plt.figure(2), plt.plot( self.L[:,1,:].T, 'b' ) # !!!!! sarmento !!!!!

		# ------ AREAS ------ #
		if area == 'sum':
			self.A = np.sum( self.L, axis=2 ) # [N, self.f]

		if area == 'max':
			self.A = np.max( self.L, axis=2 )  

		lww = 1.5#  !!!!!!!!!!!!!!!!!!!!!!!!
		#plt.figure(0), plt.plot(self.L[:,0,:].T, alpha=0.4, c='#DD2222', lw=lww) # !!!!!!!!!!!!!!!!!!!!!!!!
		#plt.figure(1), plt.plot(self.L[:,1,:].T, alpha=0.4, c='#22DD22', lw=lww) #  !!!!!!!!!!!!!!!!!!!!!!!!

		max0 = np.max(  self.L[:,0,:], )#  !!!!!!!!!!!!!!!!!!!!!!!!
		max1 = np.max(  self.L[:,1,:], )#  !!!!!!!!!!!!!!!!!!!!!!!!
		for n in range( self.L[:,0,:].shape[0] ):#  !!!!!!!!!!!!!!!!!!!!!!!!
			color0 = 0.4 + np.max(  self.L[n,0,:], ) / max0 * 0.6#  !!!!!!!!!!!!!!!!!!!!!!!!
			color1 = 0.4 + np.max(  self.L[n,1,:], ) / max1 * 0.6#  !!!!!!!!!!!!!!!!!!!!!!!!
			plt.figure(0), plt.plot(self.L[n,0,:].T, alpha=0.4, c=(color0,0.2,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!
			plt.figure(1), plt.plot(self.L[n,1,:].T, alpha=0.4, c=(0.2,color1,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!


		if area=='gaussian_coeficient':
			coef = np.zeros((self.Nc+self.Nt, self.Na+self.Ni))
			Gcoef = np.zeros((self.Nc+self.Nt, self.Na+self.Ni, 2))
			for nc in range(self.Nc):
				for na in range(self.Na):
					CC = self.G( self.L[nc,na,:], mu_range=mu_range, sigma_range=sigma_range)
					coef[nc, na] = CC[1]
					Gcoef[nc, na, :] = CC[2], CC[3] 
					if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
			self.A = coef

		if area=='gaussian_coeficient':
			self.all_gauss_aling = np.zeros((self.Nc+self.Nt, self.Na+self.Ni, l2))
			for na in range(self.Na):
				for nc in range(self.Nc): 
					self.all_gauss_aling[nc, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l2) *self.A[nc, na] 

			self.all_gauss = np.zeros((self.Nc+self.Nt, self.Na+self.Ni, l2))
			for na in range(self.Na):
				for nc in range(self.Nc): 
					self.all_gauss[nc, na, :] = self.gaussian(Gcoef[nc, na, 0], Gcoef[nc, na, 1], l2) *self.A[nc, na] 

			# ------ ALINEADO FUNCIONAL ------ # Estimacion del tensor de datos alineado (self.Da)
		if v >= 1: print( ('3- Functional alignement (calibration samples)') )
		Lmax, Amax, self.Da, sigma = np.zeros((self.Na)), np.argmax(self.A, axis=0), np.zeros((self.N, l1, l2)), l2/self.f/30
		for na in range(self.Na): Lmax[na] = np.argmax(self.L, axis=2)[Amax[na]][na]
		# choose specific center for the gaussian for each analyte
		# Lmax = [250, 150]  
		# choose specific SD for the gaussian for each analyte
		# sigma = 10 

		if v >= 1: print( ('4- Recostructing data (calibration samples)') )
		for na in range(self.Na):
			Ma = np.tensordot(self.S[0][na, :], self.gaussian(Lmax[na], sigma, l2) , axes=0)
			for nc in range(self.Nc): self.Da[nc,:,:] += Ma*self.A[nc, na]

		# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		self.La = np.zeros( (self.N, self.f, l2) )
		for na in range(self.Na):
			for nc in range(self.Nc): self.La[nc, na, :] = self.gaussian(Lmax[na], sigma, l2)*self.A[nc, na]

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
			Lc= np.dot(Y.T, np.linalg.pinv(Z) )
			if non_negativity: 	Lc = np.where(Lc>0, Lc, 0) 

			self.L = np.zeros( (self.Nt, self.f, l2) )
			for nt in range(self.Nt): self.L[nt,:,:] = Lc[nt*l2:(nt+1)*l2,:].T 
		
			#plt.figure(1), plt.plot( self.L[:,0,:].T*0.79, 'r' ) # !!!!! sarmento !!!!!
			#plt.figure(2), plt.plot( self.L[:,1,:].T*0.185, 'r' ) # !!!!! sarmento !!!!!
			#plt.show()

			# ------ AREAS ------ #
			if area == 'sum':
				self.A = np.sum( self.L, axis=2 ) # [N, self.f]
				#self.A[:,0] = self.A[:,0]*0.836
				#self.A[:,1] = self.A[:,1]*0.209

			if area == 'max':
				self.A = np.max( self.L, axis=2 )  
				#self.A[:,0] = self.A[:,0]*0.836
				#self.A[:,1] = self.A[:,1]*0.209

			if area=='gaussian_coeficient':
				coef = np.zeros((self.Nt, self.f))
				Gcoef = np.zeros((self.Nt, self.f, 2))
				for nc in range(self.Nt):
					for na in range(self.f):
						CC = self.G( self.L[nc,na,:], mu_range=mu_range, sigma_range=sigma_range )
						coef[nc, na] = CC[1]
						Gcoef[nc, na, :] = CC[2], CC[3] 
						if v >= 1: print('Gcoef: {:e} mead:{} sd:{} :: sample: {} factor: {} '.format(CC[1], CC[2], CC[3], nc, na))
				self.A = coef
				self.A[:,0] = self.A[:,0]*0.79
				self.A[:,1] = self.A[:,1]*0.185

			if linear_adjust:
				pass

			if area=='gaussian_coeficient':
				for na in range(self.Na+self.Ni):
					for nt in range(self.Nt): 
						self.all_gauss_aling[self.Nc+nt, na, :] = self.gaussian(np.mean(Gcoef[:, na, 0]), np.mean(Gcoef[:, na, 1]), l2) *self.A[nt, na] 

				for na in range(self.Na+self.Ni):
					for nt in range(self.Nt): 
						self.all_gauss[self.Nc+nt, na, :] = self.gaussian(Gcoef[nt, na, 0], Gcoef[nt, na, 1], l2) *self.A[nt, na] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			#plt.figure(0), plt.plot(self.L[:,0,:].T*0.790, alpha=0.4, c='#DD2222', lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!
			#plt.figure(1), plt.plot(self.L[:,1,:].T*0.185, alpha=0.4, c='#22DD22', lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!


			max0 = np.max(  self.L[:,0,:], )
			max1 = np.max(  self.L[:,1,:], )
			for n in range( self.L[:,0,:].shape[0] ):
				color0 = 0.4 + np.max(  self.L[n,0,:]*0.790, ) / max0 * 0.6
				color1 = 0.4 + np.max(  self.L[n,1,:]*0.185, ) / max1 * 0.6
				#plt.figure(2), plt.plot(self.La[n,0,:].T, alpha=0.5, c=(color0,0.2,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!
				#plt.figure(3), plt.plot(self.La[n,1,:].T, alpha=0.5, c=(0.2,color1,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!
				plt.figure(0), plt.plot(self.L[n,0,:].T*0.790, alpha=0.4, c=(color0,0.2,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!
				plt.figure(1), plt.plot(self.L[n,1,:].T*0.185, alpha=0.4, c=(0.2,color1,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!



			# --- ALINEADO FUNCIONAL --- # Estimacion del tensor de datos alineado (self.Da)
			# must be define before sigma = l2/self.f/6  # Amax = np.argmax(self.A, axis=0)
			if v >= 1: print( ('4b- Recostructing data (test samples)') )
			for na in range(self.Na):
				Ma = np.tensordot( self.S[0][na, :], self.gaussian(Lmax[na], sigma, l2) , axes=0)
				for nt in range(self.Nt): self.Da[self.Nc+nt,:,:] += Ma*self.A[nt, na]

		# --- Estimacion de los canales originalemente NO alineados(self.L) ya alineados(self.La).
		for na in range(self.Na):
			for nt in range(self.Nt): 
				self.La[nt+self.Nc, na, :] = self.gaussian(Lmax[na], sigma, l2)*self.A[nt, na]
				self.La[nt+self.Nc, na, :] = self.gaussian(Gcoef[nt, na, 0], sigma, l2)*self.A[nt, na]

		max0 = np.max(  self.La[:,0,:], )#!!!!!!!!!!!!!!!!!!!!!
		max1 = np.max(  self.La[:,1,:], )#!!!!!!!!!!!!!!!!!!!!!
		for n in range( self.La[:,0,:].shape[0] ):#!	!!!!!!!!!!!!!!!!!!!!!
			color0 = 0.4 + np.max(  self.La[n,0,:], ) / max0 * 0.6#!!!!!!!!!!!!!!!!!!!!!
			color1 = 0.4 + np.max(  self.La[n,1,:], ) / max1 * 0.6#!!!!!!!!!!!!!!!!!!!!!
			plt.figure(2), plt.plot(self.La[n,0,:].T, alpha=0.5, c=(color0,0.2,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!
			plt.figure(3), plt.plot(self.La[n,1,:].T, alpha=0.5, c=(0.2,color1,0.2), lw=lww) 	# !!!!!!!!!!!!!!!!!!!!!!!!

		
		

		# Devolvemos (0-self.Da)Tensor de datos alineado (1-self.La)Perfiles cromatograficos alineados (2-self.L)Estimacion original de los perfiles cromatograficos
		print('Alineado exitoso')
		
		return self.Da, self.La, self.L # 
	
	def gaussian(self, mu, sigma, n, ):
			return np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2) / (np.linalg.norm(np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)))

	def Lorentz(self,n,a,m):
		return (a*1.0/(np.pi*((np.arange(0, n, dtype=np.float32)-m))**2+a**2))

	def plot(self, names):
		color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', ]
		
		#for n in range(2):
			#plt.plot(self.L[:,n,:].T, '-o', color=color[n])
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
		except:		print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not set colors')

		try:		fig, ax = plt.subplots()
		except:		print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not make axes')

		try:
			for n in range(self.Na):
				if names != None:	ax.plot( self.S[0][n,:].T, '-o', color=color[n], lw=1.5, alpha=0.6, ms=3, label=names[n])
				else:				ax.plot( self.S[0][n,:].T, '-o', color=color[n], lw=1.5, alpha=0.6, ms=3 )
		except:		print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not PLOT spectra')

		try:
			ax.set_xlabel('variable', fontsize=12);		ax.set_ylabel('Absorbance (a.u.)', fontsize=12)
			ax.spines["right"].set_linewidth(2); 		ax.spines["right"].set_color("#333333")
			ax.spines["left"].set_linewidth(2); 		ax.spines["left"].set_color("#333333")
			ax.spines["top"].set_linewidth(2); 			ax.spines["top"].set_color("#333333")
			ax.spines["bottom"].set_linewidth(2); 		ax.spines["bottom"].set_color("#333333")
		except:		print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not configure axis')


		try:
			if names != None:
				legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
				legend.get_frame().set_facecolor('#FFFFFF')
		except:		print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not configure legend')

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
			coef[n, :, :, :] = G_matrix_descompose( data=data[n, 30:32, :], G_max=G_max )

		return coef



