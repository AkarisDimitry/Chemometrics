# *** warning supresion
import warnings
warnings.filterwarnings("ignore")
from functools import reduce
import logging, operator

# *** numeric libraries
import numpy as np #pip3 install numpy
try:
	import scipy
except: 
	print('WARNNING :: PARAFAC.py :: can NOT correctly load "scipy" libraries')

# *** graph libraries
try:
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pl
except: 
	print('WARNNING :: PARAFAC.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: (pip3 install matplotlib)')

class PARAFAC(object): # generador de datos
	def __init__(self, X=None, D=None, S=None, 
					C=None, A=None,
					Y=None, y=None,	
					N=None, Nc=None, Nt=None,
					f=None, Na=None, Ni=None, v=0):

		try: self.X = np.array(X)	
		except: print('WARNING :: code 400 PARAFAC.PARAFAC() :: can not load X data in to self.X')
		try: self.D = np.array(D)
		except: print('WARNING :: code 400 PARAFAC.PARAFAC() :: can not load X data in to self.D')

		if type(self.X) is np.ndarray and type(self.D) is np.ndarray and  self.X.shape != self.D.shape:
			self.D = self.X
			print('WARNING :: code 401 PARAFAC.PARAFAC() :: self.X.shape({}) != self.D.shape({})'.format(self.X.shape, self.D.shape) ) 
			print('WARNING :: making self.D = self.X')
			
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
		self.loadings = [] 	 	# loadings 
		self.Nloadings = []  	# normalized loadings 
		self.model_mse = None 		# MSE error 
		self.model_mse_list = []

		self.early_finish = {	'max_iter'	:	1000, 
								'conv_steps':	1e-20,
								'mse'		:	1e-40, }
		self.verbose = v

	def loading_estimation(self, x=None, nfactors=None, 
							max_iter=None, 
							inicializacion='random', constraints={}):
		try: x = x if type(x) != type(None) else self.X 
		except: pass
		
		try: max_iter = max_iter if type(max_iter) != type(None) else self.early_finish['max_iter']
		except: pass
		
		
		if not (type(nfactors) == int or type(nfactors) == float):
			if (type(self.f) == int or type(self.f) == float): 
				nfactors = int(self.f)
			elif (type(self.Na) == int or type(self.Na) == float) and (type(self.Ni) == int or type(self.Ni) == float):
				nfactors = int(self.Na + self.Ni)
				print('WARNING :: code 110A PARAFAC.loading_stimation() :: Setting to default value self.f = int(self.Na + self.Ni) ')
			else: 
				print('WARNING :: code 110A PARAFAC.loading_stimation() :: Setting to default value self.f = 1 ')
				nfactors = 1

		self.loadings, self.model_mse = self.parafac_base(x, nfactors, max_iter, inicializacion, constraints)
		s, self.Nloadings = self.normalized_loadings(self.loadings)
		return self.Nloadings
		 
	def train(self, x=None, nfactors=None, max_iter=10, inicializacion='random', constraints={}):
		if x == None: x = self.X 
		if nfactors == None: nfactors = self.Na + self.Ni

		self.loadings, self.model_mse = self.parafac_base(x, nfactors, max_iter, inicializacion, constraints)
		s, self.Nloadings = self.normalized_loadings(self.loadings)
		self.L = self.loadings
		
		return (s, self.Nloadings, self.model_mse)
		
	def ribs(self, loadings):
	  loadings = [np.atleast_2d(l) for l in loadings]
	  nfactors = loadings[0].shape[0]
	  assert np.alltrue([l.ndim == 2 and l.shape[0] == nfactors for l in loadings])
	  ribs = []
	  for mi in range(len(loadings)):
	    shape = [nfactors] + [-1 if fi == mi else 1 for fi in range(len(loadings))]
	    ribs.append(loadings[mi].reshape(shape))
	  return ribs

	def para_compose(self, ribs): return np.sum(reduce(operator.mul, ribs), axis=0)


	def parafac_base(self, x, nfactors, max_iter, inicializacion, constraints={'non-negativity':'all'}):
		# ---- X(l) = loadings ---- #
	  log = logging.getLogger('psychic.parafac')
	  if inicializacion == 'random': # inicializacion 
	  	loadings = [np.random.rand(nfactors, n) for n in x.shape]
	  else:
	  	loadings = [np.random.rand(nfactors, n) for n in x.shape]
	  	loadings[0][:inicializacion[0].shape[0], :inicializacion[0].shape[1] ] = inicializacion[0]
	  	for n in range(1, len(inicializacion) ): loadings[n][:inicializacion[n].shape[0],:] = inicializacion[n]
	  
	  # ---- Check of convergence---- #
	  last_mse = np.inf
	  for i in range(max_iter):
	    # 1) forward (predict x)
	    xhat = self.para_compose(self.ribs(loadings))
	    # 2) stopping?
	    mse = np.mean((xhat - x) ** 2)
	    if last_mse - mse < self.early_finish['conv_steps'] or mse < self.early_finish['mse']:
	      break
	    last_mse = mse
	    self.model_mse_list.append(mse)

	    # --- Channel mode --- #
	    for mode in range(len(loadings)): 
	      log.debug('iter: %d, dir: %d' % (i, mode))
	      # a) Re-compose using other factors
		# ---- Z ---- #
	      Z = self.ribs([l for li, l in enumerate(loadings) if li != mode])
	      Z = reduce(operator.mul, Z)
	      # b) Isolate mode
	      Z = Z.reshape(nfactors, -1).T # Z = [long x fact]

	    # ---- Yf ---- #
	      Y = np.rollaxis(x, mode)
	      Y = Y.reshape(Y.shape[0], -1).T # Yf = [lf, long]

	    # ---- model ---- # Zn . Xn(l) = Yf --> Xn(l) = Yf . Zn**-1
	      # c) alternating least squares estimation: x = np.lstsq(Z, Y) -> Z x = Y
	      
	      new_fact, _, _, _ = np.linalg.lstsq(Z, Y, rcond=-1)
	      loadings[mode] = new_fact

	    # ---- Model restrictions ----
	      loadings = self.restrictions(loadings, constraints)

	  if not i < max_iter - 1: # --- Max Iterations --- # 
	    log.warning('parafac did not converge in %d iterations (mse=%.2g)' %
	      (max_iter, mse))

	  return loadings, mse

	def normalized_loadings(self, loadings=None):
		if not loadings == None: loadings = self.loadings 
		mags = np.asarray([np.apply_along_axis(np.linalg.norm, 1, mode)
		for mode in loadings])

		norm_loadings = [loadings[mi] / mags[mi].reshape(-1, 1) 
		for mi in range(len(loadings))]

		mags = np.prod(mags, axis=0)
		order = np.argsort(mags)[::-1]
	 
		return mags[order], [np.asarray([mode[fi] for fi in order]) for mode in norm_loadings]

	def restrictions(self, loadings, constraints):
		for key, value in constraints.items():
			if key == 'non-negativity':
				if (type(value) == str and value == 'all') or (type(value) == list and len(value)>0 and value[0] == 'all' ):
					for i, n in enumerate(loadings): loadings[i] = np.where(loadings[i]>0, loadings[i], 0)
				else:
					for i, n in enumerate(value): loadings[n] = np.where(loadings[n]>0, loadings[n], 0)

		return loadings

	def predic(self, Nc=None, Na=None, Ni=None, N=None, f=None, v=0):
		if not type(N) is np.ndarray and not type(N) is int and N == None and (type(self.N) is np.ndarray or type(self.N) is int): N = self.N
		else: print('ERROR :: code 050 :: PARAFAC.predic() :: can NOT get N({}) or self.N({})'.format(N, self.N))
		if not type(Nc) is np.ndarray and not type(Nc) is int and Nc == None and (type(self.Nc) is np.ndarray or type(self.Nc) is int): Nc = self.Nc
		else: print('ERROR :: code 050 :: PARAFAC.predic() :: can NOT get Nc({}) or self.Nc({})'.format(Nc, self.Nc))

		if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
		else: print('ERROR :: code 050 :: PARAFAC.predic() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
		if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
		else: print('ERROR :: code 050 :: PARAFAC.predic() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		else: print('ERROR :: code 050 :: PARAFAC.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		else: print('ERROR :: code 050 :: PARAFAC.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if not type(self.y) is np.ndarray or not self.y.shape == (self.f, self.X.shape[0]): 
			if type(self.y) is np.ndarray:	print('WARNING :: code 1000 PARAFAC.predic() :: Previuos data store in self.y will be erased. ')
			self.y = np.zeros((self.f, self.X.shape[0]))

		if v >= 1:	print(' (1) Predic tensor coeficients.')
		self.A = self.Nloadings[0]

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

		print(self.A.shape, self.Y.shape)
		for n in range(self.Na):
			for m in range(self.Na+self.Ni):
				z = np.polyfit( self.A[int(m), :self.Nc] , self.Y[:self.Nc, int(n)], 1)
				self.model_prediction = self.A[int(m), :]*z[0]+z[1]
				predic = self.model_prediction
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
					print('Test{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
														(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )

		return None
		
	def plot_loadings(self, Nc=None, Na=None, Ni=None, fig=None):

		try:
			if fig == None:	fig = plt.figure()
			else: fig=fig
		except:
			print('ERROR :: code 020b PARAFAC.plot_loadings() :: can NOT generate fig obj')
		
		#try: 	ax = fig.add_subplot(100*len(self.Nloadings)+11+100)
		#except:	print('ERROR :: code 020c PARAFAC.plot_loadings() :: can NOT generate axis, maybe fig argument is NOT what except ')
				
		for i, loading in enumerate(self.Nloadings): 
			# ***** PLOT loading ***** #
			try: 	ax = fig.add_subplot(len(self.Nloadings)*100+11+i)
			except:	print('ERROR :: code 020c PARAFAC.plot_loadings() :: can NOT generate axis, maybe fig argument is NOT what except ')
			for j, vector in enumerate(loading):
				ax.plot(vector, color=self.colors[j], lw=1.5, alpha=0.6, marker='o', markersize=3, label='Factor {}'.format(j+1) )


			try:
				ax.set_xlabel('Variable')
				ax.set_ylabel('Intencity')
				#ax.set_title('Factor plot')
				#ax.set_xlabel('variable', fontsize=12);		ax.set_ylabel('Absorbance (a.u.)', fontsize=12)
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

		return None

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

				try:
					if fig == None:	fig = plt.figure()
					else: fig=fig
				except:
					print('ERROR :: code 020b DATA.plot_3d() :: can NOT generate fig obj')

				try: 	
					ax1 = fig.add_subplot(211)
					if error_plot == 'sample': 	ax2 = fig.add_subplot(212) # ** plot sample error ** #	
					if error_plot == 'cc': 		ax2 = fig.add_subplot(212, sharex=ax1) # ** plot cc error ** #	
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

		except:	print('ERROR :: code 050 :: GPV.plot_45() :: can NOT plot 45')

	def plot_convergence(self, ):
		fig = plt.figure()
		ax = fig.add_subplot()

		ax.set_xlabel('Step')
		ax.set_ylabel('MSE')
		ax.set_title('Convergence')

		plt.plot( [0, len(self.model_mse_list)] , [0,0], ':', color=(0,0,0), alpha=0.4,  )
		plt.plot( self.model_mse_list, '-o', color=(0.8,0.4,0.4) )


