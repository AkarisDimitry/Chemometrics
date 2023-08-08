# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== IMPORT libraries ====================  IMPORT libraries ==================== # # ==================== IMPORT libraries ====================  IMPORT libraries ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# *** warning supresion
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries *** #
import numpy as np
import scipy.io
#from scipy.stats import entropy

# *** graph libraries *** #
try:
	import matplotlib.pyplot as plt
	from mpl_toolkits import mplot3d
	import matplotlib as mpl
except: 
	print('WARNNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: ( pip3 install matplotlib )')
	
# *** python common libraries
import logging, operator, pickle, os

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== Obj  ====================  Obj  ==================== # # ==================== Obj ====================  Obj ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
class DATA(object): # generador de datos
	def __init__(self, X=None, D=None, Y=None, y=None, S=None, A=None, a=None,
				f=None, N=None, Nc=None, Nv=None, Nt=None, Na=None, Ni=None, 
				L=None, La=None, Ldim=None, ways=None,
				model=None, constraints=None, aling_model=None):
		# === DATA === #
		self.X = np.array(X) if not type(X) == type(None) else None
		self.Y = np.array(Y) if not type(Y) == type(None) else None

		self.D = D
		
		self.y = y
		self.S = S # MAT-32 || [] all aligned channels modes for all factors   

		# === PARAMETERS === #
		self.L = L
		self.loadings = self.L
		self.La = La
		self.Ldim = Ldim # list of loadings dimentions

		self.a = a # INT-8 || total number off well aligned ways  
		self.aligned_channel = None
		self.A = A # INT-8 || Total cumulated area

		self.N = N   # INT-8 || dataset length 
		self.Nc = Nc # INT-8 || number of training sample
		self.Nv = Nv # INT-8 || number of validation sample
		self.Nt = Nt # INT-8 || number of test-samples 

		self.f = f   # INT-8 || total number of factors   
		self.Na = Na # INT-8 || total number of analites 
		self.Ni = Ni # INT-8 || total number of interferents 

		self.ways = ways # toltal data tensor order

		# === MODELs === #
		# trainning model variables #
		self.model = model
		self.constraints = constraints
		self.model_train = None

		# aling model variables #
		self.aling_model = aling_model

		# === MODEL selection === #
		self.model_selection = None

		self.colors = [ '#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',] 
		self.font = {
				'family': 'serif',
        		'color':  (0.2, 0.2, 0.2, 1.0),
        		'weight': 'normal',
        		'size': 16,
        		}

		self.f = self.Na + self.Ni if type(self.f) != type(None) and type(self.Na) != type(None) and type(self.f) != type(None) else self.f

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== TRAIN ====================  TRAIN ==================== # # ==================== TRAIN ====================  TRAIN ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	def train(self, K=None, constraints=None, Inicializate=None, save=True, v=0):
		# *** *** *** Check variable integrity *** *** *** #
		try: # Full data set self.X || TENSOR ||  
			if type(self.X) == type(None): 
				if v: print('WARNNING :: DATA.train() :: type(self.X) == {} || self.X must be nparray.'.format(type(self.X)))
		except: pass
		
		try: # response variables self.y || Matrix ||
			if not self.y == None: self.y = y
		except: pass

		self.Ni = self.Ni if not self.Ni == None else 0
		self.Na = self.Na if not self.Na == None else 0

		if self.N == None: print('WARNING :: internal data frame inconsistency :: self.N == None')
		self.model.N  = self.N  if not self.N  == None else 0
		
		if self.Nc == None: print('WARNING :: internal data frame inconsistency :: self.Nc == None')
		self.model.Nc = self.Nc if not self.Nc == None else 0
		
		if self.Na == None: print('WARNING :: internal data frame inconsistency :: self.Na == None')
		self.model.Na = self.Na if not self.Na == None else 0

		if self.Ni == None: print('WARNING :: internal data frame inconsistency :: self.Ni == None')
		self.model.Ni = self.Ni if not self.Ni == None else 0

		self.model.f = self.Na+self.Ni if 'f' in self.model.__dict__.keys() else 0
		if self.Na+self.Ni != self.f: print('WARNING :: internal data frame inconsistency :: self.Na+self.Ni != self.f')

		if self.a == None: print('WARNING :: internal data frame inconsistency :: self.Ni == None')
		self.model.a = self.a if 'a' in self.model.__dict__.keys() and self.a != None else 1

		# *** *** *** Train model *** *** *** #
		if constraints != None:	model_train = self.model.train(constraints=constraints, )
		else: model_train = self.model.train()

		# *** *** *** Save trainning *** *** *** #
		if save: self.model_train = model_train

		return model_train

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== ALING ====================  ALING ==================== # # ==================== ALING ====================  ALING ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	def aling(self, L=None, loadings=None, K=None):
		#if not self.X == None: self.model.X = self.X
		#if not self.y == None: self.model.y = self.y
		#if not self.N == None: self.model.N = self.N
		#if not self.Nc == None: self.model.Nc = self.Nc
		#if not self.Na == None: self.model.Na = self.Na
		#if not self.Ni == None: self.model.Ni = self.Ni
		if type(self.L) is list and self.L != []:
			self.aling_model.aling()
		else:
			self.aling_init()
			self.aling_model.aling()

	def aling_init(self, L=None, init_mode='zeros'):
		if type(self.X) is np.ndarray and not self.X.all() == np.array(None) and type(self.D) is np.ndarray and not self.D.all() == np.array(None):

			try:
				if self.a == None:
					self.a = len(self.X.shape)
					print('WARNING :: code 100A DATA.aling_init() :: Number of well aligned channels is NOT especified (self.a == None)...')
					print('WARNING :: code 100A DATA.aling_init() :: Setting to default value ( asuming all channels are well alingned ). self.a = len(self.X)-1 ')
			except:	
				print('ERROR :: code 100A DATA.aling_init() :: Can not obtein self.a from data')
				pass
			
			try:
				if (self.f == 0 or self.f == None) and self.Na != None and self.Ni != None: 
					self.f = self.Na + self.Ni
					print('WARNING :: code 100B DATA.aling_init() :: Number of factors is NOT especified (self.f == None)...')
					print('WARNING :: code 100B DATA.aling_init() :: Setting to default value. self.f = self.Na + self.Ni')
				elif self.f == None:
					self.f = 1
					print('WARNING :: code 100B DATA.aling_init() :: Number of factors is NOT especified (self.f == None)...')
					print('WARNING :: code 100B DATA.aling_init() :: Setting to default value. self.f = 1')
			except:	
				print('ERROR :: code 100B DATA.aling_init() :: Can not obtein self.f from data')
				pass

			try:
				if self.L == None:
					self.L = []
					print('WARNING :: code 100C DATA.aling_init() :: (self.L == None)...')
					print('WARNING :: code 100C DATA.aling_init() :: Setting to default value self.L = [] ')
			except:	
				print('ERROR :: code 100C DATA.aling_init() :: Can not obtein self.L ')
				pass
			try:
				for i, n in enumerate(self.X.shape):
					if self.a <= i: 
						self.L.append( np.zeros( (n, self.f) ) )
					else:
						self.L.append( np.zeros( (n*self.X.shape[0], self.f) ) )	
				self.loadings = self.L
				return self.L

			except: return None
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== PREDIC ====================  PREDIC ==================== # # ==================== PREDIC ====================  PREDIC ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	def predic(self,  K=None, v=1):
		if not type(self.X) is np.ndarray: self.model.X = self.X
		#if not self.X == None: self.model.X = self.X
		if not type(self.X) is np.ndarray: self.model.y = self.y
		if not self.N == None: self.model.N = self.N
		if not self.Nc == None: self.model.Nc = self.Nc
		if not self.Na == None: self.model.Na = self.Na
		if not self.Ni == None: self.model.Ni = self.Ni

		if 'f' in self.model.__dict__.keys(): self.model.f = self.Na+self.Ni
		if 'a' in self.model.__dict__.keys() and self.a != None: self.model.a = self.a

		if type(v) != int: v=0
		y = self.model.predic(v=v)
		self.y = y 	# store predicted signal in self.y

		return y 	# return predicted signal

	def summary(self, X=None, Y=None, y=None, Nc=None, Nt=None, Na=None, Ni=None, N=None, ways=None, f=None, S=None):
		# -- Check data integroty -- #
		try: 	X = self.X if type(X) == type(None) and type(self.X) != type(None) else X 
		except: pass
		try: 
			if type(D) == type(None) and type(self.D) != type(None) : X = self.D
		except: pass

		try:
			if type(Y) == type(None) and type(self.Y) != type(None) : Y = self.Y
		except: pass

		try:
			if type(y) == type(None) and type(self.y) != type(None) : y = self.y
			elif type(self.model.y) != type(None): y = self.model.y
		except: pass


		try:
			if type(N) == type(None) and type(self.N) != type(None) : N = self.N
		except: pass

		try:
			if type(Nc) == type(None) and type(self.Nc) != type(None) : Nc = self.Nc
		except: pass

		try:
			if type(Nt) == type(None) and type(self.Nt) != type(None) : Nt = self.Nt
		except: pass

		try:
			if type(Na) == type(None) and type(self.Na) != type(None) : Na = self.Na
		except: pass

		try:
			if type(Ni) == type(None) and type(self.Ni) != type(None) : Ni = self.Ni
		except: pass

		try:
			if type(f) == type(None) and type(self.f) != type(None) : f = self.f
			elif self.isnum(Na) and self.isnum(Ni): f = Na+Ni 
		except: pass

		try:
			if type(ways) == type(None) and type(self.ways) != type(None) : ways = self.ways
			elif type(self.D) != type(None) :  ways = len(self.D.shape)
		except: pass

		try:
			if type(S) == type(None) and type(self.S) != type(None) : S = self.S
		except: pass

		# -- start summary -- #
		print('{} SUMMARY {} '.format('='*10, '='*10) )

		# ==================== ==================== ==================== ==================== === # 
		# ====================   MetaData  ====================   MetaData   ==================== # 
		# ==================== ==================== ==================== ==================== === # 

		try: # *** ----------- ways data analisys ----------- *** #
			atribute_dict = {	'Number of calibrated factors': 	Na, 
								'Number of NON-calibrated factors': Ni, 
								'Number of factors':				f,
								'Number of calibration samples':	Nc, 
								'Number of test samples':			Nt, 
								'Number of Samples':				N}

			print(' * Metadata loaded :: summary' )	
			for (key,value) in atribute_dict.items():
				print(' \t * {} : {}'.format( key, value ))
		except: print('WARNNING :: code 051 DATA.summary() :: Can not summarise Meta Data')

		# ==================== ==================== ==================== ==================== === # 
		# ====================   Spectra   ====================   Spectra    ==================== # 
		# ==================== ==================== ==================== ==================== === # 

		try:
			print(' * Spectra Text Plot ')
			heigth = 10
			lenth = 60
			for channel in range( len(S) ):
				print(' \t * channel {}'.format(channel) )
				if type(S[channel]) == list: S[channel] = np.array(S[channel])
				for sn in range( S[channel].shape[1] ): 
					spectra_disc = np.array([ S[channel][n,sn] for n in range(0, S[channel].shape[0], int(S[channel].shape[0]/lenth)) ])
					spectra_norm = np.array(heigth * spectra_disc / np.max(spectra_disc), np.dtype(np.int) )
					print('\n'.join([ ''.join(['#' if heigth-h<n+1 else ' ' for n in spectra_norm ]) for h in range(heigth) ]))
					print(' --- --- --- espectra {} --- --- --- '.format(sn) )
		except: print('WARNNING :: code 051 DATA.summary() :: Can not summarise spectra')

		# ==================== ==================== ==================== ==================== === # 
		# ==================== X X X X X X ====================  X X X X X X ==================== # 
		# ==================== ==================== ==================== ==================== === # 

		try: # *** ----------- ways data analisys ----------- *** #
			if type(ways) is np.ndarray or type(ways) is int:
				print(' * {} way data loaded'.format(str(ways)) )
		except: print('WARNNING :: code 051 DATA.summary() :: Can not summarise self.ways')

		try: # *** ----------- X data analisys ----------- *** #
			if type(X) is np.ndarray:
				print(' * X data loaded :: shape {}'.format(str(X.shape)) )
				print(' \t * number of sample {}'.format(str(X.shape[0])) )
				for cnumber, cvalue in enumerate(X.shape):
					if cnumber > 0: print(' \t * number of variables in channel {} : {}'.format(cnumber, cvalue) )
		except: print('WARNNING :: code 051b DATA.summary() :: Can not summarise self.X ')

		# ==================== ==================== ==================== ==================== === # 
		# ==================== Y Y Y Y Y Y ====================  Y Y Y Y Y Y ==================== # 
		# ==================== ==================== ==================== ==================== === # 

		try: # *** ----------- Y data analisys ----------- *** #
			if type(Y) is np.ndarray and len(Y.shape)==2:
					print(' * Y data loaded :: shape {}'.format(str(Y.shape)) )
					for i in range(Y.shape[0]): 
						temporal_text = ''
						for j in range(Y.shape[1]):	temporal_text += '{:<.6f} \t'.format(Y[i,j])
						print(' ({}) \t {}'.format(i, temporal_text) )
			elif len(Y.shape)==2:
				print('WARNING :: code 051a DATA.summary() :: Can not summarise self.Y ')

			print(' * Sample Text Plot ')
			heigth = 6
			for a in range(Y.shape[1]):
				y_norm = np.array(heigth * Y[:, a] / np.max(Y[:, a]), np.dtype(np.int) )
				print('\n'.join([ '\t'.join(['#' if heigth-h<n+1 else ' ' for n in y_norm ]) for h in range(heigth) ]))
				print( '\t'.join([ '{}'.format(n+1) for n in range(Y.shape[0])]) )

		except: print('WARNING :: code 051a DATA.summary() :: Can not summarise self.Y ')

		# ==================== ==================== ==================== ==================== === # 
		# ==================== y y y y y y ====================  y y y y y y ==================== # 
		# ==================== ==================== ==================== ==================== === # 
		try: # *** ----------- y data analisys ----------- *** #
			if not type(y) is np.ndarray:
				if type(y) == list: y = np.array(y)
				else:
					try: 
						if type(self.model.y) is np.ndarray: y = self.model.y
					except: print('WARNING :: code 051c DATA.summary() :: Can not summarise self.y can not get array from self.y or self.model.y')

			if type(y) is np.ndarray and len(y.shape)==2:
					print(' * y data loaded :: shape {}'.format(str(y.shape)) )
					for i in range(y.shape[1]): 
						temporal_text = ''
						for j in range(y.shape[0]):	temporal_text += '{:<.6f} \t'.format(y[j,i])
						print(' ({}) \t {}'.format(i, temporal_text) )
			elif len(Y.shape)==2:
				print('WARNING :: code 051c DATA.summary() :: Can not summarise self.y can not get array from self.y or self.model.y')

		except: print('ERROR :: code 051c DATA.summary() :: Can not summarise self.y can not get array from self.y or self.model.y')


		try: # *** y data analisys *** #

			if not type(y) is np.ndarray:
				try: 
					if type(self.model.y) is np.ndarray: y = self.model.y
				except: print('WARNING :: code 051c DATA.summary() :: Can not summarise self.y can not get array from self.y or self.model.y')

			if type(y) is np.ndarray and len(y.shape)==2 and type(Y) is np.ndarray :
				for a in range(y.shape[0]): 
					print(' {} Factor {} {}'.format('*'*30, int(a+1), '*'*30) )
					print('Sample \t\t\t Y \t\t\t y(stimation) \t\t\t Error \t\t\t e% \t\t\t ')
					for n in range(y.shape[1]):
						if Nc == n:
							print(' ---------------- '*5)
						
						if Nc > n: # plot CALIBRATION sample data #
							print('Cal{} \t\t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f}'.format(n+1, Y[n, a], y[a, n],  
																(Y[n, a]-y[a, n]),  (Y[n, a]-y[a, n])*100/Y[n, a] 	) )
						else:	# plot TEST sample data #
							if n < Y.shape[0]:	# when have test sample CC.
								print('Test{} \t\t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f}'.format(n+1, Y[n, a], y[a, n],  
																	(Y[n, a]-y[a, n]),  (Y[n, a]-y[a, n])*100/Y[n, a] ) )

							if n >= Y.shape[0]: # when do NOT have test sample CC.
								print('Test{} \t\t\t {} \t\t {:<.4f} \t\t {} \t\t {}'.format(n+1, '??????', y[a, n],  
																	'??????',  '??????' ) )

					print(' ---------------- '*5)
					if Y.shape[0] > a:
						# check min between expected CC. and predited CC. #
						min_sample = min( [Y.shape[0], y.shape[1]] )

						# --- some general descriptors of the trainning --- #
						mean_e 	= np.mean(np.abs(Y[:, a]-y[a, :]))						# mean value of absolute error 											#
						RMSD 	= (np.sum((Y[:, a]-y[a, :])**2)/y.shape[1])**0.5		# root-mean-square deviation (RMSD) or root-mean-square error (RMSE) 	#
						NRMSD 	= RMSD/np.mean(y[a, :])										# Normalized root-mean-square deviation (Normalizing the RMSD) 			#
						print('mean(abs(e)) {:<.4f} \t\t RMSD(RMSE) : {:<.4f} \t\t NRMSD(NRMSE) {:<.4f} '.format( 	mean_e,	RMSD, NRMSD	,))
					else:
						print('mean(abs(e)) ?? \t\t RMSD(RMSE) : ?? \t\t NRMSD(NRMSE) ?? ')

			elif len(y.shape)==2:	print('WARNING :: code 051A DATA.summary() :: Can not summarise self.y (test response stimation)')

		except: print('ERROR :: code 051B DATA.summary() :: Can not summarise self.y (test response stimation)')

		print('{} /END_SUMMARY {} '.format('*'*10, '*'*10) )

	# ***** RESPONCE ADJ ***** #
	def response_adjust(self, function='linear', order=1, coef='set', fit='y', save=True, v=True, plot=False):
		if v: print('*** *** LINEAR response adjust *** *** \n --- (0) Setting variables')

		if not fit in ['y', 'Y']: 	# --- fit --- # fit response to...
			fir = 'y'
			print('ERROR :: code 151a DATA.response_adjust() :: fir must be y or Y :: setting y by default')

		if not type(self.y) is np.ndarray: # --- y --- # response predicted
			if not type(self.y) is list:	print('ERROR :: code 151a DATA.response_adjust() :: self.y must be nparray or list')
			else:							self.y = np.array(self.y)	

		if not type(self.Y) is np.ndarray: # --- Y --- # response ground truth
			if not type(self.Y) is list:	print('ERROR :: code 151a DATA.response_adjust() :: self.Y must be nparray or list')
			else:							self.Y = np.array(self.Y)	

		if not type(self.X) is np.ndarray: # --- Y --- # response ground truth
			if not type(self.X) is list:	print('ERROR :: code 151a DATA.response_adjust() :: self.Y must be nparray or list')
			else:							self.X = np.array(self.X)	
		print(self.X.shape, self.D.shape)
		if self.X.shape == self.D.shape:		X = self.X; 	D = self.D
		else:
			print('WARNNIGN :: code 152b DATA.response_adjust() :: self.X.shape != self.D.shape')
			X = self.X; D = self.X

		if self.Y.shape == self.y.shape:		Y = self.Y; 	y = self.y
		elif self.Y.shape == self.y.T.shape:	Y = self.Y.T;	y = self.y
		else:		print('ERROR :: code 151b DATA.response_adjust() :: self.Y.shape != self.y.T.shape')

		if v: print(' --- (1) Model optimization (model:{}/ complexity:{}/ coef:{}/ fit to:{})'.format(function, order, coef, fit) )
		if fit == 'y':
			for na in range(self.Na):
				z = np.polyfit( Y[na, self.Nc:], Y[na,self.Nc:]-y[na,self.Nc:], order)
				MSE = np.sum(( Y[na, self.Nc:]*z[0]+z[1] - (Y[na,self.Nc:]-y[na,self.Nc:]))**2)
				if save:	y[na,self.Nc:] += Y[na,self.Nc:]*z[0]+z[1]
				if v: print(' Factor {} :: model {} :: order {} :: Y = X*{:<.2} + {:<.2} MSEn:{:<.2}'.format(na+1, function, order, z[0], z[1], MSE) )

		if fit == 'Y':
			for na in range(self.Na):
				z1 = np.polyfit( Y[na, :self.Nc], y[na,:self.Nc], order)
				z2 = np.polyfit( Y[na, self.Nc:], y[na,self.Nc:], order)
				MSE1 = np.sum(( Y[na, self.Nc:]*z1[0]+z1[1] - y[na,self.Nc:])**2)
				MSE2 = np.sum(( Y[na, self.Nc:]*z2[0]+z2[1] - y[na,self.Nc:])**2)
				if save:	y[na,self.Nc:] -=  (Y[na,self.Nc:]*z2[0]+z2[1]) - (Y[na,self.Nc:]*z1[0]+z1[1])
				if v: print(' Factor {} :: model {} :: order {} :: \n'.format(na+1, function, order)+
							'Y = X*{:<.2} + {:<.2} MSEn:{:<.2} :: \n'.format(z1[0], z1[1], MSE1)+
							'y = X*{:<.2} + {:<.2} MSEn:{:<.2} :: \n'.format(z2[0], z2[1], MSE2) )

		if fit == 'X':
			na = 1 
			z1 = np.polyfit( Y[na, :self.Nc], y[na,:self.Nc], order)
			z2 = np.polyfit( Y[na, self.Nc:], y[na,self.Nc:], order)
			MSE1 = np.sum(( Y[na, self.Nc:]*z1[0]+z1[1] - y[na,self.Nc:])**2)
			MSE2 = np.sum(( Y[na, self.Nc:]*z2[0]+z2[1] - y[na,self.Nc:])**2)
			if save:	
				m = np.tensordot( (Y[na,self.Nc:]*z2[0]-Y[na,self.Nc:]*z1[0]) , np.ones(X[self.Nc:, :].shape[1:]), axes=0 )
				X[self.Nc:, :] = (X[self.Nc:, :] - (z2[1] - z1[1])) / m
				#X[self.Nc:, :] = (X[self.Nc:, :] * m + (z2[1] - z1[1])) 
				#X' =  X * (m1-m2) + (h1-h1)
				if v: print(' Factor {} :: model {} :: order {} :: \n'.format(na+1, function, order)+
							'Y = X*{:<.2} + {:<.2} MSEn:{:<.2} :: \n'.format(z1[0], z1[1], MSE1)+
							'y = X*{:<.2} + {:<.2} MSEn:{:<.2} :: \n'.format(z2[0], z2[1], MSE2) )

		if v: print(' --- (2) succesfull linear response adjust ! ---')
		return y

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== PLOT ====================  PLOT ==================== # # ==================== PLOT ====================  PLOT ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	# ***** PLOT ***** #
	def plot_loadings(self, Nc=None, Na=None, Ni=None, aligned=False, mode=[0], fig=None):
		if aligned: loadings = self.La 										# - (1) if aligned use self.La - #
		elif type(self.L) == list and self.L != []: loadings = self.L 		# - (2) Use self.L - #
		elif type(self.loadings) == list and self.Nloadings != []: 
			loadings = self.Nloadings										# - (3) Use self.Nloadings - #
		else: self.aling_init() ; loadings = self.L							# - (4) Inicializate loadings - #

		for i, n in enumerate(mode):
			try:
				if fig == None:	
					plt.plot(loadings[n], '-o')
					return(plt.plot(loadings[n], '-o'), loadings[n])
				else: 
					fig.plot(loadings[n], '-o')
					return(fig.plot(loadings[n], '-o'), loadings[n])

			except:		print('ERROR :: code 021 DATA.plot_vec() :: Can not plot')

	def plot_loadings_estimations(self, S=None, aligned=False, mode=[0], fig=None):

		if type(S) is np.ndarray: 			pass
		elif type(S) is list: 				pass
		elif type(self.S) is np.ndarray:	S = self.S
		elif type(self.S) is list: 			S = self.S
		else:
			self.S = []
			for i, n in enumerate(self.X.shape):
				if i > 0:	self.S.append( np.zeros((n, 1)) )
	
		try:
			if fig == None:	fig = plt.figure()
			else: fig=fig
		except: print('WARNING :: code 021B DATA.plot_vec() :: can NOT define figure')

		# -- genera figure -- #
		try:	fig = fig.add_subplot(111)
		except: print('WARNING :: code 021C DATA.plot_vec() :: can NOT create figure')

		for i, n in enumerate(mode):
			#try:
				fig.plot(S[n], '-o')
			#except:		print('ERROR :: code 021 DATA.plot_vec() :: Can not plot')

		try:
			fig.set_xlabel('Variable.')
			fig.set_ylabel('Signal.')
			fig.set_title('Loading estimation, mode {}.'.format(n) )
		except:		print('ERROR :: code 021 DATA.plot_mat() :: Can not set axis labels')
		

	def plot_45(self, Nc=None, Na=None, Ni=None):
		for i in range(Na+Ni): plt.plot(self.Y[i,:], self.y[:,i], 'o' )
		#plt.show()

	def plot_3d(self, info='mean', sli=None, mode=[2,1], fig=None, plot_type='wireframe', cmap='magma', plot_2d=True):

		# 	(0) info			:	STR 	: how to compress order ways
		# 	(1) mode			:	VEC 	: 2D vector. modes to plot.
		# 	(2) sli 			:	VEC 	: slide choosen to plot.
		# 	(3) fig				:	MAT 	: where to plot
		# 	(4) plot_type		:	INT 	: kind of 3d plot 
		# 	(5) cmap			:	TENSOR 	: coloring method
		#	(6) plot_2d			: 	BOOL 	: 2d proyection in 3d plot

		#cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
		#
		#cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
		#                        'Dark2', 'Set1', 'Set2', 'Set3',
		#                        'tab10', 'tab20', 'tab20b', 'tab20c']
		#
		#cmaps['Diverging'] = [
		#            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
		#            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']	
		#
		#cmaps['Sequential (2)'] = [
		#            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
		#            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
		#            'hot', 'afmhot', 'gist_heat', 'copper']
		#
		#cmaps['Perceptually Uniform Sequential'] = [
		#            'viridis', 'plasma', 'inferno', 'magma', 'cividis']
		#
		#cmaps['Sequential'] = [
		#            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
		#            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
		#            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
		#
		#cmaps['Miscellaneous'] = [
		#            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
		#            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
		#            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
		#
		# eg: ['twilight', 'PiYG', 'PRGn', 'binary', 'gist_yarg', 'viridis', 'plasma', 'inferno', 'magma', 'cividis' 'Greys', 'Purples', 'Blues','prism', 'ocean', 'gist_earth', 'terrain',]
		if type(self.X) is np.ndarray:
			dat = self.X
		else:
			print('ERROR :: code 020 DATA.plot_3d() :: plot_3d need self.X in np.array form to plot')

		try:
			for j,m in enumerate( self.X.shape ):
				if len(dat.shape) <= 2: break
				if not len(self.X.shape)-j-1 in mode and len(dat.shape)>=len(self.X.shape)-j:
					if info == 'mean': # * mean all ways except mode's ones * 
						dat = np.mean(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'sum': # * sum all ways except mode's ones * 
						dat = np.sum(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'std': # * std all ways except mode's ones * 
						dat = np.std(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'slice': # * plot data slice  * 
						if type(sli) is np.ndarray: 
							sli = list(sli)
							print('WARNING :: code 020sli DATA.plot_3d() :: can NOT get list(sli) insted get nparray(sli) :: default change sli=list(sli) ')
						elif sli == None or len(sli) != len(dat.shape)-2:
							 sli = [0]*(len(dat.shape)-2)
							 print('WARNING :: code 020sli DATA.plot_3d() :: can NOT get sli :: default sli=[0]*(len(dat.shape)-2)')
						index = sli
						for n in sorted(mode): index.insert(n, slice(None))
						dat = dat[index]
		except:		print('ERROR :: code 0?? DATA.plot_3d() :: ')

		try:
			if fig == None:	fig = plt.figure()
			else: fig=fig
		except:
			print('ERROR :: code 020b DATA.plot_3d() :: can NOT generate fig obj')

		try: 	ax = fig.add_subplot(111, projection="3d")
		except:	print('ERROR :: code 020c DATA.plot_3d() :: can NOT generate axis, maybe fig argument is NOT what except ')
		
		try:
			x = np.linspace(1, dat.shape[0], dat.shape[0])
			y = np.linspace(1, dat.shape[1], dat.shape[1])
			X, Y = np.meshgrid(x, y)
		except: print('ERROR :: code 0?? DATA.plot_3d() :: can NOT generate mash grid correctly')

		try:
			if plot_2d:
				try: cmap_2D = plt.cm.get_cmap(cmap)
				except:
					print('ERROR :: code 021a DATA.plot_3d() :: can NOT open cmap ')
					pass

				try:
					dat_mode0_max = np.max(dat,axis=0)
					dat_mode0_range = (dat_mode0_max - np.min(dat_mode0_max)) / (np.max(dat_mode0_max) - np.min(dat_mode0_max))
					color_mode0 = cmap_2D(dat_mode0_range)
					#hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),tuple(color_mode0[:,0:-1]))

					for i in range(dat.shape[1]):
						ax.plot(  np.arange(0,dat.shape[0],1)+1, dat[:,i], zs=dat.shape[1], zdir='y', label='curve in (x,y)', color=color_mode0[i]  )
				except:
					print('ERROR :: code 021A DATA.plot_3d() :: plot_2d error plot 2D projection dat_mode0 ')
					pass

				try:
					dat_mode1_max = np.max(dat,axis=1)
					dat_mode1_range = (dat_mode1_max - np.min(dat_mode1_max)) / (np.max(dat_mode1_max) - np.min(dat_mode1_max))
					color_mode1 = cmap_2D(dat_mode1_range)
					#hexcolor = map(lambda rgb:'#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255),tuple(color_mode0[:,0:-1]))

					for i in range(dat.shape[0]):
						ax.plot(  np.arange(0,dat.shape[1],1)+1, dat[i,:], zs=0, zdir='x', label='curve in (x,y)', color=color_mode1[i]  )
				except:
					print('ERROR :: code 021A DATA.plot_3d() :: plot_2d error plot 2D projection dat_mode1 ')
					pass

			if plot_type == 'wireframe': 
				ax.plot_wireframe(X, Y, dat.T, cmap=cmap)

			elif plot_type == 'contour': 
				ax.contour3D(X, Y, dat.T, 50, cmap=cmap)

			elif plot_type == 'surface': 
				ax.plot_surface(X, Y, dat.T, rstride=1, cstride=1, edgecolor='none', cmap=cmap)

			elif plot_type == 'trisurf': 
				ax.plot_trisurf(X, Y, dat.T, edgecolor='none', cmap=cmap)

			else: 
				pass

			ax.set_xlabel('Mode {}'.format( sorted(mode)[0] ))
			ax.set_ylabel('Mode {}'.format( sorted(mode)[1] ))
			ax.set_zlabel('signal')
		except:	print('ERROR :: code 020e DATA.plot_3d() :: can NOT plot 3d')

		return dat

	def plot_mat(self, info='mean', mode=[2,1], sli=None,Nc=None, Na=None, Ni=None, fig=None):

		def axesDimensions(ax):
			if hasattr(ax, 'get_zlim'): 	return 3
			else:	return 2

		if type(self.X) is np.ndarray:
			dat = self.X
		else:
			print('ERROR :: code 020 DATA.plot_mat() :: plot_mat need self.X in np.array form to plot')

		try:	 
			for j,m in enumerate( self.X.shape ):
				if len(dat.shape) <= 2: break
				if not len(self.X.shape)-j-1 in mode and len(dat.shape)>=len(self.X.shape)-j:
					if info == 'mean': # * mean all ways except mode's ones * 
						dat = np.mean(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'sum': # * sum all ways except mode's ones * 
						dat = np.sum(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'std': # * std all ways except mode's ones * 
						dat = np.std(dat, axis=len(self.X.shape)-j-1);
					elif info == 'slice': # * plot data slice  * 
						if type(sli) is np.ndarray: 
							sli = list(sli)
							print('WARNING :: code 020sli DATA.plot_3d() :: can NOT get list(sli) insted get nparray(sli) :: default change sli=list(sli) ')
						elif sli == None or len(sli) != len(dat.shape)-2:
							 sli = [0]*(len(dat.shape)-2)
							 print('WARNING :: code 020sli DATA.plot_3d() :: can NOT get sli :: default sli=[0]*(len(dat.shape)-2)')
						index = sli
						for n in sorted(mode): index.insert(n, slice(None))
						dat = dat[index]	
		except:		print('ERROR :: code 0?? DATA.plot_mat() :: ')

		try:
			if fig == None:	
				if mode[0] > mode[1]: 	plt.matshow(dat.T)
				elif mode[1] > mode[0]: 	plt.matshow(dat)
			else: 
				fig = fig.add_subplot(111)
				if mode[0] > mode[1]: 	fig.matshow(dat.T, cmap='magma')
				elif mode[1] > mode[0]: 	fig.matshow(dat, cmap='magma')
		except:		print('ERROR :: code 021 DATA.plot_mat() :: Can not plot')
		
		try:
			fig.set_xlabel('Mode {}'.format( sorted(mode)[0] ))
			fig.set_ylabel('Mode {}'.format( sorted(mode)[1] ))
		except:		print('ERROR :: code 021 DATA.plot_mat() :: Can not set axis labels')
		
		return dat


	def plot_vec(self, info='mean', mode=[1,0], sli=None, Nc=None, Na=None, Ni=None, fig=None, cmap={'calibration':'winter', 'test':'autumn'}):
		if type(self.X) is np.ndarray:
			dat = self.X
		else:
			print('ERROR :: code 020 DATA.plot_vec() :: plot_vec need self.X in np.array form to plot')

		try:	 
			for j,m in enumerate( self.X.shape ):
				if len(dat.shape) <= 2: break
				if not len(self.X.shape)-j-1 in mode and len(dat.shape)>=len(self.X.shape)-j:
					if info == 'mean': # * mean all ways except mode's ones * 
						dat = np.mean(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'sum': # * sum all ways except mode's ones * 
						dat = np.sum(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'std': # * std all ways except mode's ones * 
						dat = np.std(dat, axis=len(self.X.shape)-j-1);	
					elif info == 'slice': # * plot data slice  *
						if type(sli) is np.ndarray: 
							sli = list(sli)
							print('WARNING :: code 020sli DATA.plot_3d() :: can NOT get list(sli) insted get nparray(sli) :: default change sli=list(sli) ')
						elif sli == None or len(sli) != len(dat.shape)-2:
							 sli = [0]*(len(dat.shape)-2)
							 print('WARNING :: code 020sli DATA.plot_3d() :: can NOT get sli :: default sli=[0]*(len(dat.shape)-2)')
						index = sli
						for n in sorted(mode): index.insert(n, slice(None))
						dat = dat[index]
		except:		print('ERROR :: code 0?? DATA.plot_vec() :: ')

		try:
			if not((type(self.Nt) is np.ndarray and not len(self.Nt.shape) == 1 and self.Nt.shape[0] == 1) or (not type(self.Nt) is np.ndarray and not self.Nt == None and self.isnum(self.Nt))):
				self.Nt = 0
			elif self.isnum(self.Nt): 
				self.Nt = int(self.Nt)
			
			if not((type(self.Nv) is np.ndarray and not len(self.Nv.shape) == 1 and self.Nv.shape[0] == 1) or (not type(self.Nv) is np.ndarray and not self.Nv == None and self.isnum(self.Nv))):
				self.Nv = 0
			elif self.isnum(self.Nv): 
				self.Nv = int(self.Nv)
			
			if not((type(self.N) is np.ndarray and not len(self.N.shape) == 1 and self.N.shape[0] == 1) or (not type(self.N) is np.ndarray and not self.N == None and self.isnum(self.N))):
				try: 	self.N = self.D.shape[0]
				except:	
					self.N = 0
					print('WARNING :: code 021 DATA.plot_vec() :: can NOT extract N (number of sample) direct from data (self.D).')
			elif self.isnum(self.N): 
				self.N = int(self.N)
			
			if not((type(self.Na) is np.ndarray and not len(self.Na.shape) == 1 and self.Na.shape[0] == 1) or (not type(self.Na) is np.ndarray and not self.Na == None and self.isnum(self.Na))):
				self.Nc = int(self.N-self.Nt-self.Nv)
			elif self.isnum(self.Nc): 
				self.Nc = int(self.Nc)
			
		except: 
			print('WARNING :: code 021 DATA.plot_vec() :: can NOT define self.N(TOTAL number of sample) self.Nc(number of calibration sample) self.Nv(number of validation sample) self.Nt(number of test sample)')
		
		try:
			if fig == None:	fig = plt.figure()
			else: fig=fig
		except: print('WARNING :: code 021B DATA.plot_vec() :: can NOT define figure')

		try:
			# -- genera figure -- #
			fig = fig.add_subplot(111)

			# -- create color map space -- # 
			if not type(cmap) == dict: 
				if type(cmap) == str: cmap = {'all':cmap}
				else: cmap = {}

			try: 
				if 'calibration' in cmap:	cmap_calibration = plt.cm.get_cmap(cmap['calibration'])
				else:						cmap_calibration = plt.cm.get_cmap('winter')
			except: print('WARNING :: code 021a DATA.plot_vec() :: can NOT open cmap[calibration] ')

			try: 
				if 'test' in cmap:			cmap_test = plt.cm.get_cmap(cmap['test'])
				else:						cmap_test = plt.cm.get_cmap('autumn')
			except: print('WARNING :: code 021a DATA.plot_vec() :: can NOT open cmap[test] ')

			try: 
				if 'all' in cmap:			cmap_all = plt.cm.get_cmap(cmap['all'])
				else:						cmap_all = plt.cm.get_cmap('cool')
			except: print('WARNING :: code 021a DATA.plot_vec() :: can NOT open cmap[test] ')
					
			# -- create color map space -- # 
			if mode[0] == 0:  dat_post = dat.T
			else: dat_post = dat
			
			if mode[0] != 0 and mode[1] != 0:
				# -- specify colors from the color space -- #
				dat_mode1_max = np.max(dat_post,axis=1)
				dat_mode1_range = (dat_mode1_max - np.min(dat_mode1_max)) / (np.max(dat_mode1_max) - np.min(dat_mode1_max))
				color_mode1 = cmap_all(dat_mode1_range)

				for n in range(dat_post.shape[0]):
					fig.plot(dat_post[n,:], color=color_mode1[n] )
			else:
				if self.Nc>0:	# -- CALIBRATION SAMPLE -- #
					# -- specify colors from the color space -- #
					dat_mode1_max = np.max(dat_post[:self.Nc,:],axis=1)
					dat_mode1_range = (dat_mode1_max - np.min(dat_mode1_max)) / (np.max(dat_mode1_max) - np.min(dat_mode1_max))
					color_mode1 = cmap_calibration(dat_mode1_range)

					for n in range(dat_post[:self.Nc,:].shape[0]):
						fig.plot(dat_post[n,:], color=color_mode1[n] )

				#if self.Nv>0:	fig.plot(dat[self.Nc:(self.Nc+self.Nv),:].T, color='y')
				if self.Nt>0:	# -- TEST SAMPLE -- #
					# -- specify colors from the color space -- #
					dat_mode1_max = np.max(dat_post[self.Nc:,:],axis=1)
					dat_mode1_range = (dat_mode1_max - np.min(dat_mode1_max)) / (np.max(dat_mode1_max) - np.min(dat_mode1_max))
					color_mode1 = cmap_test(dat_mode1_range)

					for n in range(dat_post[:self.Nc,:].shape[0], dat_post[self.Nc:,:].shape[0]):
						fig.plot(dat_post[n,:], color=color_mode1[n] )

		except:		print('ERROR :: code 021b DATA.plot_vec() :: Can not plot')
		
		try:
			fig.set_xlabel('Mode {}'.format( sorted(mode)[0] ))
			fig.set_ylabel('Mode {}'.format( sorted(mode)[1] ))
		except:		print('ERROR :: code 021 DATA.plot_mat() :: Can not set axis labels')
		
		return dat

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== LOAD  ====================  LOAD ==================== # # ====================LOAD ==================== LOAD ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	def load_methadata(self, file):
		file = open(file, 'r')

		for line in file:
			if 'UnCalibrated factors' in line:
				self.Ni = int( line.split(':')[-1] )
			elif 'Calibrated factors' in line:
				self.Na = int( line.split(':')[-1] )

			if 'Total number of samples' in line:
				self.N  = int( line.split(':')[-1] )
				if self.N == self.X.shape[0]:	print(' Total number of sample in metadata is consistant with data (self.X) shape.')
				else:							print('WARNING ::  Total number of sample in metadata is consistant with data (self.X) shape.')

			if 'Calibration samples' in line:
				self.Nc  = int( line.split(':')[-1] )
			if 'TEST samples' in line:
				self.Nt  = int( line.split(':')[-1] )
			if 'Well aligned channels' in line:
				self.a  = int( line.split(':')[-1] )
			if 'Analisys model' in line:
				print('This data was analised with {}'.format( line.split(':')[-1] ) )
			if 'Alignation model' in line:
				print('This data was sligned with {}'.format( line.split(':')[-1] ) )

		if self.isnum(self.Na) and self.isnum(self.Ni): self.f = self.Na + self.Ni
			
	def load_folther(self, path, data_shape='2D', save=True, v=True, summary=False, force=True):
		Xc, Xt, Yc, Yt, Md, spectra= [None]*6

		if data_shape == '2D':
			try: 
				Xc = self.load_Xfrom_callfile( '{}/{}'.format(path, 'call_file_calibration.txt') )
				dat = self.load_matlab( path+'/matlabfiles'+'/all.mat')
			except: 
				if v:	print('WARNING :: code load DATA.load_folther() --> DATA.load_Xfrom_callfile() :: Can not load {}'.format( '{}/{}'.format(path, 'call_file_calibration.txt') ) )
			try: Xt = self.load_Xfrom_callfile( '{}/{}'.format(path, 'call_file_test.txt') )
			except: 
				if v:  	print('WARNING :: code load DATA.load_folther() --> DATA.load_Xfrom_callfile() :: Can not load {}'.format( '{}/{}'.format(path, 'call_file_test.txt') ) )

		elif data_shape == '3D':
			try: 
				Xc = self.load_Xfrom_callerfile(  path+'/caller_file_calibration.txt', save=False )
				#dat = self.load_matlab( path+'/matlabfiles'+'/all.mat')
			except: 
				if v:	print('WARNING :: code load DATA.load_folther() --> DATA.load_Xfrom_callfile() :: Can not load {}'.format( '{}/{}'.format(path, 'call_file_calibration.txt') ) )
			try: 
				Xt = self.load_Xfrom_callerfile(  path+'/caller_file_test.txt', save=False )
			except: 
				if v:  	print('WARNING :: code load DATA.load_folther() --> DATA.load_Xfrom_callfile() :: Can not load {}'.format( '{}/{}'.format(path, 'call_file_test.txt') ) )

		try: Yc = self.load_Yfrom_callfile( '{}/{}'.format(path, 'Ycall_calibration.txt') )
		except:  
			if v: 	print('WARNING :: code load DATA.load_folther() --> DATA.load_Yfrom_callfile() :: Can not load {}'.format( '{}/{}'.format(path, 'Ycall_calibration.txt') ) )
		try: Yt = self.load_Yfrom_callfile( '{}/{}'.format(path, 'Ycall_test.txt') )
		except:  
			if v: print('WARNING :: code load DATA.load_folther() --> DATA.load_Yfrom_callfile() :: Can not load {}'.format( '{}/{}'.format(path, 'Ycall_test.txt') ) )

		try: Md = self.load_methadata( file='{}/{}'.format(path, 'README(data_help).txt') )
		except:  
			if v: print('WARNING :: code load DATA.load_folther() --> DATA.load_methadata() :: Can not load {}'.format( '{}/{}'.format(path, 'README(data_help).txt') ) )
		try: Md = self.load_methadata( file='{}/{}'.format(path, 'metadata.txt'))
		except:  
			if v: print('WARNING :: code load DATA.load_folther() --> DATA.load_methadata() :: Can not load {}'.format( '{}/{}'.format(path, 'metadata.txt') ) )

		try: spectra = self.load_spectra( file='{}/{}'.format(path, 'spectra.txt'))
		except:  
			if v: print('WARNING :: code load DATA.load_folther() --> DATA.load_spectra() :: Can not load {}'.format( '{}/{}'.format(path, 'spectra.txt') ) )


		if summary: self.summary()

		if save:
			print( Yc.shape, Yt.shape )
			if force:
				self.X = np.concatenate( (Xc, Xt) )
				self.D = np.concatenate( (Xc, Xt) )
			else:
				self.X = self.X if type(self.X) != type(None) else np.concatenate( (Xc, Xt) )
				self.D = self.D if type(self.D) != type(None) else np.concatenate( (Xc, Xt) )

		return Xc, Xt, Yc, Yt, Md, spectra

	def load_spectra(self, filename=None, save=True ):
		if type(filename) == type(None): 
			print('ERROR :: code 005C DATA.load_spectra() :: Need filename' )
			return None

		try: 
			spectra = self.load_txt(filename).T
			if save: 
				if type(self.S) == list: 	
					self.S.append(spectra)
				else:
					self.S = [np.array(spectra)]	
			return spectra

		except:
			print('ERROR :: code 005C DATA.load_spectra() :: can NOT load {}'.format(filename) )
	
	# ***** ***** LOAD from TYPE ***** ***** #
	def load_pickle(self, filename='default_obj.pkl'):
		try:
			if filename == None: print('WARNING :: code 005C DATA.load_pickle() :: Need filename' )
			with open(filename, 'rb') as output:  # Overwrites any existing file.
				dat = pickle.load(output)
				output.close()
			return dat
		except:
			print('ERROR :: code 005B DATA.load_pickle() :: Can not load {} as pickle'.format(filename) )

	def load_matlab(self, filename):
		try:
			dat = scipy.io.loadmat(filename)
			print(dat.keys() )
			print('Load: ' + str(filename))
			return dat
		except:
			print('Can not load: '+ str(filename))

	def load_numpy(self, file=None, store=False):
		try:np_file = file
		except: print('ERROR :: code 005 DATA.save_numpy() :: Can not get file name')

		try:
			dat = np.load( file )
			if store == 'X': # if store == true save all posible data #
				self.X = dat
				self.D = dat
				if self.N == None: self.N = self.X.shape[0]
				if self.ways == None: self.ways = len(self.X.shape)
				elif self.ways == len(self.X.shape):  
					pass
				elif self.ways != len(self.X.shape):  
					# activate if (has store NOT compatible info) 
					# ways is diferent than X.shape (previous already store information)
					self.ways = len(self.X.shape)
					print('ERROR :: code 006 DATA.load_numpy() :: not compatible data' )
				
				self.ways = len(self.X.shape)
			print('Load : {} succesfull !.'.format(str(np_file)))
			return dat
		except:
			try:
				if file.split('.')[-1] != 'np': 
					print('ERROR :: code 005A DATA.load_numpy() :: File {} must be numpy(.np) file.'.format(str(np_file)) )
				print('ERROR :: code 005 DATA.load_numpy() :: Can not load: {}'.format(str(np_file)) )
			except:
				print('ERROR :: code 005B DATA.load_numpy() :: File name can not be recognised')

	def load_txt(self, file=None,  delimiter=None):
		if type(file) == type(None):  print(' ERROR :: code 013 self.load_txt() :: self.load_txt() need file name')
		try:
			f_data = np.loadtxt( fname=file.strip(), )# delimiter=delimiter)
			print('Load: ' + str(file))
			return f_data
		except:

			print('ERROR :: code 005 self.load_txt() :: Can not load: {}'.format(str(file)) )

	# ***** ***** X LOAD from ***** ***** #
	def load_Xfrom_npfile(self, call_file=None, tipe='np', delimiter=None, sample='cal', append_data=True):
		# Load data from np file # 
		# this function decides if is it posible to join with all data or NOT #
		if tipe == None: tipe = 'np'
		try:
			if call_file == None: print('ERROR :: code 007 DATA.load_Xfrom_npfile() :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 DATA.load_Xfrom_npfile() :: [missing info] need call file to load data.')
		
		path = '/'.join(call_file.split('/')[:-1])+'/' 
		try:
			f_sample = self.load_numpy(file=call_file.strip(), store=False)
			print(' **** File {} read without problem || data dimentions {} **** '.format(call_file, f_sample.shape) ) 
		except: print('ERROR  :: code 011 DATA.load_Xfrom_npfile() :: error while reading {} !!! '.format(call_file) ) 

		try: 
			if np.array(f_sample).all() == np.array(None):			dat = None
			elif type(f_sample) is np.ndarray: 						dat = np.array(f_sample)
		except: 
			print('ERROR :: code 016 DATA.load_Xfrom_npfile() :: VCan not load {}.'.format(call_file) )
			dat = None

		try:
			if not type(self.X) is np.ndarray: self.X = dat
			elif append_data: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))][j])==(len(self.X.shape)-1) for j,o in enumerate(self.X.shape)]):
					if n: self.X = np.concatenate( (self.X, dat),axis=i); break
			if not type(self.D) is np.ndarray: self.D = dat
			elif append_data: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))][j])==(len(self.D.shape)-1) for j,o in enumerate(self.D.shape)]):
					if n: self.D = np.concatenate( (self.D, dat),axis=i); break

			if type(self.X) is np.ndarray and type(self.D) is np.ndarray and self.X.shape != self.D.shape:
				self.X = self.D
				print('ERROR :: code 030 DATA.load_Xfrom_npfile() :: self.X and self.D has NOT compatible shape. \r\n For here self.X = self.D would solved the problem.')
		except Exception as e: 
			print('ERROR/ :: {} :: /ERROR'.format(e) )
			print('ERROR  :: code ?? DATA.load_Xfrom_npfile() ::  in >> self.load_from_callfile()')

	def load_Xfrom_txtfile(self, call_file=None, tipe='txt', delimiter=None, sample='cal', append_data=True):
		if tipe == None: tipe = 'txt'
		try:
			if call_file == None: print('ERROR :: code 007 DATA.load_Xfrom_npfile() :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 DATA.load_Xfrom_npfile() :: [missing info] need call file to load data.')
		
		path = '/'.join(call_file.split('/')[:-1])+'/' 
		try:
			f_sample = self.load_txt( file=call_file,  delimiter=delimiter)
			print(' **** File {} read without problem || data dimentions {} **** '.format(call_file, f_sample.shape) ) 
		except: print('ERROR  :: code 011 DATA.load_Xfrom_txtfile() :: error while reading {} !!! '.format(call_file) ) 

		dat = np.array(f_sample)

		try:
			if not type(self.X) is np.ndarray: self.X = dat
			elif append_data: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))][j])==(len(self.X.shape)-1) for j,o in enumerate(self.X.shape)]):
					if n: self.X = np.concatenate( (self.X, dat),axis=i); break
			if not type(self.D) is np.ndarray: self.D = dat
			elif append_data: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))][j])==(len(self.D.shape)-1) for j,o in enumerate(self.D.shape)]):
					if n: self.D = np.concatenate( (self.D, dat),axis=i); break

			if self.X.shape != self.D.shape:
				self.X = self.D
				print('ERROR :: code 030 DATA.load_Xfrom_txtfile() :: self.X and self.D has NOT compatible shape. \r\n For here self.X = self.D would solved the problem.')
		except Exception as e: 
			print('ERROR/ :: {} :: /ERROR'.format(e) )
			print('ERROR  :: code ?? DATA.load_Xfrom_txtfile() ::  in >> self.load_from_callfile()')

	def load_Xfrom_callerfile(self, call_file=None, tipe='txt', delimiter=None, sample='cal', append_data=True, save=True):
		if tipe == None: tipe = 'txt'
		try:
			if call_file == None: print('ERROR :: code 007 :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 :: [missing info] need call file to load data.')
		file_call = open(call_file, 'r')
		path = '/'.join(call_file.split('/')[:-1])+'/' 
		try:
			files_list = [n[:-1] for n in file_call if n != '' and n != ' ' and len(n) > 2 ] # list with all files 
			dat = [] # temporal tensor data
			for i, n in enumerate(files_list):
				try:
					print(' * {} * Reading file {} ...'.format(i, n)) 
					if tipe == 'txt':
						f_sample = self.load_Xfrom_callfile(call_file=path+str(n), tipe='txt', delimiter=None, 
															sample='cal', append_data=False, save=False)

						dat.append( f_sample )

					print(' **** File {} read without problem || data dimentions {} **** '.format(n, f_sample.shape) ) 
				except: print('ERROR :: code 0014 ::  while reading {} !!! '.format(n) ) 

		except: print('ERROR :: code 0014 self.load_Xfrom_callfile() ::  while reading {} !!! '.format(file_call) ) 

		try:	dat = np.array(dat)
		except: pass

		try:
			if not type(self.X) is np.ndarray: self.X = dat
			elif append_data and save: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))][j])==(len(self.X.shape)-1) for j,o in enumerate(self.X.shape)]):
					if n: self.X = np.concatenate( (self.X, dat),axis=i); break
			if not type(self.D) is np.ndarray: self.D = dat
			elif append_data and save: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))][j])==(len(self.D.shape)-1) for j,o in enumerate(self.D.shape)]):
					if n: self.D = np.concatenate( (self.D, dat),axis=i); break

			if self.X.shape != self.D.shape:
				self.X = self.D
				print('ERROR :: code 030 DATA.load_Xfrom_callfile() :: self.X and self.D has NOT compatible shape. \r\n For here self.X = self.D would solved the problem.')
		except Exception as e: 
			print('ERROR/ :: {} :: /ERROR'.format(e) )
			print('ERROR  :: code ?? DATA.load_Xfrom_callfile() ::  in >> self.load_from_callfile()')
		
		return dat


	def load_Xfrom_callfile(self, call_file=None, tipe='txt', delimiter=None, sample='cal', append_data=True, save=True):
		if tipe == None: tipe = 'txt'
		try:
			if call_file == None: print('ERROR :: code 007 :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 :: [missing info] need call file to load data.')
		file_call = open(call_file, 'r')
		path = '/'.join(call_file.split('/')[:-1])+'/' 
		try:
			files_list = [n[:-1] for n in file_call if n != '' and n != ' ' and len(n) > 2 ] # list with all files 
			dat = [] # temporal tensor data
			for i, n in enumerate(files_list):
				try:
					print(' * {} * Reading file {} ...'.format(i, n)) 
					if tipe == 'txt':
						f_sample = self.load_txt( file=path+str(n),  delimiter=delimiter)
						dat.append( f_sample )

					print(' **** File {} read without problem || data dimentions {} **** '.format(n, f_sample.shape) ) 
				except: print('ERROR :: code 0014 ::  while reading {} !!! '.format(n) ) 

		except: print('ERROR :: code 0014 self.load_Xfrom_callfile() ::  while reading {} !!! '.format(file_call) ) 

		try:	dat = np.array(dat)
		except: pass

		try:
			if not type(self.X) is np.ndarray: self.X = dat
			elif append_data and save: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.X.shape, dat.shape))][j])==(len(self.X.shape)-1) for j,o in enumerate(self.X.shape)]):
					if n: self.X = np.concatenate( (self.X, dat),axis=i); break
			if not type(self.D) is np.ndarray: self.D = dat
			elif append_data and save: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.D.shape, dat.shape))][j])==(len(self.D.shape)-1) for j,o in enumerate(self.D.shape)]):
					if n: self.D = np.concatenate( (self.D, dat),axis=i); break

			if self.X.shape != self.D.shape:
				self.X = self.D
				print('ERROR :: code 030 DATA.load_Xfrom_callfile() :: self.X and self.D has NOT compatible shape. \r\n For here self.X = self.D would solved the problem.')
		except Exception as e: 
			print('ERROR/ :: {} :: /ERROR'.format(e) )
			print('ERROR  :: code ?? DATA.load_Xfrom_callfile() ::  in >> self.load_from_callfile()')
		
		return dat

	# ***** ***** Y LOAD from ***** ***** #
	def load_Yfrom_file(self, file=None, tipe='txt', delimiter=None, sample='cal', append_data=True):
		if tipe == None: tipe = 'txt'
		try:
			if file == None: print('ERROR :: code 007 :: DATA.load_Yfrom_file.py :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 :: DATA.load_Yfrom_file.py :: [missing info] need call file to load data.')

		Ydata = self.load_txt( file=file.strip(),  delimiter=delimiter)
		Ydata = np.array(Ydata)

		try:
			if len( Ydata.shape ) == 1: # if new axis if its necessary
				Ydata = Ydata[:,np.newaxis]
		except: print('  ERROR  :: code ?? :: DATA.load_Yfrom_file.py :: can NOT add new axis to Y' )

		try: 
			if type(self.Y) is np.ndarray:
				for i,(n,m) in enumerate(zip(self.Y.shape, Ydata.shape)):
					if n != m: print( 'ERROR dimention {} :: DATA.load_Yfrom_file.py :: new data(dim={}) is NOT compatible with old store data(dim={})'.format(i,m,n) )

			elif not type(self.Y) is np.ndarray: self.Y = Ydata
			elif append_data: self.Y = np.concatenate( (self.Y, Ydata),axis=0) 
		except: print(' !!!! ERROR  :: code ?? :: DATA.load_Yfrom_file.py :: can NOT load Y data from {} '.format(file) )

		return self.Y

	def load_Yfrom_npfile(self, file=None, tipe='txt', delimiter=None, sample='cal', append_data=True):
		if tipe == None: tipe = 'np'
		try:
			if file == None: print('ERROR :: code 007 :: DATA.load_Yfrom_npfile.py :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 :: DATA.load_Yfrom_npfile.py :: [missing info] need call file to load data.')

		Ydata = self.load_numpy( file=file, store=False)
		try: Ydata = np.array(Ydata)
		except: print('ERROR :: code 007B :: DATA.load_Yfrom_npfile.py :: can NOT make array with data in file {}.'.format(file) )

		try: 
			if type(self.Y) is np.ndarray:
				for i,(n,m) in enumerate(zip(self.Y.shape, Ydata.shape)):
					if n != m: print( 'ERROR dimention {} :: DATA.load_Yfrom_file.py :: new data(dim={}) is NOT compatible with old store data(dim={})'.format(i,m,n) )

			elif not type(self.Y) is np.ndarray: self.Y = Ydata
			elif append_data: self.Y = np.concatenate( (self.Y, Ydata),axis=0) 
		except: print(' !!!! ERROR  :: code ?? :: DATA.load_Yfrom_file.py :: can NOT load Y data from {} '.format(file) )

		if len(self.Y.shape) == 1:	self.Y = np.array( [self.Y] ).T
		return self.Y

	def load_Yfrom_callfile(self, call_file=None, tipe='txt', delimiter=None, sample='cal', append_data=True):
		if tipe == None: tipe = 'txt'
		try:
			if call_file == None: print('ERROR :: code 007 DATA.load_Yfrom_callfile() :: [missing info] need call file to load data.')
		except: print('ERROR :: code 007 DATA.load_Yfrom_callfile() :: [missing info] need call file to load data.')
		file_call = open(call_file, 'r')
		path = '/'.join(call_file.split('/')[:-1]) 
		try:
			files_list = [n[:-1] for n in file_call if n != '' and n != ' ' and len(n) > 2 ] # list with all files 
			dat = [] # temporal tensor data
			print('Y call files list :', files_list)
			for i, file in enumerate(files_list):
				try:
					print(' * {} * Reading file {} ...'.format(i, '{}/{}'.format(path, file))) 
					if tipe == 'txt':
							f_sample = self.load_txt( file='{}/{}'.format(path, file),  delimiter=delimiter)
							dat.append( f_sample )
					print(' **** File {} read without problem || data dimentions {} **** '.format(file, f_sample.shape) ) 
				except: print('ERROR :: code 0014 DATA.load_Yfrom_callfile() ::  while reading {}'.format( '{}/{}'.format(path, file) ) ) 

		except: print('ERROR :: code 0014 DATA.load_Yfrom_callfile() ::  while reading {}'.format(file_call) ) 

		try:dat = np.array(dat).T
		except: pass
		
		try:
			if not type(self.Y) is np.ndarray: self.Y = dat
			elif append_data: 
				for i, n in enumerate([(sum([ n==m for i,(n,m) in enumerate(zip(self.Y.shape, dat.shape))])-[n==m for i,(n,m) in enumerate(zip(self.Y.shape, dat.shape))][j])==(len(self.Y.shape)-1) for j,o in enumerate(self.Y.shape)]):
					if n: self.Y = np.concatenate( (self.Y, dat),axis=i); break
			
		except Exception as e: 
			print('ERROR/ :: {} :: /ERROR'.format(e) )
			print('ERROR  :: code ?? DATA.load_Xfrom_callfile() ::  in >> self.load_from_callfile()')
		
		return self.Y

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== SAVE ====================  SAVE ==================== # # ==================== SAVE ====================  SAVE ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	# ***** SAVE from TYPE ***** #
	def save_pickle(self, filename='default_obj.pkl', data=None):
		try:
			if filename == None: print('WARNING :: code 005C DATA.save_pickle() :: Need filename' )
			if data == None:	
				print('WARNING :: code 005C DATA.save_pickle() :: Need data as argument to store. default = self ' )
				data = self
			
			with open(filename, 'wb') as output:  # Overwrites any existing file.
				pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
				output.close()

		except:
			print('ERROR :: code 005B DATA.save_pickle() :: Can not SAVE {} as pickle'.format(filename) )

	def save_numpy(self, file=None, data=None, allow_pickle=True, fix_imports=True):

		# file : file, str, or pathlib.Path
		#			File or filename to which the data is saved. 
		#			If file is a file-object, then the filename is unchanged. 
		#			If file is a string or Path, a .npy extension will be 
		#			appended to the file name if it does not already have one.
		# arr : array_like
		# 			Array data to be saved.
		# allow_pickle : bool, optional
		#			Allow saving object arrays using Python pickles. 
		# fix_imports : bool, optional
		#			Only useful in forcing objects in object arrays

		try:np_file = file
		except: print('ERROR :: code 005 DATA.save_numpy() :: Can not get file name')

		try: 
			if type(data) is np.ndarray:	data = data
			else: 
				print('WARNING :: code 005B :: DATA.save_numpy() :: data NOT spicified. default = self.X')
				data = self.X
		except: print('ERROR :: code 005 :: DATA.save_numpy() :: Can not get data')

		try:
			dat = np.save( file, arr, allow_pickle=True, fix_imports=True )

		except:
			try:
				if file.split('.')[-1] != 'np': 
					print('ERROR :: code 005A DATA.save_numpy() :: File {} must be numpy(.np) file.'.format(str(np_file)) )
				print('ERROR :: code 005 DATA.save_numpy() :: Can not save: {}'.format(str(np_file)) )
			except:
				print('ERROR :: code 005B DATA.save_numpy() :: File name can not be recognised')

	def save_text(self, file=None, data=None, delimiter=None):
		try:np_file = file
		except: print('ERROR :: code 005 DATA.save_text() :: Can not get file name')

		try: 
			if type(data) is np.ndarray:
				data = data
			else: data = self.X
		except: print('ERROR :: code 005 DATA.save_text() :: Can not get data')

		try:
			if len(data.shape) == 2: # 3er order data
				np.savetxt( fname=file, X=data, delimiter=delimiter )
			if len(data.shape) == 3: # 3er order data
				self.export_3D_data(file=file, data=data, delimiter=delimiter, v=True)
			pass
		except:
			try:
				if file.split('.')[-1] != 'txt': 
					print('ERROR :: code 005A DATA.save_text() :: File {} must be numpy(.np) file.'.format(str(np_file)) )
				print('ERROR :: code 005 DATA.save_text() :: Can not save: {}'.format(str(np_file)) )
			except:
				print('ERROR :: code 005B DATA.save_text() :: File name can not be recognised')
		
		return None

	def export(self, path):
		if path == None: 
			print('WARNNING :: code 005A :: DATA.export_4D_data() :: undefined path ' )
			path = './data_export/'

		if len(self.X.shape) == 3: self.export_3D_data(path=path)
		if len(self.X.shape) == 4: self.export_4D_data(path=path)
		print('Data export to {} ... check... OK! '.format(path) )
		return True

	def export_3D_data(self, path=None, data=None, delimiter=None, v=True, ):
		# **** Check data integrity **** #
		if path == None: 
			print('WARNNING :: code 005A :: DATA.export_4D_data() :: undefined path ' )
			path = './data_export/'

		if delimiter == None: delimiter = '\t'

		if not type(data) is np.ndarray and type(self.X) is np.ndarray:
			data = self.X
		if type(data) is np.ndarray and len(data.shape) == 4:
			print('Data shape... check... OK! ' )

		if v: print('Exporting data set...' )

		# **** Create path **** #
		try:		os.makedirs( '/'.join(path.split('/')) )
		except:	    print("WARNING :: DATA.export_3D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1]))
		
		# **** DATA helper **** #
		file_help = open('{}/README(data_help).txt'.format(path), 'w')
		file_help.write('Brief automatic generated description of the dataset \n' )

		file_help.write('{:50} : {} \n'.format('Full data(X) set shape', 				self.X.shape))
		file_help.write('{:50} : {} \n'.format('Response variable(Y)  shape', 			self.Y.shape))
		file_help.write('{:50} : {} \n'.format('Calibrated factors (analytes)', 		self.Na))
		file_help.write('{:50} : {} \n'.format('UnCalibrated factors (interferents)', 	self.Ni))
		file_help.write('{:50} : {} \n'.format('Total number of samples', 				self.N))
		file_help.write('{:50} : {} \n'.format('Calibration samples',					self.Nc))
		file_help.write('{:50} : {} \n'.format('TEST samples', 							self.Nt))
		file_help.write('{:50} : {} \n'.format('Well aligned channels',					self.a))
		file_help.write('{:50} : {} \n'.format('Analisys model', 						self.model))
		file_help.write('{:50} : {} \n'.format('Alignation model',						self.aling))

		try:	file_help.write('{:50} : {} [{}] \n'.format('Spectra',					len(self.S), ';'.join(['({},{})'.format(n.shape[0], n.shape[1]) for n in self.S]) ) )
		except: pass

		file_help.write('{} {}'.format('-'*100,'\n'))
		file_help.write('File name {} Content \n'.format(' '*30) )
		file_help.write('{} {}'.format('-'*100,'\n'))

		# **** (S) spectra  **** # **** **** **** **** **** **** 
		try:
			if type(self.S) == list:
				file_spectra = open('{}/spectra.txt'.format(path), 'w')
				for sn, s in eknumerate(self.S):
					file_spectra.write('spectra_channel_{}.txt'.format(sn))
					np.savetxt( fname='{}/spectra_channel_{}.txt'.format(path, sn), X=self.S[sn], delimiter=delimiter, fmt='%10.5f' )
				file_spectra.close()
			print('Save spectra... check... OK! ' )
		except: pass

		if type(self.Nc) == int and type(self.Nt) == int:
			# **** Save CAL and TEST samples **** # ------------------------------------------------------------------------------
			# **** (X) CALIBRATION  **** # **** **** **** **** **** **** 
			for nc in range(self.Nc):	
				np.savetxt( fname='{}/calibration_sample{}.txt'.format(path, nc+1), X=data[nc,:,:], delimiter=delimiter, fmt='%10.5f' )
				file_help.write('{:50} {} \n'.format('calibration_sample{}.txt'.format(nc+1), 'Calibration data file number {}/{}, shape {}, min:{}  max:{}'.format(nc+1, self.Nc, data[nc,:,:].shape, np.min(data[nc,:,:]), np.max(data[nc,:,:])) ))
			print('Save individual CALIBRATION samples... check... OK! ' )
			
			# **** (X) Make call file **** #
			file_call = open('{}/call_file_calibration.txt'.format(path), 'w')
			for nc in range(self.Nc):		
				file_call.write('calibration_sample{}.txt\n'.format(nc+1))
			file_call.close()
			file_help.write('\n{:50} {} \n\n'.format('call_file_calibration.txt', 'Names of calibration files, total calibration files:{}'.format(self.Nc)) )
			print('Save CALIBRATION call file... check... OK! ' )

			# **** (x) TEST  **** # **** **** ****  **** **** **** 
			for nt in range(self.Nt):	
				np.savetxt( fname='{}/test_sample{}.txt'.format(path, nt+1), X=data[self.Nc+nt,:,:], delimiter=delimiter, fmt='%10.5f' )
				file_help.write('{:50} {} \n'.format('test_sample{}.txt'.format(nt+1), 'TEST data file number {}/{}, shape {}, min:{}  max:{}'.format(nt+1, self.Nt, data[nc,:,:].shape, np.min(data[self.Nc+nt,:,:]), np.max(data[self.Nc+nt,:,:])) ))
			print('Save individual TEST samples... check... OK! ' )

			# **** (x) Make call file **** # 
			file_call = open('{}/call_file_test.txt'.format(path), 'w')
			for nt in range(self.Nt):		
				file_call.write('test_sample{}.txt\n'.format(nt+1))
			file_call.close()
			file_help.write('\n{:50} {} \n\n'.format('call_file_test.txt', 'Names of test files, total test files:{}'.format(self.Nt)) )
			print('Save TEST call file... check... OK! ' )

			for f in range(self.Y.shape[1]):
				# **** Y CALIBRATION factor f **** #
				file_calibration = open('{}/Y_calibration_factor{}.txt'.format(path, f+1), 'w')
				for nc in range(self.Nc):		
					file_calibration.write('{} \n'.format(self.Y[nc, f] ))
				file_calibration.close()
				file_help.write('{:50} {} \n'.format('Y_calibration_factor{}.txt'.format(f+1), 'Calibration concentrations for factor(analyte):{}/{} min:{}  max:{}'.format(f+1, self.Y.shape[1], np.min(self.Y[:self.Nc, f]), np.max(self.Y[:self.Nc, f]) ) )) # , np.min(self.Y[:self.Nc, f], np.max(self.Y[:self.Nc, f]) )) ))
				print('Save response(Y) CALIBRATION factor {}... check... OK! '.format(f+1) )

				# **** Y TEST factor f **** #
				file_test = open('{}/Y_test_factor{}.txt'.format(path, f+1), 'w')
				for nt in range(self.Nt):		
					file_test.write('{}\n'.format(self.Y[self.Nc+nt, f] ))
				file_test.close()
				file_help.write('{:50} {} \n'.format('Y_test_factor{}.txt'.format(f+1), 'TEST concentrations for factor(analyte):{}/{} min:{}  max:{}'.format(f+1, self.Y.shape[1], np.min(self.Y[self.Nc:, f]), np.max(self.Y[self.Nc:, f]) ) )) # , np.min(self.Y[:self.Nc, f], np.max(self.Y[:self.Nc, f]) )) ))
				print('Save response(Y) TEST factor {}... check... OK! '.format(f+1) )
				
			file_call_calibration = open('{}/Ycall_calibration.txt'.format(path, f+1), 'w')
			file_call_test        = open('{}/Ycall_test.txt'.format(path, f+1), 'w')
			for f in range(self.Y.shape[1]):
				file_call_calibration.write('Y_calibration_factor{}.txt\n'.format(f+1) )
				file_call_test.write('Y_test_factor{}.txt\n'.format(f+1) )

			np.savetxt( fname='{}/Y_calibration.txt'.format(path), X=self.Y[:self.Nc,:], delimiter=delimiter, fmt='%10.5f' )
			np.savetxt( fname='{}/Y_test.txt'.format(path), X=self.Y[self.Nc:,:], delimiter=delimiter, fmt='%10.5f' )
			file_help.write('\n{:50} {} \n\n'.format('Ycall_calibration.txt', 'Calibration concentrations for factor(analyte), total test files:{}'.format(self.Y.shape[1])) )
			file_help.write('\n{:50} {} \n\n'.format('Ycall_test.txt', 'TEST concentrations for factor(analyte), total test files:{}'.format(self.Y.shape[1])) )
			

			# **** Save CAL and TEST data as matlab array **** # ------------------------------------------------------------------------------
			# --- save complete data array in ONE file with MATLAB structure --- #
			try: 	os.makedirs( '/'.join(path.split('/'))+'/matlabfiles' )
			except:	print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1])+'/matlabfiles' )

			scipy.io.savemat('{}/matlabfiles/X_calibration.mat'.format(path), 	{'X_calibration': data[:self.Nc,:,:]})
			scipy.io.savemat('{}/matlabfiles/Y_calibration.mat'.format(path), 	{'Y_calibration':self.Y[:self.Nc, f]})
			scipy.io.savemat('{}/matlabfiles/X_test.mat'.format(path), 			{'X_test':data[self.Nc:,:,:]})
			scipy.io.savemat('{}/matlabfiles/Y_test.mat'.format(path), 			{'Y_test':self.Y[self.Nc:, f]})

			scipy.io.savemat('{}/matlabfiles/all.mat'.format(path), 			{	'X_calibration': data[:self.Nc,:,:], 	'X_test':data[self.Nc:,:,:],
																					'Y_calibration':self.Y[:self.Nc, f],	'Y_test':self.Y[self.Nc:, f]})

		else:
			# **** Save all samples **** # ------------------------------------------------------------------------------
			# **** (Xx) CALIBRATION and TEST together  **** #
			for n in range(data.shape[0]):	
				np.savetxt( fname='{}/sample{}.txt'.format(path, n), X=data[n,:,:], delimiter=delimiter )
			print('Save individual samples... check... OK! ' )

			# **** (Xx) Make call file **** #
			file_call = open('{}/call_file.txt'.format(path8, n), 'w')
			for n in range(data.shape[0]):	file_call.write('sample{}.txt\n'.format(n))
			file_call.close()
			print('Save call file... check... OK! ' )

		
		return None

	def export_4D_data(self, data=None, path=None, delimiter=None, v=True):

		# **** Check data integrity **** #
		if path == None: 
			print('WARNNING :: code 005A :: DATA.export_4D_data() :: undefined path ' )
			path = './data_export/'

		if delimiter == None: delimiter = '\t'

		if not type(data) is np.ndarray and type(self.X) is np.ndarray:
			data = self.X
		if type(data) is np.ndarray and len(data.shape) == 4:
			print('Data shape... check... OK! ' )

		if v: print('Exporting data set...' )
		
		# **** Create path **** #
		try:		os.makedirs( '/'.join(path.split('/')) )
		except:	    print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1]))
		
		# **** DATA helper **** #
		file_help = open('{}/README(data_help).txt'.format(path), 'w')
		file_help.write('Brief automatic generated description of the dataset \n' )

		file_help.write('{:50} : {} \n'.format('Full data(X) set shape', 				data.shape))
		file_help.write('{:50} : {} \n'.format('Response variable(Y)  shape', 			self.Y.shape))
		file_help.write('{:50} : {} \n'.format('Calibrated factors (analytes)', 		self.Na))
		file_help.write('{:50} : {} \n'.format('UnCalibrated factors (interferents)', 	self.Ni))
		file_help.write('{:50} : {} \n'.format('Total number of samples', 				self.N))
		file_help.write('{:50} : {} \n'.format('Calibration samples',					self.Nc))
		file_help.write('{:50} : {} \n'.format('TEST samples', 							self.Nt))
		file_help.write('{:50} : {} \n'.format('Well aligned channels',					self.a))
		file_help.write('{:50} : {} \n'.format('Analisys model', 						self.model))
		file_help.write('{:50} : {} \n'.format('Alignation model',						self.aling))

		try:	file_help.write('{:50} : {} [{}] \n'.format('Spectra',					len(self.S), ';'.join(['({},{})'.format(n.shape[0], n.shape[1]) for n in self.S]) ) )
		except: pass

		file_help.write('{} {}'.format('-'*100,'\n'))
		file_help.write('File name {} Content \n'.format(' '*30) )
		file_help.write('{} {}'.format('-'*100,'\n'))

		min_channel = np.argmin( data.shape )
		min_len = data.shape[min_channel]
		data_roll = np.rollaxis(data, min_channel, 4)

		# **** (S) spectra  **** # **** **** **** **** **** **** 
		try:
			if type(self.S) == list:
				file_spectra = open('{}/spectra.txt'.format(path), 'w')
				for sn, s in eknumerate(self.S):
					file_spectra.write('spectra_channel_{}.txt'.format(sn))
					np.savetxt( fname='{}/spectra_channel_{}.txt'.format(path, sn), X=self.S[sn], delimiter=delimiter, fmt='%10.5f' )
				file_spectra.close()
			print('Save spectra... check... OK! ' )
		except: pass

		if type(self.Nc) == int and type(self.Nt) == int:
			# **** Save CAL and TEST samples **** # ------------------------------------------------------------------------------
			# **** (X) CALIBRATION  **** # **** **** **** **** **** **** 
			try: 	os.makedirs( '/'.join(path.split('/'))+'/calibration_samples' )
			except:	print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1])+'/calibration_samples' )
			
			for nc in range(self.Nc):	
				try: 	os.makedirs( '/'.join(path.split('/'))+'/calibration_samples/sample{}'.format(nc+1) )
				except:	print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1])+'/calibration_samples/sample{}'.format(nc+1) )
			
				for s in range( min_len):
					np.savetxt( fname='{0}/calibration_samples/sample{1}/variable{2}.txt'.format(path, nc+1, s+1), X=data_roll[nc,:,:,s], delimiter=delimiter, fmt='%10.5f' )
					file_help.write('{0} {1} \n'.format('calibration_sample{0} :: variable{1}.txt'.format(nc+1,s+1), 'Calibration data file number {}/{}, shape {}, min:{:.4}  max:{:.4}'.format(nc+1, self.Nc, data_roll[nc,:,:,s].shape, np.min(data_roll[nc,:,:,s]), np.max(data_roll[nc,:,:,s])) ))
			print('Save individual CALIBRATION samples... check... OK! ' )

			# **** (X) Make caller file **** #
			file_caller = open('{}/caller_file_calibration.txt'.format(path), 'w')
			for nc in range(self.Nc):	
				file_caller.write('calibration_samples/call_file_calibration_sample_{1}.txt\n'.format(path, nc+1))
			file_caller.close()
			file_help.write('\n{:50} {} \n\n'.format('caller_file_calibration.txt', 'Names of cell calibration files, total calibration samples:{}, total variable number:{}'.format(self.Nc, min_len)) )
			print('Save CALIBRATION caller file... check... OK! ' )

			# **** (X) Make call file **** #
			file_call = open('{}/call_file_calibration.txt'.format(path), 'w')
			for nc in range(self.Nc):	
				file_call_sample = open('{0}/calibration_samples/call_file_calibration_sample_{1}.txt'.format(path, nc+1), 'w')
				for s in range( min_len):	
					file_call.write('/calibration_samples/sample{0}/variable{1}.txt\n'.format(nc+1, s+1))
					file_call_sample.write('sample{0}/variable{1}.txt\n'.format(nc+1, s+1))
				file_help.write('\n{:50} {} \n\n'.format('call_file_calibration_sample_{0}.txt'.format(nc+1), 'Names of calibration files, total calibration samples:{}, total variable number:{}'.format(self.Nc, min_len)) )
				file_call_sample.close()
			file_call.close()
			file_help.write('\n{:50} {} \n\n'.format('call_file_calibration.txt', 'Names of calibration files, total calibration samples:{}, total variable number:{}'.format(self.Nc, min_len)) )
			print('Save CALIBRATION call file... check... OK! ' )

			# **** (x) TEST  **** # **** **** ****  **** **** **** 
			try: 	os.makedirs( '/'.join(path.split('/'))+'/test_samples' )
			except:	print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1])+'/test_samples' )
			
			for nt in range(self.Nt):	
				try: 	os.makedirs( '/'.join(path.split('/'))+'/test_samples/sample{}'.format(nt+1) )
				except:	print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1])+'/test_samples/sample{}'.format(nt+1) )
			
				for s in range( min_len):
					#np.savetxt( fname='{}/test_sample{}.txt'.format(file, nt+1), X=data[self.Nc+nt,:,:], delimiter=delimiter, fmt='%10.5f' )
					np.savetxt( fname='{0}/test_samples/sample{1}/variable{2}.txt'.format(path, nt+1, s+1), X=data_roll[self.Nc+nt,:,:,s], delimiter=delimiter, fmt='%10.5f' )
					file_help.write('{0} {1} \n'.format('test_sample{0} :: variable{1}.txt'.format(nt+1,s+1), 'Test data file number {}/{}, shape {}, min:{:.4}  max:{:.4}'.format(self.Nc+nt+1, self.Nc, data_roll[self.Nc+nt,:,:,s].shape, np.min(data_roll[self.Nc+nt,:,:,s]), np.max(data_roll[self.Nc+nt,:,:,s])) ))

				file_help.write('{:50} {} \n'.format('test_sample{}.txt'.format(nt+1), 'TEST data file number {}/{}, shape {}, min:{:.4}  max:{:.4}'.format(nt+1, self.Nt, data_roll[nc,:,:,s].shape, np.min(data_roll[self.Nc+nt,:,:,s]), np.max(data_roll[self.Nc+nt,:,:,s])) ))
			print('Save individual TEST samples... check... OK! ' )


			# **** (X) Make caller file **** #
			file_caller = open('{}/caller_file_test.txt'.format(path), 'w')
			for nt in range(self.Nt):	
				file_caller.write('test_samples/call_file_test_sample_{1}.txt\n'.format(path, nt+1))
			file_caller.close()
			file_help.write('\n{:50} {}\n\n'.format('caller_file_calibration.txt', 'Names of cell test files, total test samples:{}, total variable number:{}'.format(self.Nc, min_len)) )
			print('Save TEST caller file... check... OK! ' )

			# **** (x) Make call file **** # 
			file_call = open('{}/call_file_test.txt'.format(path), 'w')
			for nt in range(self.Nt):		
				file_call_sample = open('{0}/test_samples/call_file_test_sample_{1}.txt'.format(path, nt+1), 'w')
				for s in range( min_len):
					file_call.write('/test_samples/sample{0}/variable{1}.txt\n'.format(nt+1, s+1))
					file_call_sample.write('sample{0}/variable{1}.txt\n'.format(nt+1, s+1))
				file_help.write('\n{:50} {}\n\n'.format('call_file_test_sample_{0}.txt'.format(nc+1), 'Names of test files, total test samples:{}, total variable number:{}'.format(self.Nc, min_len)) )
				file_call_sample.close()
			file_call.close()
			file_help.write('\n{:50} {} \n\n'.format('call_file_test.txt', 'Names of test files, total test files:{}'.format(self.Nt)) )
			print('Save TEST call file... check... OK! ' )

			# **** Save CAL and TEST response values **** # ------------------------------------------------------------------------------
			print(self.Y.shape)
			for f in range(self.Y.shape[1]):
				# **** Y CALIBRATION factor f **** #
				file_calibration = open('{}/Y_calibration_factor{}.txt'.format(path, f+1), 'w')
				for nc in range(self.Nc):		
					file_calibration.write('{}\n'.format(self.Y[nc, f] ))
				file_calibration.close()
				file_help.write('{:50} {} \n'.format('Y_calibration_factor{}.txt'.format(f+1), 'Calibration concentrations for factor(analyte):{}/{} min:{:.4}  max:{:.4}'.format(f+1, self.Y.shape[1], np.min(self.Y[:self.Nc, f]), np.max(self.Y[:self.Nc, f]) ) )) # , np.min(self.Y[:self.Nc, f], np.max(self.Y[:self.Nc, f]) )) ))
				print('Save response(Y) CALIBRATION factor {}... check... OK! '.format(f+1) )

				# **** Y TEST factor f **** #
				file_test = open('{}/Y_test_factor{}.txt'.format(path, f+1), 'w')
				for nt in range(self.Nt):		
					file_test.write('{}\n'.format(self.Y[self.Nc+nt, f] ))
				file_test.close()
				file_help.write('{:50} {}\n'.format('Y_test_factor{}.txt'.format(f+1), 'TEST concentrations for factor(analyte):{}/{} min:{:.4}  max:{:.4}'.format(f+1, self.Y.shape[1], np.min(self.Y[self.Nc:, f]), np.max(self.Y[self.Nc:, f]) ) )) # , np.min(self.Y[:self.Nc, f], np.max(self.Y[:self.Nc, f]) )) ))
				print('Save response(Y) TEST factor {}... check... OK! '.format(f+1) )
				
			file_call_calibration = open('{}/Ycall_calibration.txt'.format(path, f+1), 'w')
			file_call_test        = open('{}/Ycall_test.txt'.format(path, f+1), 'w')
			for f in range(self.Y.shape[1]):
				file_call_calibration.write('Y_calibration_factor{}.txt\n'.format(f+1) )
				file_call_test.write('Y_test_factor{}.txt\n'.format(f+1) )

			np.savetxt( fname='{}/Y_calibration.txt'.format(path), X=self.Y[:self.Nc,:], delimiter=delimiter, fmt='%10.5f' )
			np.savetxt( fname='{}/Y_test.txt'.format(path), X=self.Y[self.Nc:,:], delimiter=delimiter, fmt='%10.5f' )
			file_help.write('\n{:50} {} \n\n'.format('Ycall_calibration.txt', 'Calibration concentrations for factor(analyte), total test files:{}'.format(self.Y.shape[1])) )
			file_help.write('\n{:50} {} \n\n'.format('Ycall_test.txt', 'TEST concentrations for factor(analyte), total test files:{}'.format(self.Y.shape[1])) )
			
			# **** Save CAL and TEST data as matlab array **** # ------------------------------------------------------------------------------
			# --- save complete data array in ONE file with MATLAB structure --- #
			try: 	os.makedirs( '/'.join(path.split('/'))+'/matlabfiles' )
			except:	print("WARNING :: DATA.export_4D_data() :: Creation of the directory %s failed" % '/'.join(path.split('/')[:-1])+'/matlabfiles' )

			scipy.io.savemat('{}/matlabfiles/X_calibration.mat'.format(path), 	{'X_calibration': data[:self.Nc,:,:,:]})
			scipy.io.savemat('{}/matlabfiles/Y_calibration.mat'.format(path), 	{'Y_calibration':self.Y[:self.Nc, f]})
			scipy.io.savemat('{}/matlabfiles/X_test.mat'.format(path), 			{'X_test':data[self.Nc:,:,:,:]})
			scipy.io.savemat('{}/matlabfiles/Y_test.mat'.format(path), 			{'Y_test':self.Y[self.Nc:, f]})

			scipy.io.savemat('{}/matlabfiles/all.mat'.format(path), 			{	'X_calibration': data[:self.Nc,:,:,:], 	'X_test':data[self.Nc:,:,:,:],
																					'Y_calibration':self.Y[:self.Nc, f],	'Y_test':self.Y[self.Nc:, f]})

		return True

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== MODIFICATION ==================== MODIFICATION ==================== # # ==================== MODIFICATION ====================  MODIFICATION ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
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

	# ***** EDIT ***** #
	def transpose_order_channels(self, data=None, channels_info=None, save=True, v=0):
		if not type(data) is np.ndarray:
			if not type(data) is list:		data = self.X
			else: 							data = np.array(data)
		try:
			if (type(channels_info) is np.ndarray or type(channels_info) is list or type(channels_info) is tuple) and type(data) is np.ndarray:
				order = [ i for i, n in enumerate(channels_info) if n == 0 ] + [ i for i, n in enumerate(channels_info) if n == 1 ] + [ i for i, n in enumerate(channels_info) if n == -1 ]
				print( 'Order channel by : {} '.format(order) )
				data = np.transpose(data, order)
				if save: self.X = data; self.D = data
		except: print('ERROR :: code 001B :: DATA.transpose_order_channels.py :: can NOT order channel in this way {}.'.format(channels_info) )
		
		if v and type(self.X) is np.ndarray:		print(self.X.shape)
		if v and type(self.D) is np.ndarray:		print(self.D.shape)

		return data

	def transpose(self, s, X=None, axes=None):
		try: 
			if not X == None: self.X = X
			else: X = self.X
		except: pass

		try: s = np.array(s)
		except: pass

		try: 
			error = 0
			if s.all() == None: print('ERROR :: code 001 :: DATA.py :: missing data need expected X shape.'); error = 1
			if len(X.shape) != s.shape[0]: print('ERROR :: code 002 :: DATA.py :: NOT compatible shape.'); error = 2
		except: print('ERROR :: code 00? :: UNKNOW.'); error = 3

		if error == 0:
			try:
				ref = []
				for i,n in enumerate(s):
					for j,m in enumerate(X.shape):
						if n == m: ref.append( j )
				if len(ref) == len(X.shape):	self.X = np.transpose(X, ref)
				else: 	print('ERROR :: code 004 :: DATA.py :: NOT compatible shape.')
			except: pass

	# ***** UTILITy ***** #
	def isnum(self, n):
		# ------------------ Define if n is or not a number ------------------ # 
		# n     :   VAR     :   VAR to check if it is a numerical VAR
		# return :  BOOL    : True/False
		try: float(n); return True
		except: return False

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== GETTERs ==================== GETTERs ==================== # # ==================== GETTERs ====================  GETTERs ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	def get_complexity(self, coeficient=[], verbosity=False, ):
		return self.complexity_analysis()

	def get_entropy(self, definition='shannon', S_dimention='1D', 
						  f=None, Ni=None, Na=None, Nc=None, Nt=None, X=None, 
						  verbosity=True, plot=False,
						  ): 

		def information(prob):
			# quantifies the amount of surprise of a probability
			return -np.log2(prob)


		def entropy(probs):
		    # quantifies the average amount of surprise
		    p = np.full(probs.shape, 0.0)
		    np.log2(probs, out=p, where=(probs > 0))
		    return -((p * probs).sum())


		def relative_entropy(probs1, probs2):
		    # kullback-leibler divergence
		    # measures how different is probs1 from probs2
		    # it is an "distribution-wise" asymmetric measure
		    # result ranges from 0.0 to inf
		    p = np.empty_like(probs1)
		    p[probs2 == 0.0] = np.inf
		    p[probs1 == 0.0] = 0
		    mask = (probs1 != 0.0) & (probs2 != 0.0)
		    np.divide(probs1, probs2, out=p, where=mask)
		    np.log2(p, out=p, where=mask)
		    np.multiply(p, probs1, out=p, where=mask)
		    return p.sum()


		def cross_entropy(probs1, probs2):
		    # note that probs1 represent the true distribution
		    # and probs2 might be the estimated distribution
		    # measures the average number of bits needed to identify
		    # an event drawn from the set if a coding scheme used for the set
		    # is optimized for probs2 rather than probs1
		    # this is like get the average surprise of random variable probs1
		    # but also add in the relative extra surprise between probs1 and probs2
		    # assuming that the true data might be assumed to be probs2 instead
		    return entropy(probs1) + relative_entropy(probs1, probs2)


		def cross_entropy2(probs1, probs2):
		    # alternative equivalent definition of cross entropy
		    return -((probs1 * np.log2(probs2)).sum())

		def estimate_discrete_shannon_entropy(A):
			# (1) Your probability distribution is discrete. 
			# Then you have to translate what appear to be relative 
			# frequencies to probabilities.
			pA = A / A.sum()
			entropy_value = -np.sum(pA*np.log2(pA))
			return entropy_value

		def estimate_continuous_shannon_entropy(A):
			# 2) Your probability distribution is continuous. 
			# In that case the values in your input needn't sum to one. 
			# Assuming that the input is sampled regularly from the 
			# entire space, you'd get
			pA = A / A.sum()
			entropy_value = -np.sum(pA*np.log2(A))
			return entropy_value


		def vn_eig_entropy(rho):
			EV = np.linalg.eigvals(rho)

			# Drop zero eigenvalues so that log2 is defined
			my_list = [x for x in EV.tolist() if x]
			EV = np.array(my_list)

			log2_EV = np.matrix(np.log2(EV))
			EV = np.matrix(EV)
			S = -np.dot(EV, log2_EV.H)
			return(S)

		Ni = Ni if type(Ni) != type(None) else self.Ni
		Na = Na if type(Na) != type(None) else self.Na
		Nc = Nc if type(Nc) != type(None) else self.Nc
		Nt = Nt if type(Nt) != type(None) else self.Nt
		f  = f  if type(f)  != type(None) else self.f

		D  = X  if type(X)  != type(None) else self.D
		X  = X  if type(X)  != type(None) else self.X
		if verbosity:
			print('*** Evaluating information entropy *** ')
			print(f' >> data.get_entropy({definition}, {S_dimention})')



		if 	 len(D.shape) == 1: pass # NEED implementation
		elif len(D.shape) == 2: pass # NEED implementation
		elif len(D.shape) == 3: pass # NEED implementation
		elif len(D.shape) == 4:
			
			N, l1, l2, l3 = D.shape
			De = np.sum(np.sum(D, axis=1), axis=1)
	
			entropy_cross_entropy 	= np.zeros((N,N))
			entropy_sample 			= np.zeros(N)
			entropy_sample_noinfo	= 0
			entropy_sensor 			= np.zeros(l3)
			entropy_sensor_noinfo 	= 0
	
			if S_dimention == '2D': # NEED implementation / need to separe sample from test
				if N != l3:
					M = np.zeros( (N+l3, N+l3) )
					M[l3:,:l3] = De
					entropy_sample = vn_eig_entropy( M )
				else:
					M = De
					entropy_sample = vn_eig_entropy( M )

			elif S_dimention == '1D':
				entropy_sample = np.zeros(N)
				for n in range(N):
					entropy_sample[n] = entropy( De[n, :]/np.linalg.norm(De[n,:]) )
				entropy_sample_noinfo = entropy( np.ones(l3)/np.linalg.norm(np.ones(l3)) )

				entropy_sensor = np.zeros(l3)
				for n in range(l3):
					entropy_sensor[n] = entropy( De[:, n]/np.linalg.norm(De[:, n]) )
				entropy_sensor_noinfo = entropy( np.ones(N)/np.linalg.norm(np.ones(N)) )

				entropy_cross_entropy = np.zeros((N,N))
				for n1 in range(N):
					for n2 in range(N):
						entropy_cross_entropy[n1][n2] = cross_entropy( 	De[n1, :]/np.linalg.norm(De[n1,:]), 
													 				   	De[n2, :]/np.linalg.norm(De[n2,:]) )

				if plot:
					plt.figure(1), plt.plot(entropy_sensor, '-', c=(0.8,0.3,0.3), alpha=0.9)
					plt.figure(1), plt.plot([0, l3], [entropy_sensor_noinfo, entropy_sensor_noinfo], ':', c=(0.3,0.3,0.3), alpha=0.5)
					
					plt.figure(2), plt.plot(entropy_sample, 'o', c=(0.3,0.8,0.3), alpha=0.9)
					plt.figure(2), plt.plot([0, N], [entropy_sample_noinfo, entropy_sample_noinfo], ':', c=(0.3,0.3,0.3), alpha=0.5 )
					plt.figure(2), plt.plot([Nc, Nc], [0, np.max(entropy_sample)], ':', c=(0.3,0.3,0.3), alpha=0.5)
					
					plt.matshow(entropy_cross_entropy)
					plt.show()

		return {'entropy_calibration': 			entropy_sample,
				'entropy_test': 				entropy_sample_noinfo,
				'entropy_calibration_noinfo': 	entropy_sensor,
				'entropy_test_noinfo': 			entropy_sensor_noinfo,
				'cross_entropy': 				entropy_cross_entropy,
				}
				
