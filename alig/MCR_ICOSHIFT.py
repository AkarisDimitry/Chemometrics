from __future__ import division, print_function

# *** warning supresion
import warnings
warnings.filterwarnings("ignore")
from functools import reduce
import logging, operator, sys
from copy import copy

# *** numeric libraries
import numpy
import numpy as np #pip3 install numpy
try:
	import scipy
	import scipy as sp
	import scipy.sparse
except: 
	print('WARNNING :: PARAFAC.py :: can NOT correctly load "scipy" libraries')

# *** graph libraries
try:
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pl
except: 
	print('WARNNING :: PARAFAC.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by "pip3 install matplotlib" ')

class MCR_ICOSHIFT(object):
	def __init__(self, X=None, D=None, Xc=None, Dc=None, Xa=None, Da=None, C=None, S=None,
				L=None, La=None, f=None, A=None, a=1, 
				Y=None, y=None,	N=None, Nc=None, Na=None, Ni=None, Nt=None,
				shift_range=None, warp_range=None ):

		self.D = D
		self.X = X

		self.Dc = Dc
		self.Xc = Xc

		self.Da = Da
		self.Xa = Xa

		self.S = np.array(S)
		self.C = np.array(C)

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

		self.MRC_train = None
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


	def Uncompress_2WAY(self, Xc=None, Dc=None, save=True, shape=None):
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
		if   type(Xc) is list: 							Xc = np.array(Xc)
		elif type(Xc) is np.ndarray:					pass
		elif type(self.Xc) is np.ndarray:				Xc = np.array(self.Xc)

		if type(Xc) == type(None) and type(Dc) is list: 			Xc = np.array(Dc)
		elif type(Xc) == type(None) and type(Dc) is np.ndarray:		Xc = Dc

		if type(shape) == type(None):
			print('ERROR :: ICOSHIFT.Uncompress_3WAY() :: the original shape is needed in order to restore the original tensor ')

		# step 1 | Decompresion 
		Da = np.reshape(Xc, newshape=[shape[1],shape[0],shape[2]], order='C')
		Da = np.moveaxis(Da , 1, 0)

		# step 2 | save 
		if save: self.Da = Da
		else: pass
		
		return Da

	def Uncompress_3WAY(self, Xc=None, Dc=None, save=True, shape=None):
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
		if   type(Xc) is list: 							Xc = np.array(Xc)
		elif type(Xc) is np.ndarray:					pass
		elif type(self.Xc) is np.ndarray:				Xc = np.array(self.Xc)

		if type(Xc) == type(None) and type(Dc) is list: 			Xc = np.array(Dc)
		elif type(Xc) == type(None) and type(Dc) is np.ndarray:		Xc = Dc

		if type(shape) == type(None):
			print('ERROR :: MCR_ICOSHIFT.Uncompress_3WAY() :: the original shape is needed in order to restore the original tensor ')

		# step 1 | Decompresion ]

		Da = np.reshape(Xc, newshape=[shape[1],shape[2],shape[0],shape[3]], order='C')
		Da = np.moveaxis(Da , 2, 0)

		# step 2 | save 
		if save: self.Da = Da
		else: pass
		
		return Da

	def estructure_3WAY(self, D=None, S=None, C=None, v=False, save=True ):
		if type(D) == type(None):	D = self.D
		elif v: pass
		if type(S) == type(None):	S = self.S
		elif v: pass
		if type(C) == type(None):	C = self.C
		elif v: pass

		N, l1, l2, l3 = D.shape
		f = S.shape[1]

		Da = np.zeros_like(D)

		for n in range(f):
			Da += np.rollaxis(np.tensordot( C[n,:,:], S[:,n], axes=0 ).reshape(N, l3, l1, l2), 1, 4)
		
		if save: self.Da = Da

		return Da

	def aling(self, D=None, Dc=None, Da=None,
					X=None, Xc=None, Xa=None, 
					constraints={'Normalization':['S'], 'non-negativity':['all']}, inicializacion='random', max_iter=200,
					augment=None, save=True, v=1):
		# ---------------------------------------- #
		# ----- (0) Check some variables      ---- #
		# ---------------------------------------- #
		try:
			print('Reading data...')
			if not type(D) is np.ndarray and type(self.D) is np.ndarray: 	D = self.D
			else: print('ERROR :: code 000 :: MCR_ICOSHIFT.aling() :: can NOT get D or self.D as np.array type')
			if not type(X) is np.ndarray and type(self.X) is np.ndarray:	X = self.X		
		except:
			print('ERROR :: code 000 :: MCR_ICOSHIFT.aling() :: can NOT get D or self.D')
		
		try:
			print('Reading data...')
			if not type(Dc) is np.ndarray:
				pass
			if type(self.D) is np.ndarray: 	D = self.D
			if not type(X) is np.ndarray and type(self.X) is np.ndarray:	X = self.X		
		except:
			print('ERROR :: code 000 :: MCR_ICOSHIFT.aling() :: can NOT get D or self.D')
		
		if not type(augment) == type(None):
			self.Dc = np.rollaxis(D, augment)
			self.Xc = np.rollaxis(X, augment)

		# ---------------------------------------- #
		# ----- (1) Train and solve MCR model ---- #
		# ---------------------------------------- #
		MRC_train = self.train(Dc=None, S=None, constraints={'Normalization':['S'], 'non-negativity':['all']}, 
						inicializacion='random', max_iter=200, save=True)

		# ---------------------------------------- #
		# ----- (2) Aling data with ICOSHIFT  ---- #
		# ---------------------------------------- #		
		Ca = np.array([ self.aling_ICOSHIFT_2D(xt='sample', X=self.Ce[:,:,i])[0] for i in range( self.Ce.shape[2] ) ])

		# ------------------------------------------ #
		# ----- (3) Restore original data shape ---- #
		# ------------------------------------------ #
		Da = self.Da if type(Da) == type(None) else Da
		if type(Da) == type(None): 	
			if type(self.D) == type(None): 	
				print('ERROR :: Null extended tensor :: Dc = None')
			elif len(self.D.shape) == 3:
				Da = self.estructure_2WAY(D=self.D, S=self.S, C=Ca, v=0) # TODO
			elif len(self.D.shape) == 4:
				Da = self.estructure_3WAY(D=self.D, S=self.S, C=Ca, v=0)
			elif len(self.D.shape) > 4:
				Da = self.estructure_nWAY(D=self.D, S=self.S, C=Ca, v=0) # TODO

		# ---------------------------------------- #
		# ----- (4)   	Store data 	 		  ---- #
		# ---------------------------------------- #
		if save: 
			self.Dc = Dc
			self.Da = Da
			self.Ca = Ca
			self.MRC_train = MRC_train

		return Da, MRC_train, Ca
	
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

	def isnum(self, n):
		# ------------------ Define if n is or not a number ------------------ # 
		# n     :   VAR     :   VAR to check if it is a numerical VAR
		# return :  BOOL    : True/False
		try: float(n); return True
		except: return False

	def isINT(self, n):
		# ------------------ Define if n is or not a INT ------------------ # 
		# n     :   VAR     :   VAR to check if it is a numerical VAR
		# return :  BOOL    : True/False
		try: int(n); return True
		except: return False

	def aling_ICOSHIFT_2D(self, xt='average',  X=None,  inter='whole',  n='f', scale=None, coshift_preprocessing=False,
					 coshift_preprocessing_max_shift=None, fill_with_previous=True, average2_multiplier=3):

		def is_number(s):
			try:
				float(s)
				return True
			except ValueError:
				return False

		def cat(dim, *args):
			return numpy.concatenate([r for r in args if r.shape[0] > 0], axis=dim)

		def sortrows(a, i):
			i = numpy.argsort(a[:, i])
			b = a[i, :]
			return b

		def nans(r, c):
			a = numpy.empty((r, c))
			a[:] = numpy.nan
			return a

		def min_with_indices(d):
			d = d.flatten()
			ml = numpy.min(d)
			mi = numpy.array(list(d).index(ml))
			return ml, mi

		def max_with_indices(d):
			d = d.flatten()
			ml = numpy.max(d)
			mi = numpy.array(list(d).index(ml))
			return ml, mi


		def icoshift(xt,  xp,  inter='whole',  n='f', scale=None, coshift_preprocessing=False,
					 coshift_preprocessing_max_shift=None, fill_with_previous=True, average2_multiplier=3):
			'''
			interval Correlation Optimized shifting
			[xcs, ints, ind, target] = icoshift(xt, xp, inter[, n[, options[, scale]]])
			Splits a spectral database into "inter" intervals and coshift each vector
			left-right to get the maximum correlation toward a reference or toward an
			average spectrum in that interval. Missing parts on the edges after
			shifting are filled with "closest" value or with "NaNs".
			INPUT
			xt (1 * mt)    : target vector.
							 Use 'average' if you want to use the average spectrum as a reference
							 Use 'median' if you want to use the median spectrum as a reference
							 Use 'max' if you want to use for each segment the corresponding actual spectrum having max features as a reference
							 Use 'average2' for using the average of the average multiplied for a requested number (default=3) as a reference
			xp (np * mp)   : Matrix of sample vectors to be aligned as a sample-set
							 towards common target xt
			inter          : definition of alignment mode
							 'whole'         : it works on the whole spectra (no intervals).
							 nint            : (numeric) number of many intervals.
							 'ndata'         : (string) length of regular intervals
											   (remainders attached to the last).
							 [I1s I1e, I2s...]: interval definition. ('i(n)s' interval
											   n start,  'i(n)e' interval n end).
							 (refs:refe)     : shift the whole spectra according to a
											   reference signal(s) in the region
											   refs:refe (in sampling points)
							 'refs-refe'     : `shift the whole spectra according to a
											   reference signal(s) in the region
											   refs-refe (in scale units)
			n (1 * 1)      : (optional)
							 n = integer n.: maximum shift correction in data
											 points/scale units (cf. options[4])
											 in x/rows. It must be >0
							 n = 'b' (best): the algorithm search for the best n
											 for each interval (it can be time consuming!)
							 n = 'f' (fast): fast search for the best n for each interval (default)
							 a logging.warn is displayed for each interval if "n" appears too small
			options (1 * 5): (optional)
							 (0) triggers plots & warnings:
								 0 : no on-screen output
								 1 : only warnings (default)
								 2 : warnings and plots
							 (1) selects filling mode
								 0 : using not a number
								 1 : using previous point (default)
							 (2) turns on Co-shift preprocessing
								 0 : no Co-shift preprocessing (default)
								 1 :
							 (3)
								 it has to be given in scale units if option(5)=1
							 (4) 0 : intervals are given in No. of datapoints  (deafult)
								 1 : intervals are given in ppm --> use scale for inter and n
			scale           : vector of scalars used as axis for plot (optional)
			coshift_preprocessing (bool) (optional; default=False): Execute a Co-shift step before carrying out iCOshift
			coshift_preprocessing_max_shift (int) (optional): Max allowed shift for the Co-shift preprocessing
															  (default = equal to n if not specified)
			fill_with_previous (bool) (optional; default=True): Fill using previous point (default); set to False to np.nan
			average2_multiplier (int) (optional; default=3): Multiplier used for the average2 algorithm
			OUTPUT
			xcs  (np * mt): shift corrected vector or matrix
			ints (ni * 4) : defined intervals (Int. No.,  starting point,  ending point,  size)
			ind  (np * ni): matrix of indexes reporting how many points each spectrum
							has been shifted for each interval (+ left,  - right)
			target (1 x mp): actual target used for the final alignment
			Authors:
			Francesco Savorani - Department of Food Science
								 Quality & Technology - Spectroscopy and Chemometrics group
								 Faculty of Sciences
								 University of Copenhagen - Denmark
			email: frsa@life.ku.dk - www.models.life.ku.dk
			Giorgio Tomasi -     Department of Basic Science and Environment
								 Soil and Environmental Chemistry group
								 Faculty of Life Sciences
								 University of Copenhagen - Denmark
			email: giorgio.tomasi@ec.europa.eu - www.igm.life.ku.dk
			Python implementation by:
			Martin Fitzpatrick -  Rheumatology Research Group
								  Centre for Translational Inflammation Research
								  School of Immunity and Infection
								  University of Birmingham - United Kingdom
			email: martin.fitzpatrick@gmail.com
			170508 (FrSa) first working code
			211008 (FrSa) improvements and bugs correction
			111108 (Frsa) Splitting into regular intervals (number of intervals or wideness in datapoints) implemented
			141108 (GT)   FFT alignment function implemented
			171108 (FrSa) options implemented
			241108 (FrSa) Automatic search for the best or the fastest n for each interval implemented
			261108 (FrSa) Plots improved
			021208 (FrSa) 'whole' case added & robustness improved
			050309 (GT)   Implentation of interpolation modes (nan); Cosmetics; Graphics
			240309 (GT)   Fixed bug in handling missing values
			060709 (FrSa) 'max' target and output 'target' added. Some speed,  plot and robustness improvements
			241109 (GT)   Interval and band definition in units (added options[4])
			021209 (GT)   Minor debugging for the handling of options[4]
			151209 (FrSa) Cosmetics and minor debugging for the handling of options[4]
			151110 (FrSa) Option 'Max' works now also for alignment towards a reference signal
			310311 (FrSa) Bugfix for the 'whole' case when mp < 101
			030712 (FrSa) Introducing the 'average2' xt (target) for a better automatic target definition. Plots updated to include also this case
			281023 (MF)   Initial implementation of Python version of icoshift algorithm. PLOTS NOT INCLUDED
			'''

			# RETURNS [xcs, ints, ind, target]

			# Take a copy of the xp vector since we mangle it somewhat
			xp = copy(xp)


			if scale is None:
				using_custom_scale = False
				scale = numpy.array(range(0, xp.shape[1]))

			else:
				using_custom_scale = True

				dec_scale = numpy.diff(scale)
				inc_scale = scale[0] - scale[1]

				flag_scale_dir = inc_scale < 0
				flag_di_scale = numpy.any(abs(dec_scale) > 2 * numpy.min(abs(dec_scale)))

				if len(scale) != max(scale.shape):
					raise(Exception, 'scale must be a vector')

				if max(scale.shape) != xp.shape[1]:
					raise(Exception, 'x and scale are not of compatible length %d vs. %d' % (max(scale.shape), xp.shape[1]))

				if inc_scale == 0 or not numpy.all(numpy.sign(dec_scale) == - numpy.sign(inc_scale)):
					raise(Exception, 'scale must be strictly monotonic')

			if coshift_preprocessing_max_shift is None:
				coshift_preprocessing_max_shift = n

			# ERRORS CHECK
			# Constant
			# To avoid out of memory errors when 'whole',  the job is divided in
			# blocks of 32MB
			block_size = 2 ** 35


			max_flag = False
			avg2_flag = False

			xt_basis = xt

			if xt == 'average':
				xt = numpy.nanmean(xp, axis=0).reshape(1, -1)

			elif xt == 'median':
					xt = numpy.nanmedian(xp, axis=0).reshape(1, -1)

			elif xt == 'average2':
					xt = numpy.nanmean(xp, axis=0).reshape(1,-1)
					avg2_flag = True

			elif xt == 'max':
					xt = numpy.zeros((1, xp.shape[1]))
					max_flag = True

			elif xt == 'sample':
					xt = xp[0,:].reshape(1, -1)

			nt, mt = xt.shape
			np, mp = xp.shape

			if mt != mp:
				raise(Exception, 'Target "xt" and sample "xp" must have the same number of columns')

			if is_number(inter):
				if inter > mp:
					raise(Exception, 'Number of intervals "inter" must be smaller than number of variables in xp')

			# Set defaults if the settings are not set
			# options = [options[oi] if oi < len(options) else d for oi, d in enumerate([1, 1, 0, 0, 0]) ]

			if using_custom_scale:
				prec = abs(numpy.min(numpy.unique(dec_scale)))
				if flag_di_scale:
					logging.warn('Scale vector is not continuous, the defined intervals might not reflect actual ranges')

			flag_coshift = (not inter == 'whole') and coshift_preprocessing

			if flag_coshift:

				if using_custom_scale:
					coshift_preprocessing_max_shift = dscal2dpts(coshift_preprocessing_max_shift, scale, prec)

				if max_flag:
					xt = numpy.nanmean(xp, axis=0)

				co_shift_scale = scale if using_custom_scale else None
				xp, nil, wint, _ = icoshift(xt, xp, 'whole',
											coshift_preprocessing=False,
											coshift_preprocessing_max_shift=coshift_preprocessing_max_shift,
											scale=co_shift_scale,
											fill_with_previous=True,
											average2_multiplier=average2_multiplier )

				if xt_basis == 'average' or xt_basis == 'average2':
					xt = numpy.nanmean(xp).reshape(1, -1)

				elif xt_basis == 'median':
					xt = numpy.nanmedian(xp).reshape(1, -1)

				else:  # max?
					xt = xt.reshape(1,-1)


			whole = False
			flag2 = False

			try:				basestring_check = isinstance(inter, basestring)
			except NameError:	basestring_check = isinstance(inter, str)

			if basestring_check:

				if inter == 'whole':
					inter = numpy.array([0, mp - 1]).reshape(1, -1)
					whole = True

				elif '-' in inter:
					interv = regexp(inter, '(-{0,1}\\d*\\.{0,1}\\d*)-(-{0,1}\\d*\\.{0,1}\\d*)', 'tokens')
					interv = sort(scal2pts(float(cat(0, interv[:])), scale, prec))

					if interv.size != 2:
						raise(Exception, 'Invalid range for reference signal')

					inter = range(interv[0], (interv[1] + 1))
					flag2 = True

				else:

					interv = float(inter)

					if using_custom_scale:
						interv = dscal2dpts(interv, scale, prec)
					else:
						interv = round(interv)

					inter = defints(xp, interv, options[0])

			elif isinstance(inter, int):

				# Build interval list
				# e.g. 5 intervals on 32768 to match MATLAB algorithm should be
				#0, 6554, 6554, 13108, 13108, 19662, 19662, 26215, 26215, 32767

				# Distributes vars_left_over in the first "vars_left_over" intervals
				# startint     = [(1:(N+1):(remain - 1)*(N+1)+1)'; ((remain - 1) * (N + 1) + 1 + 1 + N:N:mP)'];

				remain = mp % inter
				step = int( float(mp) / inter )
				segments = []
				o = 0

				while o < mp:
					segments.extend([o, o])
					if remain > 0:
						o += 1
						remain -=1
					o += step

				# Chop of duplicate zero
				segments = segments[1:]
				segments.append(mp)  # Add on final step
				inter = numpy.array(segments, dtype=int).reshape(1, -1)

				logging.info("Calculated intervals: %s" % inter)

			elif isinstance(inter, list):  # if is a list of tuples ; add else
				inter = np.array(inter)

				flag2 = numpy.array_equal(numpy.fix(inter), inter) and max(inter.shape) > 1 and numpy.array_equal(
					numpy.array([1, numpy.max(inter) - numpy.min(inter) + 1]).reshape(1, -1), inter.shape) and numpy.array_equal(unique(numpy.diff(inter, 1, 2)), 1)

				if not flag2 and using_custom_scale:
					inter = scal2pts(inter, scale, prec)

					if numpy.any(inter[0:2:] > inter[1:2:]) and not flag_scale_dir:
						inter = flipud(numpy.reshape(inter, 2, max(inter.shape) / 2))
						inter = inter[:].T

			else:
				raise(Exception, 'The number of intervals must be "whole", an integer, or a list of tuples of integers')


			nint, mint = inter.shape
			scfl = numpy.array_equal(numpy.fix(scale), scale) and not using_custom_scale

			try:				basestring_check = isinstance(inter, basestring)
			except NameError:	basestring_check = isinstance(inter, str)

			if basestring_check and n not in ['b', 'f']:
				raise(Exception, '"n" must be a scalar b or f')

			elif isinstance(n, int) or isinstance(n, float):
				if n <= 0:
					raise(Exception, 'Shift(s) "n" must be larger than zero')

				if scfl and not isinstance(n, int):
					logging.warn('"n" must be an integer if scale is ignored; first element (i.e. %d) used' % n)
					n = numpy.round(n)
				else:
					if using_custom_scale:
						n = dscal2dpts(n, scale, prec)

				if not flag2 and numpy.any(numpy.diff(numpy.reshape(inter, (2, mint // 2)), 1, 0) < n):
					raise(Exception, 'Shift "n" must be not larger than the size of the smallest interval')

			flag = numpy.isnan(cat(0, xt.reshape(1, -1), xp))
			frag = False
			ref = lambda e: numpy.reshape(e, (2, max(e.shape) // 2)).T
			vec = lambda a: a.flatten()

			mi, pmi = min_with_indices(inter)
			ma, pma = max_with_indices(inter)

			# There are missing values in the dataset; so remove them before starting
			# if they line up between datasets
			if vec(flag).any():

				if numpy.array_equal(flag[numpy.ones((np, 1), dtype=int), :], flag[1:,:]):
					select = numpy.any
				else:
					select = numpy.all

				if flag2:
					intern_ = remove_nan(
						numpy.array([0, pma - pmi]).reshape(1, -1), cat(0, xt[:, inter], xp[:, inter]), select)
					if intern_.shape[0] != 1:
						raise(Exception, 'Reference region contains a pattern of missing that cannot be handled consistently')

					elif not numpy.array_equal(intern_, numpy.array([1, inter[-2] - inter[0] + 1]).reshape(1, -1)):
						logging.warn('The missing values at the boundaries of the reference region will be ignored')

					intern_ = range(inter[0] + intern_[0], (inter[0] + intern_[1] + 1))
				else:
					intern_, flag_nan = remove_nan(
						ref(inter), cat(0, xt, xp), select, flags=True)
					intern_ = vec(intern_.T).T

				if 0 in intern_.shape:
					raise(Exception, 'Cannot handle this pattern of missing values.')

				if max(intern_.shape) != max(inter.shape) and not flag2:
					if whole:
						if max(intern_.shape) > 2:

							xseg, in_or = extract_segments(cat(0, xt, xp), ref(intern_))
							InOrf = in_or.flatten()
							inter = numpy.array([InOrf[0], InOrf[-1] - 1]).reshape(1, -1)
							in_or = cat(1, ref(intern_), in_or)
							xp = xseg[1:, :]
							xt = xseg[0, :].reshape(1, -1)
							frag = True

					else:
						logging.warn('To handle the pattern of missing values, %d segments are created/removed' % (abs(max(intern_.shape) - max(inter.shape)) / 2) )
						inter = intern_
						nint, mint = inter.shape
			xcs = xp
			mi, pmi = min_with_indices(inter)
			ma, pma = max_with_indices(inter)


			flag = max(inter.shape) > 1 and numpy.array_equal(
				numpy.array([1, pma - pmi + 1]).reshape(1, -1), inter.shape) and numpy.array_equal(unique(numpy.diff(inter, 1, 2)), 1)

			if flag:
				if n == 'b':
					logging.info('Automatic searching for the best "n" for the reference window "ref_w" enabled. That can take a longer time.')

				elif n == 'f':
					logging.info('Fast automatic searching for the best "n" for the reference window "ref_w" enabled.')

				if max_flag:
					amax, bmax = max_with_indices( numpy.sum(xp) )
					xt[mi:ma] = xp[bmax, mi:ma]

				ind = nans(np, 1)
				missind = not all(numpy.isnan(xp), 2)
				xcs[missind, :], ind[missind], _ = coshifta(xt, xp[missind,:], inter, n, fill_with_previous=fill_with_previous,
															block_size=block_size)
				ints = numpy.array([1, mi, ma]).reshape(1, -1)

			else:
				if mint > 1:
					if mint % 2:
						raise(Exception, 'Wrong definition of intervals ("inter")')

					if ma > mp:
						raise(Exception, 'Intervals ("inter") exceed samples matrix dimension')

					# allint=[(1:round(mint/2))' inter(1:2:mint)' inter(2:2:mint)'];
					# allint =
					#        1           1        6555
					#        2        6555       13109
					#        3       13109       19663
					#        4       19663       26216
					#        5       26216       32768
					# ans =
					#  5     3

					inter_list = list(inter.flatten())

					allint = numpy.array([
						range(mint//2),
						inter_list[0::2],
						inter_list[1::2],
					])

					allint = allint.T

				sinter = numpy.sort(allint, axis=0)
				intdif = numpy.diff(sinter)

				if numpy.any(intdif[1:2:max(intdif.shape)] < 0):
					logging.warn('The user-defined intervals are overlapping: is that intentional?')

				ints = allint
				ints = numpy.append(ints, ints[:, 2] - ints[:, 1])
				ind = numpy.zeros((np, allint.shape[0]))

				if n == 'b':
					logging.info('Automatic searching for the best "n" for each interval enabled. This can take a long time...')

				elif n == 'f':
					logging.info('Fast automatic searching for the best "n" for each interval enabled')

				for i in range(0, allint.shape[0]):

					if whole:
						logging.info('Co-shifting the whole %s samples...' % np)
					else:
						logging.info('Co-shifting interval no. %s of %s...' % (i, allint.shape[0]) )

					# FIXME? 0:2, or 1:2?
					intervalnow = xp[:, allint[i, 1]:allint[i, 2]]

					if max_flag:
						amax, bmax = max_with_indices(numpy.sum(intervalnow, axis=1))
						target = intervalnow[bmax, :]
						xt[0, allint[i, 1]:allint[i, 2]] = target
					else:
						target = xt[:, allint[i, 1]:allint[i, 2]]

					missind = ~numpy.all(numpy.isnan(intervalnow), axis=1)

					if not numpy.all(numpy.isnan(target)) and numpy.any(missind):

						cosh_interval, loc_ind, _ = coshifta(target, intervalnow[missind, :], 0, n,
															 fill_with_previous=fill_with_previous, block_size=block_size)
						xcs[missind, allint[i, 1]:allint[i, 2]] = cosh_interval
						ind[missind, i] = loc_ind.flatten()

					else:
						xcs[:, allint[i, 1]:allint[i, 1]] = intervalnow

				if avg2_flag:

					for i in range(0, allint.shape[0]):
						if whole:
							logging.info('Co-shifting again the whole %d samples... ' % np)
						else:
							logging.info('Co-shifting again interval no. %d of %d... ' % (i, allint.shape[0]))

						intervalnow = xp[:, allint[i, 1]:allint[i, 2]]
						target1 = numpy.mean(xcs[:, allint[i, 1]:allint[i, 2]], axis=0)
						min_interv = numpy.min(target1)
						target = (target1 - min_interv) * average2_multiplier
						missind = ~numpy.all(numpy.isnan(intervalnow), 1)

						if (not numpy.all(numpy.isnan(target))) and (numpy.sum(missind) != 0):
							cosh_interval, loc_ind, _ = coshifta(target, intervalnow[missind, :], 0, n,
																 fill_with_previous=fill_with_previous, block_size=block_size)
							xcs[missind, allint[i, 1]:allint[i, 2]] = cosh_interval

							xt[0, allint[i, 1]:allint[i, 2]] = target
							ind[missind, i] = loc_ind.T

						else:
							xcs[:, allint[i, 1]:allint[i, 2]] = intervalnow

			if frag:

				xn = nans(np, mp)
				for i_sam in range(0, np):
					for i_seg in range(0, in_or.shape[0]):
						xn[i_sam, in_or[i_seg, 0]:in_or[i_seg, 1]
							+ 1] = xcs[i_sam, in_or[i_seg, 2]:in_or[i_seg, 3] + 1]
						if loc_ind[i_sam] < 0:
							if flag_nan[i_seg, 0, i_sam]:
								xn[i_sam, in_or[i_seg, 0]:in_or[i_seg, 0]
									- loc_ind[i_sam, 0] + 1] = numpy.nan
						else:
							if loc_ind[i_sam] > 0:
								if flag_nan[i_seg, 1, i_sam]:
									xn[i_sam, (in_or[i_seg, 1] - loc_ind[i_sam, 0] + 1):in_or[i_seg, 1]+1] = numpy.nan


				xcs = xn
			target = xt

			if flag_coshift:
				ind = ind + wint * numpy.ones( (1, ind.shape[1]) )

			return xcs, ints, ind, target


		def coshifta(xt, xp, ref_w=0, n=numpy.array([1, 2, 3]), fill_with_previous=True, block_size=(2 ** 25)):

			if ref_w == 0 or ref_w.shape[0] == 0:
				ref_w = numpy.array([0])

			if numpy.all(ref_w >= 0):
				rw = max(ref_w.shape)

			else:
				rw = 1

			if fill_with_previous:
				filling = -numpy.inf

			else:
				filling = numpy.nan

			if xt == 'average':
				xt = np.nanmean(xp, axis=0)

			# Make two dimensional
			xt = xt.reshape(1, -1)

			nt, mt = xt.shape
			np, mp = xp.shape

			if len(ref_w.shape) > 1:
				nr, mr = ref_w.shape
			else:
				nr, mr = ref_w.shape[0], 0

			logging.info('mt=%d, mp=%d' % (mt, mp))

			if mt != mp:
				raise(Exception, 'Target "xt" and sample "xp" must be of compatible size (%d, %d)' % (mt, mp) )

			try:
				if numpy.any(n <= 0):
					raise(Exception, 'shift(s) "n" must be larger than zero')
			except: 
				pass

			if nr != 1:
				raise(Exception, 'Reference windows "ref_w" must be either a single vector or 0')

			if rw > 1 and (numpy.min(ref_w) < 1) or (numpy.max(ref_w) > mt):
				raise(Exception, 'Reference window "ref_w" must be a subset of xp')

			if nt != 1:
				raise(Exception, 'Target "xt" must be a single row spectrum/chromatogram')

			auto = 0
			if n == 'b':
				auto = 1
				if rw != 1:
					n = int(0.05 * mr)
					n = 10 if n < 10 else n
					src_step = int(mr * 0.05)
				else:
					n = int(0.05 * mp)
					n = 10 if n < 10 else n
					src_step = int(mp * 0.05)
				try_last = 0

			elif n == 'f':

				auto = 1
				if rw != 1:
					n = mr - 1
					src_step = numpy.round(mr / 2) - 1
				else:
					n = mp - 1
					src_step = numpy.round(mp / 2) - 1
				try_last = 1

			if nt != 1:
				raise(Exception, 'ERROR: Target "xt" must be a single row spectrum/chromatogram')

			xw = nans(np, mp)
			ind = numpy.zeros((1, np))

			n_blocks = int(numpy.ceil(sys.getsizeof(xp) / block_size))
			sam_xblock = numpy.array([int(np / n_blocks)])

			sam_xblock = sam_xblock.T

			ind_blocks = sam_xblock[numpy.ones(n_blocks, dtype=bool)]
			ind_blocks[0:int(np % sam_xblock)] = sam_xblock + 1
			ind_blocks = numpy.array([0, numpy.cumsum(ind_blocks, 0)]).flatten()

			if auto == 1:
				while auto == 1:
					if filling == -numpy.inf:
						xtemp = cat(1, numpy.tile(xp[:, :1], (int(1.), n)),
									xp, numpy.tile(xp[:, -1:, ], (int(1.), n)))

					elif numpy.isnan(filling):
						# FIXME
						xtemp = cat(1,
							nans(np, n), xp, nans(np, n))

					if rw == 1:
						ref_w = numpy.arange(0, mp) #.reshape(1,-1)

					ind = nans(np, 1)
					r = False


					for i_block in range(0, n_blocks):
						block_indices = range(
							ind_blocks[i_block], int(ind_blocks[i_block + 1]))

						_, ind[block_indices], ri = cc_fft_shift(xt[0, ref_w].reshape(1,-1), xp[block_indices, :][:, ref_w],
																 numpy.array([-n, n, 2, 1, filling]) )

						if not r:
							r = numpy.empty((0, ri.shape[1]))
						r = cat(0, r, ri).T

					temp_index = range(-n, n+1)

					for i_sam in range(0, np):
						index = numpy.flatnonzero(temp_index == ind[i_sam])
						xw[i_sam, :] = xtemp[int(i_sam), int(index):int(index + mp)]

					if (numpy.max(abs(ind)) == n) and try_last != 1:
						if n + src_step >= ref_w.shape[0]:
							try_last = 1
							continue
						n += src_step
						continue

					else:
						if (numpy.max(abs(ind)) < n) and n + src_step < len(ref_w) and try_last != 1:
							n += src_step
							try_last = 1
							continue
						else:
							auto = 0
							logging.info('Best shift allowed for this interval = %d' % n)

			else:
				if filling == -numpy.inf:
					xtemp = cat(1, numpy.tile(xp[:, :1], (1., n)),
								xp, numpy.tile(xp[:, -1:, ], (1., n)))

				elif numpy.isnan(filling):
					xtemp = cat(1,
						nans(np, n), xp, nans(np, n))


				if rw == 1:
					ref_w = numpy.arange(0, mp) #.reshape(1,-1)

				ind = nans(np, 1)
				r = numpy.array([])
				for i_block in range(n_blocks):

					block_indices = range(ind_blocks[i_block], ind_blocks[i_block + 1])
					dummy, ind[block_indices], ri = cc_fft_shift(xt[0, ref_w].reshape(1,-1), xp[block_indices, :][:, ref_w],
																 numpy.array([-n, n, 2, 1, filling]))
					r = cat(0, r, ri)

				temp_index = numpy.arange(-n, n+1)

				for i_sam in range(0, np):
					index = numpy.flatnonzero(temp_index == ind[i_sam])
					xw[i_sam, :] = xtemp[i_sam, index:index + mp]

				if numpy.max(abs(ind)) == n:
					logging.warn('Scrolling window size "n" may not be enough wide because extreme limit has been reached')

			return xw, ind, r


		def defints(xp, interv):
			np, mp = xp.shape
			sizechk = mp / interv - round(mp / interv)
			plus = (mp / interv - round(mp / interv)) * interv
			logging.warn('The last interval will not fulfill the selected intervals size "inter" = %f' % interv)

			if plus >= 0:
				logging.warn('Size of the last interval = %d ' % plus)
			else:
				logging.warn('Size of the last interval = %d' % (interv + plus))

			if sizechk != 0:
				logging.info('The last interval will not fulfill the selected intervals size "inter"=%f.' % interv)
				logging.info('Size of the last interval = %f ' % plus)

			t = cat(1, range(0, (mp + 1), interv), mp)
			if t[-2] == t[-2]:
				t[-2] = numpy.array([])

			t = cat(0, t[0: - 1] + 1, t[1:])
			inter = t[:].T
			return inter


		def cc_fft_shift(t, x=False, options=numpy.array([])):

			dim_x = numpy.array(x.shape)
			dim_t = numpy.array(t.shape)

			options_default = numpy.array([-numpy.fix(dim_t[-1] * 0.5), numpy.fix(dim_t[-1] * 0.5), len(t.shape) - 1, 1, numpy.nan])
			options = numpy.array([options[oi] if oi < len(options) else d for oi, d in enumerate(options_default)])
			options[numpy.isnan(options)] = options_default[numpy.isnan(options)]

			if options[0] > options[1]:
				raise(Exception, 'Lower bound for shift is larger than upper bound')

			time_dim = int(options[2] - 1)

			if dim_x[time_dim] != dim_t[time_dim]:
				raise(Exception, 'Target and signals do not have compatible dimensions')

			ord_ = numpy.array(
				[time_dim] +
				list(range(1, time_dim)) +
				list(range(time_dim, len(x.shape) - 1)) +
				[0]
			).T

			x_fft = numpy.transpose(x, ord_)  # permute
			x_fft = numpy.reshape(x_fft, (dim_x[time_dim], numpy.prod(dim_x[ord_[1:]])))

			# FIXME? Sparse/dense switchg
			p = numpy.arange(0, numpy.prod(dim_x[ ord_[1:] ] ) )
			s = numpy.max(p) + 1
			b = sp.sparse.dia_matrix( (1.0/numpy.sqrt(numpy.nansum(x_fft ** 2, axis=0)), [0]), shape=(s,s) ).todense()
			x_fft = numpy.dot(x_fft, b)

			t = numpy.transpose(t, ord_)
			t = numpy.reshape(t, (dim_t[time_dim], numpy.prod(dim_t[ord_[1:]])))
			t = normalise(t)

			np, mp = x_fft.shape
			nt = t.shape[0]
			flag_miss = numpy.any(numpy.isnan(x_fft)) or numpy.any(numpy.isnan(t))

			if flag_miss:

				if len(x.shape) > 2:
					raise(Exception, 'Multidimensional handling of missing not implemented, yet')
				miss_off = nans(1, mp)

				for i_signal in range(0, mp):

					limits = remove_nan(
						numpy.array([0, np - 1]).reshape(1, -1), x_fft[:, i_signal].reshape(1, -1), numpy.all)
					print(limits)
					if limits.shape != (2, 1):
						raise(Exception, 'Missing values can be handled only if leading or trailing')

					if numpy.any(cat(1, limits[0], mp - limits[1]) > numpy.max(abs(options[0:2]))):
						raise(Exception, 'Missing values band larger than largest admitted shift')

					miss_off[i_signal] = limits[0]

					if numpy.any(miss_off[i_signal-1] > 1):
						x_fft[0:limits[1] - limits[0] + 1,
							  i_signal] = x_fft[limits[0]:limits[1], i_signal]

					if limits[1] < np:
						x_fft[(limits[1] - limits[0] + 1):np, i_signal] = 0

				limits = remove_nan(numpy.array([0, nt - 1]), t.T, numpy.all)
				t[0:limits[1] - limits[0] + 1, :] = t[limits[0]:limits[1],:]
				t[limits[1] - limits[0] + 1:np, :] = 0
				miss_off = miss_off[0:mp] - limits[0]

			x_fft = cat(0, x_fft, numpy.zeros(
				(int(numpy.max(numpy.abs(options[0:2]))), int(numpy.prod(dim_x[int(ord_[1:])], axis=0)))
			))

			t = cat(0, t, numpy.zeros(
					(int(numpy.max(numpy.abs(options[0:2]))), int(numpy.prod(dim_t[ord_[1:]], axis=0)))
					))

			len_fft = max(x_fft.shape[0], t.shape[0])
			shift = numpy.arange(options[0], options[1] + 1)

			if (options[0] < 0) and (options[1] > 0):
				ind = list(range(int(len_fft + options[0]), int(len_fft))) + \
					list(range(0,  int(options[1] + 1)))

			elif (options[0] < 0) and (options[1] < 0):
				ind = range(len_fft + options[0], (len_fft + options[1] + 1))

			elif (options[0] < 0) and (options[1] == 0):
				ind = range(int(len_fft + options[0]),
							int(len_fft + options[1] + 1)) + [1]

			else:
				# ind = Options(1) + 1:Options(2) + 1
				ind = range(int(options[0]), int(options[1] + 1))

			# Pad to next ^2 for performance on the FFT
			fft_pad = int( 2**numpy.ceil( numpy.log2(len_fft) ) )

			x_fft = numpy.fft.fft(x_fft, fft_pad, axis=0)
			t_fft = numpy.fft.fft(t, fft_pad, axis=0)
			t_fft = numpy.conj(t_fft)
			t_fft = numpy.tile(t_fft, (1, dim_x[0]))

			dt = x_fft * t_fft
			cc = numpy.fft.ifft(dt, fft_pad, axis=0)

			if len(ord_[1:-1]) == 0:
				k = 1
			else:
				k = numpy.prod(dim_x[ord_[1:-1]])

			cc = numpy.reshape(cc[ind, :], ( int(options[1]-options[0]+1), int(k), int(dim_x[0]))  )

			if options[3] == 0:
				cc = numpy.squeeze(numpy.mean(cc, axis=1))
			else:
				if options[3] == 1:
					cc = numpy.squeeze(numpy.prod(cc, axis=1))
				else:
					raise(Exception, 'Invalid options for correlation of multivariate signals')

			pos = cc.argmax(axis=0)
			values = cat(1, numpy.reshape(shift, (len(shift), 1)), cc)
			shift = shift[pos]

			if flag_miss:
				shift = shift + miss_off

			x_warp = nans(*[dim_x[0]] + list(dim_t[1:]))
			ind = numpy.tile(numpy.nan, (len(x.shape), 18))
			indw = ind

			time_dim = numpy.array([time_dim])

			for i_X in range(0, dim_x[0]):
				ind_c = i_X

				if shift[i_X] >= 0:

					ind = numpy.arange(shift[i_X], dim_x[time_dim]).reshape(1, -1)
					indw = numpy.arange(0, dim_x[time_dim] - shift[i_X]).reshape(1, -1)

					if options[4] == - numpy.inf:
						o = numpy.zeros(int(abs(shift[int(i_X)]))).astype(int)
						if len(o) > 0:

							ind = cat(1,
									  ind,
									  numpy.array(dim_x[time_dim[o]] - 1).reshape(1, -1)
									  )

							indw = cat(1,
									   indw,
									   numpy.arange(dim_x[time_dim] - shift[i_X],
												 dim_x[time_dim]).reshape(1, -1)
									   )
				elif shift[i_X] < 0:

					ind = numpy.arange(0, dim_x[time_dim] + shift[i_X]).reshape(1, -1)
					indw = numpy.arange(-shift[i_X], dim_x[time_dim]).reshape(1, -1)

					if options[4] == - numpy.inf:

						ind = cat(1, numpy.zeros((1, int(-shift[i_X]))), ind)
						indw = cat( 1, numpy.arange(0, -shift[i_X]).reshape(1, -1), indw)

				x_warp[ind_c, indw.astype(int)] = x[ind_c, ind.astype(int)]

			shift = numpy.reshape(shift, (len(shift), 1))

			return x_warp, shift, values


		def remove_nan(b, signal, select=numpy.any, flags=False):
			'''
			Rearrange segments so that they do not include nan's
			[Bn] = remove_nan(b,  signal,  select)
			[an, flag]
			INPUT
			b     : (p * 2) Boundary matrix (i.e. [Seg_start(1) Seg_end(1); Seg_start(2) Seg_end(2);...]
			signal: (n * 2) Matrix of signals (with signals on rows)
			select: (1 * 1) function handle to selecting operator
							 e.g. numpy.any (default) eliminate a column from signal matrix
												 if one or more elements are missing
								  numpy.all           eliminate a column from signal matrix
												 if all elements are missing
			OUTPUT
			Bn  : (q * 2)     new Boundary matrix in which nan's are removed
			flag: (q * 2 * n) flag matrix if there are nan before (column 1) or after (column 2)
							   the corresponding segment in the n signals.
			Author: Giorgio Tomasi
					Giorgio.Tomasi@gmail.com
			Created      : 25 February,  2009
			Last modified: 23 March,  2009; 18:02
			Python implementation: Martin Fitzpatrick
								   martin.fitzpatrick@gmail.com
			Last modified: 28th October,  2013
			HISTORY
			1.00.00 09 Mar 09 -> First working version
			2.00.00 23 Mar 09 -> Added output for adjacent nan's in signals
			2.01.00 23 Mar 09 -> Added select input parameter
			'''
			c = nans(b.shape[0], b.shape[1] if len(b.shape) > 1 else 1)
			b = b.reshape(1, -1)
			count = 0
			signal = numpy.isnan(signal)
			for i_el in range(0, b.shape[0]):

				ind = numpy.arange(b[i_el, 0], b[i_el, 1] + 1)
				in_ = select(signal[:, ind], axis=0)

				if numpy.any(in_):

					p = numpy.diff(numpy.array([0] + in_).reshape(1, -1), 1, axis=1)
					a = numpy.flatnonzero(p < 0) + 1
					b = numpy.flatnonzero(p > 0)

					if numpy.any(~in_[0]):
						a = cat(1, numpy.array([[0]]), np.array([a]) ) 

					else:
						b = b[1:]

					if numpy.any(~in_[-1]):
						b = cat(1, numpy.array([b]), numpy.array([[max(ind.shape) - 1]]))

					a = numpy.unique(a)
					b = numpy.unique(b)
	
					try:
						d = ind[cat(0, a, b)]
					except:
						d = np.array([])

					c.resize(d.shape)
					try:
						c[count:count + max(a.shape) + 1] = d
					except:
						c = d
					count = count + max(a.shape)

				else:
					c[count, :] = b[i_el,:]
					count += 1

			c = c.astype(int).T
			an = c

			if flags:
				flag = numpy.empty((c.shape[0], 2, signal.shape[0]), dtype=bool)

				flag[:] = False
				c_inds = c[:] > 1
				c_inds = c_inds.astype(bool)

				c_inde = c[:] < signal.shape[1]
				c_inde = c_inde.astype(bool)
				flag[c_inds, 0, :] = signal[:, c[c_inds] - 1].T

				flag[c_inde, 1, :] = signal[:, c[c_inde] - 1].T
				return an, flag
			else:
				return an


		def normalise(x, flag=False):
			'''
			Column-wise normalise matrix
			nan's are ignored
			[xn] = normalise(x,  flag)
			INPUT
			x   : Marix
			flag: true if any NaNs are present (optional - it saves time for large matrices)
			OUTPUT
			xn: Column-wise normalised matrix
			Author: Giorgio Tomasi
					 Giorgio.Tomasi@gmail.com
			Created      : 09 March,  2009; 13:18
			Last modified: 09 March,  2009; 13:50
			Python implementation: Martin Fitzpatrick
								   martin.fitzpatrick@gmail.com
			Last modified: 28th October,  2013
			'''

			if not flag:
				p_att = ~numpy.isnan(x)
				flag = numpy.any(~p_att[:])

			else:
				p_att = ~numpy.isnan(x)

			m, n = x.shape
			xn = nans(m, n)
			if flag:
				for i_n in range(0, n):
					n = numpy.linalg.norm(x[p_att[:, i_n], i_n])
					if not n:
						n = 1
					xn[p_att[:, i_n], i_n] = x[p_att[:, i_n], i_n] / n

			else:
				for i_n in range(0, n):
					n = numpy.linalg.norm(x[:, i_n])
					if not n:
						n = 1
					xn[:, i_n] = x[:, i_n] / n

			return xn


		def extract_segments(x, segments):
			'''
			Extract segments from signals
			[xseg] = extract_segments(x,  segments)
			? [xseg, segnew] = extract_segments(x,  segments)
			INPUT
			x       : (n * p) data matrix
			segments: (s * 2) segment boundary matrix
			OUTPUT
			xseg: (n * q) data matrix in which segments have been removed
			segnew: New segment layout
			Author: Giorgio Tomasi
					 Giorgio.Tomasi@gmail.com
			Python implementation: Martin Fitzpatrick
								   martin.fitzpatrick@gmail.com
			Last modified: 28th October,  2013
			Created      : 23 March,  2009; 07:51
			Last modified: 23 March,  2009; 15:07
			HISTORY
			0.00.01 23 Mar 09 -> Generated function with blank help
			1.00.00 23 Mar 09 -> First working version
			'''
			n, p = x.shape
			Sd = numpy.diff(segments, axis=1)

			q = numpy.sum(Sd + 1)
			s, t = segments.shape

			flag_si = t != 2
			flag_in = numpy.any(segments[:] != numpy.fix(segments[:]))
			flag_ob = numpy.any(segments[:, 0] < 1) or numpy.any(segments[:, 1] > p)
			flag_ni = numpy.any(numpy.diff(segments[:, 0]) < 0) or numpy.any(
				numpy.diff(segments[:, 1]) < 0)
			flag_ab = numpy.any(Sd < 2)

			if flag_si:
				raise(Exception, 'Segment boundary matrix must have two columns')

			if flag_in:
				raise(Exception, 'Segment boundaries must be integers')

			if flag_ob:
				raise(Exception, 'Segment boundaries outside of segment')

			if flag_ni:
				raise(Exception, 'segments boundaries must be monotonically increasing')

			if flag_ab:
				raise(Exception, 'segments must be at least two points long')

			xseg = nans(n, q)
			origin = 0
			segnew = []

			for seg in segments:
				data = x[:, seg[0]:seg[1] + 1]
				segment_size = data.shape[1]
				xseg[:, origin:origin + segment_size] = data

				segnew.append([origin, origin + segment_size - 1])
				origin = origin + segment_size

			segnew = numpy.array(segnew)

			return xseg, segnew


		def find_nearest(array, value):
			idx = (numpy.abs(array-value)).argmin()
			return array[idx], idx

		def scal2pts(ppmi,  ppm=[],  prec=None):
			"""
			Transforms scalars into data points
			pts = scal2pts(values, scal)
			INPUT
			values: scalars whose position is sought
			scal  : vector scalars
			prec  : precision (optional) to handle endpoints
			OUTPUT
			pts   : position of the requested scalars (nan if it is outside of 'scal')
			Author: Giorgio Tomasi
					Giorgio.Tomasi@gmail.com
			Created      : 12 February,  2009; 17:43
			Last modified: 11 March,  2009; 15:14
			Python implementation: Martin Fitzpatrick
								   martin.fitzpatrick@gmail.com
			Last modified: 28th October,  2013
			HISTORY
			1.00.00 12 Feb 09 -> First working version
			1.01.00 11 Mar 09 -> Added input parameter check
			"""
			rev = ppm[0] > ppm[1]

			if prec is None:
				prec = min(abs(unique(numpy.diff(ppm))))

			pts = []
			for i in ppmi:
				nearest_v, idx = find_nearest(ppm, i)
				if abs(nearest_v-i) > prec:
					pts.append(numpy.nan)
				else:
					pts.append( idx )

			return numpy.array(pts)



		def dscal2dpts(d, ppm, prec=None):
			"""
			Translates an interval width from scal to the best approximation in sampling points.
			i = dppm2dpts(delta, scal, prec)
			INPUT
			delta: interval width in scale units
			scal : scale
			prec : precision on the scal axes
			OUTPUT
			i: interval widths in sampling points
			Author: Giorgio Tomasi
					Giorgio.Tomasi@gmail.com
			Last modified: 21st February,  2009
			Python implementation: Martin Fitzpatrick
								   martin.fitzpatrick@gmail.com
			Last modified: 28th October,  2013
			"""
			if d == 0:
				return 0

			if d <= 0:
				raise(Exception, 'delta must be positive')

			if ppm[0] < ppm[1]: # Scale in order
				i = scal2pts(numpy.array([ppm[0] + d]), ppm, prec) -1

			else:
				i = max(ppm.shape) - scal2pts(numpy.array([ppm[-1] + d]), ppm, prec) +1

			return i[0]

		xCS, ints, ind, target = icoshift(xt, X,  inter='whole',  n='f', scale=None, coshift_preprocessing=False,
					 coshift_preprocessing_max_shift=None, fill_with_previous=True, average2_multiplier=3)

		return xCS, ints, ind, target

	def test_(self, N=50, seed=1, mu_sd=200, sd_sd=50):
		print('Checking libraries... ok')
		np.random.seed(seed=seed)
		N = N
		parameters = np.random.rand( N,2 ) * [[mu_sd, sd_sd]] + [[400, 20]]
		x = np.arange(1000)
		data = np.array([ np.e**(-((x-n[0])/n[1])**2) for n in parameters])
		print('Generating test samples... ok')
		xCS, ints, ind, target = ico.aling_ICOSHIFT_2D(xt='sample', X=data)
		plt.figure(1), plt.plot(data.T)
		plt.title('Original Data')
		plt.figure(2), plt.plot(target.T)
		plt.title('Target vector')
		plt.figure(3), plt.plot(xCS.T-data.T)
		plt.title('Displasement vector')
		plt.figure(4), plt.plot(xCS.T)
		plt.title('Aling data')

		print('Shift reference value || Estimated shifting value')
		score = 0
		for i, n in enumerate(ind):
			score += (parameters[i,0]-parameters[0,0] - ind[i][0])**2
			print('S {}\t \t O {:3.1f}\t\t E {:3.1f}'.format(i, parameters[i,0]-parameters[0,0], ind[i][0]))
		
		print('mean error: {}'.format( score**0.5/N ) )
		print('Normalized mean error: {}'.format( score**0.5/N/mu_sd ) )
		plt.show()

# ********************************************* # ********************************************* # ********************************************* # *********************************************
# *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	***
# ********************************************* # ********************************************* # ********************************************* # *********************************************
# ********************************************* # ********************************************* # ********************************************* # *********************************************
# *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	***
# ********************************************* # ********************************************* # ********************************************* # *********************************************
# ********************************************* # ********************************************* # ********************************************* # *********************************************
# *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	***
# ********************************************* # ********************************************* # ********************************************* # *********************************************
# ********************************************* # ********************************************* # ********************************************* # *********************************************
# *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	*** # *** 		MCR MCR MCR MCR MCR 		 	***
# ********************************************* # ********************************************* # ********************************************* # *********************************************
	def unfold_3D(self, D=None, S=None, C=None, N=None, f=None, Na=None, Ni=None, method='mean', v=0):
		if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
		if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if type(D) == type(None):	D = self.D
		elif v: pass
		if type(S) == type(None):	S = self.S
		elif v: pass
		if type(C) == type(None):	C = self.C
		elif v: pass
		if type(f) == type(None): 	f = S.shape[1]
		elif v: pass
		
		if S.shape[1] != f and v: 	print('WARNNING :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: Number of considered factors could be NOT correct')
		# falta implementar method PARAFAC/PCA/etc

		N,l1,l2,l3 = D.shape
		Se = [ S[:,n].reshape( (l1,l2) ) for n in range(f) ] 
		Ce = C.reshape( (N,l3,f) )

		if v: print('(Uf-0) Unfold prosses ... DONE ')
		return Se, Ce

	def unfold_2D(self, D=None, S=None, C=None, N=None, f=None, Na=None, Ni=None, method='mean', v=0):
		if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
		if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

		if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get f({}) or self.f({})'.format(f, self.f))

		if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
		elif v: print('ERROR :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: can NOT get f({}) or self.f({})'.format(f, self.f))
		
		if type(D) == type(None):	D = self.D
		elif v: pass
		if type(S) == type(None):	S = self.S
		elif v: pass
		if type(C) == type(None):	C = self.C
		elif v: pass
		if type(f) == type(None): 	f = S.shape[1]
		elif v: pass
		
		if S.shape[1] != f and v: 	print('WARNNING :: code 050 :: MCR_ICOSHIFT.unfold_3D() :: Number of considered factors could be NOT correct')
		# falta implementar method PARAFAC/PCA/etc

		Se = s
		Ce = C.reshape( (N,l3,3) )

		if v: print('(Uf-0) Unfold prosses ... DONE ')
		return Se, Ce

	def train(self, Dc=None, S=None, constraints={'Normalization':['S'], 'non-negativity':['all']}, 
					inicializacion='random', max_iter=200, save=True, v=0):
		# MCR cannon. Datos de tercer  orden. 2 canales NO alineados y 1 canal alineado. 
		# D = [Nc, l1, l2] ; l1 : alineados  	l2 l3 : NO alineados
		# 	(1) D			:	MAT 	: Matrix con todos los datos. 										X = [l1, l2]
		# 	(2) S			:	MAT 	: vector con las estimaciones de los espectros - canal alineado. 	S = [f, l1]
		# 	(3) max_iter	:	INT 	: Numero maximo de iteraciones

		# ---------------------------------------- #
		# ----- (0) Check tensor integrity 	  ---- #
		# ---------------------------------------- #
		Se = None
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

		# -------------------------------------------------- #
		# ----- (1) initial stimation of matrix S and C ---- #
		# -------------------------------------------------- #
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

		# -------------------------------------------------- #
		# ----- (2) alternating least squares (MCR-ALS) ---- #
		# -------------------------------------------------- #
		for i in range(max_iter):
			# ---- Zn ---- Z1 = C | Z2 = S		------> en este caso particular no requiere operaciones adicionales
			# ---- Yf ---- Y1 = D | Y2 = D.T 	------> en este caso particular no requiere operaciones adicionales
			S=np.dot(Dc, np.linalg.pinv(C).T);	
			S, C = self.restrictions(S, C, constraints) # restrictions are apply here 
			
			C=np.dot(Dc.T, np.linalg.pinv(S).T)
			S, C = self.restrictions(S, C, constraints) # restrictions are apply here 

		if type(Se) == type(None): 	
			if type(S) == type(None): 	
				if v: print('ERROR :: MCR_ICOSHIFT.train() :: Null extended tensor :: Dc = None')
				else: print('ERROR :: MCR_ICOSHIFT.train()')
			elif len(self.D.shape) == 3:
				Se, Ce = self.unfold_2D( S=S, C=C, v=v )
			elif len(self.D.shape) == 4:
				Se, Ce = self.unfold_3D( S=S, C=C, v=v )

		# -------------------------------------------------- #
		# ----- (3) SAVE actual trainning in class MCR  ---- #
		# -------------------------------------------------- #
		if save:	# --- SAVE actual trainning in class MCR --- #
			self.S = S
			self.Se = Se
			self.C = C
			self.Ce = Ce
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
					print('Cal{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
														(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )
				else:
					print( self.A )
					print('Test{} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f} \t\t\t {:<.2f}'.format(n+1, self.Y[n, a], self.y[a, n],  
														(self.Y[n, a]-self.y[a, n]),  (self.Y[n, a]-self.y[a, n])*100/self.Y[n, a] ) )

		return None


