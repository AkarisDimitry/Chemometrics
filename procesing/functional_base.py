# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== IMPORT libraries ====================  IMPORT libraries ==================== # # ==================== IMPORT libraries ====================  IMPORT libraries ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# *** warning supresion
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries *** #
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

# *** graph libraries *** #
try:
	import matplotlib.pyplot as plt
	from mpl_toolkits import mplot3d
	import matplotlib as mpl
except: 
	print('WARNNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: ( pip3 install matplotlib )')

# *** python common libraries
import itertools

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== Obj  ====================  Obj  ==================== # # ==================== Obj ====================  Obj ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #

class BASE(object): # generador de datos
	def __init__(self, 	functional=None, base=None, data=None,
						dimention=None, complete=None,
						alpha=None, beta=None, mu=None, sigma=None):
		self.base_set	    	= base
		self.base_parameters	= None
		self.base_coeficients	= None

		self.data = data 

		self.functional 	= functional
		self.dimention 		= dimention
		self.complete 		= complete

		self.alpha  = alpha
		self.beta   = beta 
		self.mu  	= mu
		self.sigma  = sigma

		self.functional_list = ['Gaussian', 'SGaussian', 'sgaussian', 'Lorentz', 'salorentz', 'alorentz', 'Rayleigh', 'GIG']

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== Functions ================ Functions ==================== # # =================== Functions ==================  Functions  ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
	def entropy(self, probs):
	    # quantifies the average amount of surprise
	    p = np.full(probs.shape, 0.0)
	    np.log2(probs, out=p, where=(probs > 0))
	    return -((p * probs).sum())

	def relative_entropy(self, probs1, probs2):
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

	def gaussian(self, mu, sigma, n, norm=True):
		# return Gaussian vector with n dimension G(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mean value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# ---------------------------------------------------- #
		X = np.arange(n) if type(n) == int else n
		f = np.e**(-1.0/2 * ((mu-X)/sigma)**2)
		return f/np.linalg.norm(f) if norm else f

	def Sgaussian(self, mu, sigma, n, scale):
		# return Gaussian vector with n dimension G(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mean value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# scale		:	FLOAT 	:   scale value of the max value
		# ---------------------------------------------------- #
		X = np.arange(n) if type(n) == int else n
		f = np.e**(-1.0/2 * ((mu-X)/sigma)**2)
		return f*scale

	def Lorentz(self, n, a, m): 
		# return Lorentz vector with n dimension L(mu, sigma) 
		# simetric lorentz function
		# ---------------------------------------------------- #
		# n 		: 	INT 	: 	vector dimension 
		# m 		: 	FLOAT 	:	mean value
		# a 		: 	FLOAT 	:  	standard deviation 
		# ---------------------------------------------------- #
		X = np.arange(n) if type(n) == int else n
		return float(a)/(np.pi*((X-m))**2+a**2)

	def ALorentz(self, n, a, m, b):
		# return Lorentz vector with n dimension L(mu, sigma)
		# NON-simetric lorentz function 
		# ---------------------------------------------------- #
		# n 		: 	INT 	: 	vector dimension 
		# m 		: 	FLOAT 	:	mean value
		# a 		: 	FLOAT 	:  	standard deviation 
		# b 		: 	INT 	: 	asimetric coef 
		# ---------------------------------------------------- #
		L = ((a*1.0/(np.pi*((np.arange(n)-m))**2+a**2))*float(b)/(1 + np.e**(b*(np.arange(n)-m))))
		return ((a*1.0/(np.pi*((np.arange(n)-m))**2+a**2))*float(b)/(1 + np.e**(b*(np.arange(n)-m))))

	def SALorentz(self, n, a, m, b, s):
		# return Lorentz vector with n dimension L(mu, sigma) 
		# NON-simetric scaled lorentz function
		# ---------------------------------------------------- #
		# n 		: 	INT 	: 	vector dimension 
		# m 		: 	FLOAT 	:	mean value
		# a 		: 	FLOAT 	:  	standard deviation 
		# b 		: 	INT 	: 	asimetric coef 
		# s			:	FLOAT 	:   scale value of the max value
		# ---------------------------------------------------- #
		X = np.arange(n) if type(n) == int else n
		L = ((float(a)/(np.pi*((X-m))**2+a**2)) * float(b)/(1 + np.e**(-b*(X-m))))
		return L/np.max(L)*s

	def Rayleigh(self, mu, sigma, n, norm=True ):
		# return Rayleigh distribution vector with n dimension R(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mode value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# ---------------------------------------------------- #
		x = np.arange(n) - mu + sigma # x + desire_mean - actual_mean  
		f = x/sigma**2 * np.e**(-1.0/2 * (x/sigma)**2)
		f[f<0]=0
		return f/np.linalg.norm(f) if norm else f

	def GIG(self,mu, sigma, n, a=2, b=1, p=-1, norm=True):
		'''
		In probability theory and statistics, the generalized inverse Gaussian 
		distribution (GIG) is a three-parameter family of continuous probability 
		istributions with probability density function
		# return generalized inverse Gaussian distribution vector with n dimension
		 R(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mode value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# ---------------------------------------------------- #
		'''
		x = (np.arange(n))*5/n* sigma
		f = x**(p-1) * np.e**(-(a*x+b/x)/2)
		f[f<0]=0
		f = np.nan_to_num(f, copy=True, nan=0.0, posinf=None, neginf=None)

		x = (np.arange(n)+np.argmax(f)-mu)*5/n * sigma
		x[x<0] = 0

		f = x**(p-1) * np.e**(-(a*x+b/x)/2)
		f[f<0]=0
		f = np.nan_to_num(f, copy=True, nan=0.0, posinf=None, neginf=None)

		return f/np.linalg.norm(f) if norm else f

# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== Generators ================ Generators ==================== # # =================== Generators ==================  Generators  ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #

	def generate_discrete_base(self, 	functional=None, dimention=None, base_parameters=None,
										alpha=None, beta=None, mu=None, sigma=None, 
										verbosity=False, save=True):
		'''
		This function alocate and generate complete discrete functional space. 
		Generating functional space before parforming any proyection could 
		increase performance.

		generate_discrete_base()
		# ---------------------------------------------------- #
		# functional 	: 	STR 	: 	eg. Gaussian	
		# dimention 	: 	INT 	: 	indicates the dimention of the discrete vertor thar represents each function eg. 100	
		# verbosity 	: 	BOOL	: 	print some data
		# ---------------------------------------------------- #
		'''
		functional = functional if type(functional) == str else self.functional
		if not functional in self.functional_list: 
			print('WARNNING :: functional_base.generate_discrete_base() :: Can not identify the function. allowed functional_list just include : {0}'.format(self.functional_list) )
		
		alpha 			= np.array(alpha) 			if type(alpha) 				!= type(None) else np.array(self.alpha) 
		beta 			= np.array(beta)			if type(beta) 				!= type(None) else np.array(self.beta) 
		mu 				= np.array(mu) 				if type(mu) 				!= type(None) else np.array(self.mu)  
		sigma 			= np.array(sigma) 			if type(sigma) 				!= type(None) else np.array(self.sigma)  
		base_parameters = np.array(base_parameters) if type(base_parameters) 	!= type(None) else np.array(self.base_parameters)  

		base = np.zeros( [*base_parameters.shape[:-1],dimention] )

		if functional.lower() == 'gaussian':
			for i1, d1 in enumerate(base_parameters):
				for i2, (m, s) in enumerate(d1):
					# m : mu
					# s : sigma
					base[i1, i2, :] = self.gaussian(m, s, dimention, norm=True)

		elif functional.lower() == 'sgaussian':
			for i1, d1 in enumerate(base_parameters):
				for i2, d2 in enumerate(d1):
					for i3, (m, s, a) in enumerate(d2):
						# m : mu
						# s : sigma
						# a : alpha
						base[i1, i2, i3, :] = self.Sgaussian(m, s, dimention, a)

		elif functional.lower() == 'salorentz':
			for i1, d1 in enumerate(base_parameters):
				for i2, d2 in enumerate(d1):
					for i3, d3 in enumerate(d2):
						for i4, (a, m, b, s) in enumerate(d3):
		# ---------------------------------------------------- #
		# n 		: 	INT 	: 	vector dimension 
		# m 		: 	FLOAT 	:	mean value
		# a 		: 	FLOAT 	:  	standard deviation 
		# b 		: 	INT 	: 	asimetric coef 
		# s			:	FLOAT 	:   scale value of the max value
		# ---------------------------------------------------- #
							base[i1, i2, i3, i4, :] = self.SALorentz(dimention, a, m, b, s)

		if save: 
			self.base_set 	= base

		return base

	def generate_parameters_base(self, functional=None, dimention=None, complete=False,
										alpha=None, beta=None, mu=None, sigma=None, 
										verbosity=False, save=True):
		'''
		This function alocate and generate complete discrete parameters space. 

		generate_parameters_base()
		# ---------------------------------------------------- #
		# functional 	: 	STR 	: 	eg. Gaussian	
		# dimention 	: 	INT 	: 	indicates the dimention of the discrete vertor thar represents each function eg. 100	
		# complete 		:	BOOL 	: 	Generate a complete base for a give dimentionality, this parameter ignores mu, sigma, alpha and beta indications
		# verbosity 	: 	BOOL	: 	print some data
		# ---------------------------------------------------- #
		'''
		functional = functional if type(functional) == str else self.functional
		if not functional in self.functional_list: 
			print('WARNNING :: functional_base.generate_discrete_base() :: Can not identify the function. allowed functional_list just include : {0}'.format(self.functional_list) )
		alpha 	= np.array(alpha) 	if type(alpha) 	!= type(None) else np.array(self.alpha) 
		beta 	= np.array(beta)	if type(beta) 	!= type(None) else np.array(self.beta) 
		mu 		= np.array(mu) 		if type(mu) 	!= type(None) else np.array(self.mu)  
		sigma 	= np.array(sigma) 	if type(sigma) 	!= type(None) else np.array(self.sigma)  

		if verbosity: print( 'generate_parameters_base :: alocating memory ')
		if functional.lower() == 'gaussian':
			if type(mu) == type(None) or type(sigma) == type(None):
				print('WARNNING :: functional_base.generate_discrete_base() :: In order to generate a gaussian functional space it is a requiremente a minimun set of sigma and mu values.' )
			base_parameters = np.zeros((mu.shape[0], sigma.shape[0], 2))

			for i, m in enumerate(mu):
				for j, s in enumerate(sigma): 
					# m : mu     : gaussian center
					# s : sigma  : gaussian variance 
					base_parameters[i][j][:] = np.array([m, s])

		elif functional.lower() == 'sgaussian':
			if type(mu) == type(None) or type(sigma) == type(None) or type(alpha) == type(None):
				print('WARNNING :: functional_base.generate_discrete_base() :: In order to generate a gaussian functional space it is a requiremente a minimun set of sigma and mu values.' )
			base_parameters = np.zeros((mu.shape[0], sigma.shape[0], alpha.shape[0], 3))

			for i, m in enumerate(mu):
				for j, s in enumerate(sigma): 
					for k, a in enumerate(alpha): 
						# m : mu     : gaussian center
						# s : sigma  : gaussian variance 
						# a : alpha  : gaussian max value 
						base_parameters[i][j][k][:] = np.array([m, s, a])

		elif functional.lower() == 'salorentz':
			if type(mu) == type(None) or type(sigma) == type(None) or type(alpha) == type(None):
				print('WARNNING :: functional_base.generate_discrete_base() :: In order to generate a gaussian functional space it is a requiremente a minimun set of sigma and mu values.' )
			base_parameters = np.zeros((mu.shape[0], sigma.shape[0], alpha.shape[0], beta.shape[0], 4))

			for i, m in enumerate(mu):
				for j, a in enumerate(sigma): 
					for k, s in enumerate(alpha): 
						for l, b in enumerate(beta): 
		# ---------------------------------------------------- #
		# n 		: 	INT 	: 	vector dimension 
		# m 		: 	FLOAT 	:	mean value
		# a 		: 	FLOAT 	:  	standard deviation 
		# b 		: 	INT 	: 	asimetric coef 
		# s			:	FLOAT 	:   scale value of the max value
		# ---------------------------------------------------- # (self, n, a, m, b, s)
							base_parameters[i][j][k][l][:] = np.array([a, m, b, s])


		if verbosity: print( f'generate_parameters_base :: functional base {functional} generated :: mu {mu[0]}-{mu[-1]} sigma {sigma[0]}-{sigma[-1]}')
		
		if save: 
			self.base_parameters 	= base_parameters
			self.functional 		= functional
			self.alpha 	= alpha if type(alpha) 			!= type(None) else self.alpha
			self.beta 	= beta 	if type(beta) 			!= type(None) else self.beta
			self.mu 	= mu 	if type(mu) 			!= type(None) else self.mu
			self.sigma 	= sigma if type(sigma) 			!= type(None) else self.sigma
			self.complete = complete if type(sigma) 	!= type(None) else self.sigma

		return base_parameters

	def evaluate_coeficients(self, data=None, base=None, base_parameters=None, functional=None,
									non_negativity=True, over_estimation_penalization=0,
									verbosity= False, save=True,): 
		'''
		This function evaluaten the proyection coeficients. 

		evaluate_coeficients()
		# ---------------------------------------------------- # # ---------------------------------------------------- #
		# functional 		: 	STR 	: 	eg. Gaussian	
		# data     	 		: 	array 	: 	proyection space	
		# base     	 		: 	array 	: 	functional base to evaluate coeficients 	
		# base_parameters	: 	BOOL	: 	print some data
		# non_negativity	:	BOOL 	: 	Generate a complete base for a give dimentionality, this parameter ignores mu, sigma, alpha and beta indications
		# ---------------------------------------------------- # # ---------------------------------------------------- #
		'''
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			!= type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters)!= type(None) else np.array(self.base_parameters)  
		functional      = functional 				if type(functional)	    == str        else self.functional

		if not functional in self.functional_list: 
			print('WARNNING :: functional_base.evaluate_coeficients() :: Can not identify the function. allowed functional_list just include : {0}'.format(self.functional_list) )

		if not type(data) == type(np.array([0])): 
			print('ERROR :: functional_base.evaluate_coeficients() :: Incorrect data type' )

		if not data.shape[0] == base.shape[-1]:
			print('ERROR :: functional_base.evaluate_coeficients() :: Input data.shape[0] [{0}] and base.shape[-1] [{1}] must have same shape'.format(self.data.shape, self.base.shape) )

		coeficients = np.zeros( base_parameters.shape[:-1] )
		if functional.lower() == 'gaussian':
			for i1, d1 in enumerate(base_parameters):
				for i2, (m, s) in enumerate(d1):
					coeficients[i1][i2] = np.dot( base[i1][i2], data )

		if functional.lower() == 'sgaussian':
			# lower coeficients determines better fit 
			for i1, d1 in enumerate(base_parameters):
				for i2, d2 in enumerate(d1):
					for i3, (m, s, a) in enumerate(d2):
						coeficients[i1][i2][i3] = -np.sum( np.abs(base[i1][i2][i3]-data) )
						if over_estimation_penalization > 0:
							positive_vector = base[i1][i2][i3]-data
							positive_vector[ positive_vector<0 ] = 0
							coeficients[i1][i2][i3] -= np.sum(positive_vector*data)*over_estimation_penalization


		elif functional.lower() == 'salorentz':

			# lower coeficients determines better fit 
			for i1, d1 in enumerate(base_parameters):
				for i2, d2 in enumerate(d1):
					for i3, d3 in enumerate(d2):
						for i4, (a, m, b, s) in enumerate(d3):
		# ---------------------------------------------------- #
		# n 		: 	INT 	: 	vector dimension 
		# m 		: 	FLOAT 	:	mean value
		# a 		: 	FLOAT 	:  	standard deviation 
		# b 		: 	INT 	: 	asimetric coef 
		# s			:	FLOAT 	:   scale value of the max value
		# ---------------------------------------------------- #
							coeficients[i1][i2][i3][i4] = -np.sum( np.abs(base[i1][i2][i3][i4]-data) )
							if over_estimation_penalization > 0:
								positive_vector = base[i1][i2][i3][i4]-data
								positive_vector[ positive_vector<0 ] = 0
								coeficients[i1][i2][i3][i4] -= np.sum(positive_vector*data)*over_estimation_penalization

		if save: 	
			self.coeficients 		= coeficients 		if type(coeficients) 		!= type(None) 	else self.coeficients
			self.functional 		= functional 		if type(functional) 		!= type(None) 	else self.functional
			self.non_negativity 	= non_negativity	if type(non_negativity) 	!= type(None) 	else self.non_negativity
			self.base_parameters 	= base_parameters 	if type(base_parameters)	!= type(None) 	else self.base_parameters
			self.base 				= base  			if type(base) 				!= type(None) 	else self.base
			self.data 				= data 				if type(data)				!= type(None)	else self.data

		return coeficients

	def SFR(self, data=None, non_negativity=True, 
				  base=None, base_parameters=None, functional=None,
				  verbosity=False, save=True,):
		'''
		For a given functiuonal base this function (SFR) find the best representation. 

		single_function_representacion()
		# ---------------------------------------------------- # # ---------------------------------------------------- #
		# functional 		: 	STR 	: 	eg. Gaussian	
		# data     	 		: 	array 	: 	proyection space	
		# base     	 		: 	array 	: 	functional base to evaluate coeficients 	
		# base_parameters	: 	BOOL	: 	print some data
		# non_negativity	:	BOOL 	: 	Generate a complete base for a give dimentionality, this parameter ignores mu, sigma, alpha and beta indications
		# ---------------------------------------------------- # # ---------------------------------------------------- #
		'''
		return self.single_function_representacion(	data=None, non_negativity=True, 
													base=None, base_parameters=None, functional=None,
													verbosity=False, save=True,)

	def single_function_representacion(self, data=None, non_negativity=True, over_estimation_penalization=0,
										base=None, base_parameters=None, functional=None,
										verbosity=False, save=True,):
		'''
		For a given functiuonal base this function (SFR) find the best representation. 

		single_function_representacion()
		# ---------------------------------------------------- # # ---------------------------------------------------- #
		# functional 		: 	STR 	: 	eg. Gaussian	
		# data     	 		: 	array 	: 	proyection space	
		# base     	 		: 	array 	: 	functional base to evaluate coeficients 	
		# base_parameters	: 	BOOL	: 	print some data
		# non_negativity	:	BOOL 	: 	Generate a complete base for a give dimentionality, this parameter ignores mu, sigma, alpha and beta indications
		# ---------------------------------------------------- # # ---------------------------------------------------- #
		'''
		data 			= np.array(data)			if type(data) 			 != type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			 != type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters) != type(None) else np.array(self.base_parameters)  
		functional      = functional                if type(functional)      == str        else self.functional

		# == initial Vector estimation == #
		data_estimation = np.zeros_like(data)
		remain = data - data_estimation

		# == Best function representation == #
		coeficients = self.evaluate_coeficients( data=data, functional=functional, over_estimation_penalization=over_estimation_penalization )
		max_arg = np.unravel_index( coeficients.argmax(), coeficients.shape )

		# == Functional estimation == #
		data_estimation = base[max_arg] * coeficients[max_arg]
		remain = data - data_estimation

		return {'estimation':data_estimation, 'remain':remain, 'coeficients':coeficients, 'max_arg':max_arg, 'function':base[max_arg]}

	# ======================== FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM - FPM ======================== #
	def FPM(self, 
				data=None, loss='entropy', non_negativity=True,
				base=None, base_parameters=None, functional=None, 
				iterations=100, entropy_stop_criteria=None, 
				mu=None, sigma=None,
				verbosity=False, save=True,): 

		return functional_proyection_method(data=None, loss='entropy', non_negativity=True,
											base=None, base_parameters=None, functional=None, 
											iterations=100, entropy_stop_criteria=None, 
											mu=None, sigma=None,
											verbosity=False, save=True,)

	def functional_proyection_method(self, 	
							data=None, loss='entropy', non_negativity=True,
							base=None, base_parameters=None, functional=None, 
							iterations=100, entropy_stop_criteria=None, 
							mu=None, sigma=None, over_estimation_penalization=0,
							verbosity=False, save=True,): 
		data 			= np.array(data)			if type(data) 			 != type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			 != type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters) != type(None) else np.array(self.base_parameters)  
		functional      = functional 				if type(functional)      == str        else self.functional

		# == Vector estimation == #
		data_estimation = np.zeros_like(data)
		remain = data - data_estimation
		mu    = np.arange(1, data.shape[0]) if mu    is None else mu
		sigma = np.arange(1, data.shape[0]) if sigma is None else sigma 

		# == Indicator == #
		if loss.lower() == 'entropy':	loss_change = [ self.entropy(data_estimation) ]
		else:							loss_change
		
		if verbosity: print(' == Generating functional base == ')

		self.generate_parameters_base(	functional = 'Gaussian', 
										mu   	   = mu, 
										sigma 	   = sigma,
										verbosity=verbosity )

		self.generate_discrete_base(dimention=data.shape[0])

		if save: step_data_estimation, step_remain, step_coeficients, step_max_arg, step_function, step_RMSE = [], [], [], [], [], []

		for n in range(iterations):

			SFR = self.single_function_representacion( data=remain, functional=functional,
														over_estimation_penalization=over_estimation_penalization ) # SFR
			data_estimation += SFR['estimation']
			RMSE = np.sum(np.abs(data - data_estimation))

			if save:
				step_data_estimation.append( 	SFR['estimation'] 	)
				step_remain.append( 			SFR['remain'] 		)
				step_coeficients.append( 		SFR['coeficients'] 	)
				step_max_arg.append( 			SFR['max_arg'] 		)
				step_function.append( 			SFR['function'] 	)
				step_RMSE.append( 				RMSE   )

			remain = SFR['remain']
			loss_change.append( self.entropy(data_estimation) )

			if verbosity: print( f' FPM :: iteration {n} :: loss_change {loss_change[-2]-loss_change[-1]} :: RMSE {RMSE} ' )

		if verbosity: print( f' FPM :: Convengence criteria archived or max steps after iteration {n} :: final loss {loss_change[-1]} :: RMSE {RMSE} ' )

		if save: 
			self.loss_change 			= np.array(loss_change)
			self.data_estimation 		= np.array(data_estimation)
			self.loss_change 			= np.array(loss_change)
			self.step_data_estimation 	= np.array(step_data_estimation)
			self.step_remain 			= np.array(step_remain)
			self.step_coeficients 		= np.array(step_coeficients)
			self.step_max_arg 			= np.array(step_max_arg)
			self.step_function 			= np.array(step_function)
			self.step_RMSE 				= np.array(step_RMSE)

		return { 'data_estimation': data_estimation , 
				 'loss_change'    : loss_change     , 'estimation' : step_data_estimation , 'remain'   : step_remain,
				 'coeficients'    : step_coeficients, 'max_arg'    : step_max_arg         , 'function' : step_function }

	# ======================== PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM - PEM ======================== #
	def single_peak_representacion(self, data=None, non_negativity=True, 
										 base=None, base_parameters=None, functional=None,
										 sigma=None, beta=None, over_estimation_penalization=0,
										 verbosity=False, save=True,):
		data 			= np.array(data)			if not type(data) 			 is type(None) else np.array(self.data)  
		base 			= np.array(base) 			if not type(base) 			 is type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if not type(base_parameters) is type(None) else np.array(self.base_parameters)  
		functional      = functional                if type(functional)          == str  	   else 'sgaussian'

		max_arg = data.argmax()
		max_value = data[max_arg]
		sigma = np.arange(2, data.shape[0], 0.5) if sigma is None else sigma
		beta = sigma/100 if beta is None else beta

		if verbosity: print( f' PEM :: single_peak_representacion :: Generating functional base :: mu {max_arg} sigma {sigma[0]} to {sigma[-1]}' )
		self.generate_parameters_base(	functional=functional, 
										mu=[max_arg], 
										sigma=sigma,
										alpha=[max_value],
										beta=beta, )

		self.generate_discrete_base( dimention=data.shape[0], functional=functional )	
		SPR = self.single_function_representacion( data=data, functional=functional, over_estimation_penalization=over_estimation_penalization )
		SPR['max'] = max_arg, max_value, sigma[ SPR['max_arg'][1]]

		return SPR

	def PEM(self,	data=None, loss='entropy', non_negativity=True, 
					iterations=100, entropy_stop_criteria=None,
					functional=None, precision=10, 
					sigma=None, beta=None, over_estimation_penalization=0,
					verbosity=False, save=True): # metodo de eliminacion de picos 
		# === PEAK ELIMINATION METHOD === #
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  

		# == Vector estimation == #
		#data_estimation = np.zeros_like(data)
		#remain = data - data_estimation

		# == Functional estimation (splines) == #
		if verbosity: print( f' PEM :: transform data into functional space {data.shape[0]} to {data.shape[0]*precision} :: factor 1/{precision}' )
		fsp     = interp1d(np.linspace(1, 100, data.shape[0]), data, kind='cubic')
		data_sp = fsp(np.linspace(1, 100, num=data.shape[0]*precision, endpoint=False))

		data_estimation = np.zeros_like(data_sp)
		remain = data_sp - data_estimation

				# == Indicator == #
		loss_change = [ self.entropy(data_estimation) ]

		if save: step_data_estimation, step_remain, step_coeficients, step_max_arg, step_max, step_function, step_RMSE = [], [], [], [], [], [], []

		for n in range(iterations): 
			SPR = self.single_peak_representacion(data=remain, sigma=sigma, beta=beta, functional=functional, 
													over_estimation_penalization=over_estimation_penalization, verbosity=False)

			data_estimation += SPR['function']
			remain -= SPR['function']
			RMSE = np.sum(np.abs(data_sp - data_estimation))
			loss_change.append( self.entropy(data_estimation) )

			if save:
				step_data_estimation.append( 	SPR['estimation'] 	)
				step_remain.append( 			SPR['remain'] 		)
				step_coeficients.append( 		SPR['coeficients'] 	)
				step_max_arg.append( 			SPR['max_arg'] 		)
				step_max.append(				SPR['max']	)
				step_function.append( 			SPR['function'] 	)
				step_RMSE.append( 				RMSE   )

			if verbosity: print( f' FPM :: iteration {n} :: loss_change {loss_change[-2]-loss_change[-1]} :: RMSE {RMSE} ' )

		if verbosity: print( f' FPM :: Convengence criteria archived or max steps after iteration {n} :: final loss {loss_change[-1]} :: RMSE {RMSE} ' )

		if save: 

			self.data_estimation 		= np.array(data_estimation)
			self.loss_change 			= np.array(loss_change)
			self.step_data_estimation 	= np.array(step_data_estimation)
			self.step_remain 			= np.array(step_remain)
			self.step_coeficients 		= np.array(step_coeficients)
			self.step_max_arg 			= np.array(step_max_arg)
			self.step_max 				= np.array(step_max)
			self.step_function 			= np.array(step_function)
			self.step_RMSE 				= np.array(step_RMSE)
			self.data_sp 				= data_sp

		return { 'data_estimation': data_estimation , 
				 'loss_change'    : loss_change     , 'estimation' : step_data_estimation , 'remain'   : step_remain,
				 'coeficients'    : step_coeficients, 'max_arg'    : step_max_arg         , 'function' : step_function }

	def recursive_PEM(self, ):
		pass

	def spring_PEM(self, ):
		pass
		
	def plot_PEM(self, ax=None, save=False):

		# === make figure === #
		fig = plt.figure(figsize=(10, 10), dpi=80, constrained_layout=False, facecolor='0.9')
		fig.suptitle('PEM analisys plot', size=20)
		gs = fig.add_gridspec(nrows=6, ncols=6,)#left=0.05, right=0.75, hspace=0.1, wspace=0.05)

		X = np.arange( 0, self.data_sp.shape[0])
		ax_data_components 	= fig.add_subplot(gs[:-2, 2:4])
		ax_data_components.fill_between(X, self.data_sp, alpha=0.3)
		ax_data_components.plot( self.data_sp, alpha=0.8)
		ax_data_components.plot( self.step_function.T )

		ax_data_estimation 	= fig.add_subplot(gs[:-2, :2])
		ax_data_estimation.fill_between(X, self.data_sp, alpha=0.3)
		ax_data_estimation.plot( self.data_sp )
		ax_data_estimation.fill_between(X, self.data_estimation, alpha=0.3)
		ax_data_estimation.plot( self.data_estimation )

		ax_entropy 			= fig.add_subplot(gs[-2, :])
		ax_entropy.plot( self.loss_change )

		ax_RMSE 			= fig.add_subplot(gs[-1, :])
		ax_RMSE.plot( np.linspace( 0, 1, self.step_RMSE.shape[0]), self.step_RMSE )
		
		ax_TABLE 			= fig.add_subplot(gs[:-2, 4:])
		fig.patch.set_visible(False)
		ax_TABLE.axis('off')
		ax_TABLE.axis('tight')

		table = ax_TABLE.table(cellText= [ ['{:.3f}'.format(nx) if i>0 else '{:d}'.format(int(nx)) for i, nx in enumerate(ny)] for ny in self.step_max[:20,:]], 
								loc='center', fontsize=17, colLabels=['Center','Coef','SD'], rowLabels=None)
		table.auto_set_font_size(False)
		table.scale(1.2, 1.5) 
		table.set_fontsize(15)

		#plt.tight_layout()
		plt.show( )

	def get_PEM_coef(self, ):
		return self.step_max

def example():
	def cromatograma(x, shifting, warping, scale):
		return scale * np.e**( -(x-50+shifting)**2/(10*warping) )

	x = np.linspace(0,100, 100)
	y = cromatograma(x, 0, 1, 1) + cromatograma(x, 20, 1, 1) + cromatograma(x, 28, 1, 1)
	for n in range(5):
		y += cromatograma(x, np.random.rand()*50-25, 1, 1)

	base = BASE()
	#base.functional_proyection_method(data = y, verbosity=True)
	base.PEM(data = y, verbosity=True, iterations=51, sigma=np.arange(1, 50, 1), beta=[10**int(n) for n in np.arange(-10, 10, 1)], functional='sgaussian')
	base.plot_PEM( )


#example()
'''
# === COOKBOOK === #
[0] base = BASE()

[1] base.generate_parameters_base(	functional='Gaussian', mu=[1,2,3,4], sigma=[1,2,3,4])

[2]	base.generate_discrete_base(dimention=100)

[3]	base.evaluate_coeficients()

[4]

[5]
'''