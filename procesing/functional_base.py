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

		self.functional_list = ['Gaussian', 'SGaussian', 'Lorentz', 'Rayleigh', 'GIG']

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
		f = np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)
		return f/np.linalg.norm(f) if norm else f

	def Sgaussian(self, mu, sigma, n, scale):
		# return Gaussian vector with n dimension G(mu, sigma) E R**n
		# ---------------------------------------------------- #
		# mu 		: 	FLOAT 	:	mean value
		# sigma 	: 	FLOAT 	:  	standard deviation 
		# n 		: 	INT 	: 	vector dimension 
		# scale		:	FLOAT 	:   scale value of the max value
		# ---------------------------------------------------- #
		f = np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)
		return f*scale

	def Lorentz(self,n,a,m):
		return (a*1.0/(np.pi*((np.arange(0, n, dtype=np.float32)-m))**2+a**2))

	def Rayleigh(self,mu, sigma, n, norm=True ):
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

		if functional.lower() == 'gaussian':
			if type(mu) == type(None) or type(sigma) == type(None):
				print('WARNNING :: functional_base.generate_discrete_base() :: In order to generate a gaussian functional space it is a requiremente a minimun set of sigma and mu values.' )
			base_parameters = np.zeros((mu.shape[0], sigma.shape[0], 2))

			for i, m in enumerate(mu):
				for j, s in enumerate(sigma): 
					base_parameters[i][j][:] = np.array([m, s])

		elif functional.lower() == 'sgaussian':
			if type(mu) == type(None) or type(sigma) == type(None) or type(alpha) == type(None):
				print('WARNNING :: functional_base.generate_discrete_base() :: In order to generate a gaussian functional space it is a requiremente a minimun set of sigma and mu values.' )
			base_parameters = np.zeros((mu.shape[0], sigma.shape[0], alpha.shape[0], 3))

			for i, m in enumerate(mu):
				for j, s in enumerate(sigma): 
					for k, a in enumerate(alpha): 
						base_parameters[i][j][k][:] = np.array([m, s, a])

		if save: 
			self.base_parameters 	= base_parameters
			self.functional 		= functional
			self.alpha 	= alpha if type(alpha) 	!= type(None) else self.alpha
			self.beta 	= beta 	if type(beta) 	!= type(None) else self.beta
			self.mu 	= mu 	if type(mu) 	!= type(None) else self.mu
			self.sigma 	= sigma if type(sigma) 	!= type(None) else self.sigma
			self.complete = complete if type(sigma) 	!= type(None) else self.sigma

		return base_parameters

	def evaluate_coeficients(self, data=None, base=None, base_parameters=None, functional=None,
									non_negativity=True, 
									verbosity= False, save=True,): 
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			!= type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters)!= type(None) else np.array(self.base_parameters)  
		functional = functional if type(functional) == str else self.functional

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
			for i1, d1 in enumerate(base_parameters):
				for i2, d2 in enumerate(d1):
					for i3, (m, s, a) in enumerate(d2):
						coeficients[i1][i2][i3] = -np.sum(np.abs(base[i1][i2][i3]-data))


		if save: 
			self.coeficients 		= coeficients 		if type(coeficients) 		!= type(None) 	else self.coeficients
			self.functional 		= functional 		if type(functional) 		!= type(None) 	else self.functional
			self.non_negativity 	= non_negativity	if type(non_negativity) 	!= type(None) 	else self.non_negativity
			self.base_parameters 	= base_parameters 	if type(base_parameters)	!= type(None) 	else self.base_parameters
			self.base 				= base  			if type(base) 				!= type(None) 	else self.base
			self.data 				= data 				if type(data)				!= type(None)	else self.data

		return coeficients

	def single_function_representacion(self, data=None, non_negativity=True, 
										base=None, base_parameters=None, functional=None,
										verbosity=False, save=True,):
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			!= type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters)!= type(None) else np.array(self.base_parameters)  
		functional = functional if type(functional) == str else self.functional

		# == initial Vector estimation == #
		data_estimation = np.zeros_like(data)
		remain = data - data_estimation

		# == Best function representation == #
		coeficients = self.evaluate_coeficients( data = data )

		max_arg = np.unravel_index(coeficients.argmax(), coeficients.shape)

		# == Functional estimation == #
		data_estimation = base[max_arg] * coeficients[max_arg]
		remain = data - data_estimation

		return {'estimation':data_estimation, 'remain':remain, 'coeficients':coeficients, 'max_arg':max_arg, 'function':base[max_arg]}

	def proyection(self, 	data=None, loss='entropy', non_negativity=True,
							base=None, base_parameters=None, functional=None,
							verbosity=False, save=True,): 
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			!= type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters)!= type(None) else np.array(self.base_parameters)  
		functional = functional if type(functional) == str else self.functional

		# == Vector estimation == #
		data_estimation = np.zeros_like(data)
		remain = data - data_estimation

		# == Indicator == #
		entropy_change = [ self.entropy(data_estimation) ]

		for n in range(5):
			SFR = self.single_function_representacion( data = remain ) # SFR
			data_estimation += SFR['estimation']
			remain = SFR['remain']
			entropy_change.append( self.entropy(data_estimation) )
			plt.figure(2), plt.plot(data, lw=3)
			plt.figure(2), plt.plot(data_estimation)
			plt.show()

		plt.figure(1), plt.plot(entropy_change, 'o')
		plt.show()

	def single_peak_representacion(self, data=None, non_negativity=True, 
										 base=None, base_parameters=None, functional=None,
										 verbosity=False, save=True,):
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			!= type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters)!= type(None) else np.array(self.base_parameters)  
		functional = functional if type(functional) == str else self.functional

		# == initial Vector estimation == #
		#data_estimation = np.zeros_like(data)
		#remain = data - data_estimation

		max_arg = data.argmax()
		max_value = data[max_arg]

		self.generate_parameters_base(	functional='SGaussian', 
										mu=[max_arg], 
										sigma=np.arange(2,30),
										alpha=[max_value] )

		self.generate_discrete_base(dimention=data.shape[0], )	
		SPR = self.single_function_representacion( data = data )

		return SPR

	def PEM(self,	data=None, loss='entropy', non_negativity=True, interpolation=True,
					max_peaks=None, min_entropy_change=None,
					base=None, base_parameters=None, functional=None,
					verbosity=False, save=True): # metodo de eliminacion de picos 
		# === PEAK ELIMINATION METHOD === #
		data 			= np.array(data)			if type(data) 			!= type(None) else np.array(self.data)  
		base 			= np.array(base) 			if type(base) 			!= type(None) else np.array(self.base_set)  
		base_parameters = np.array(base_parameters) if type(base_parameters)!= type(None) else np.array(self.base_parameters)  
		functional = functional if type(functional) == str else self.functional
	
		# == Vector estimation == #
		#data_estimation = np.zeros_like(data)
		#remain = data - data_estimation

		# == Functional estimation (splines) == #
		fsp = interp1d(np.linspace(1, data.shape[0], 100), data, kind='cubic')
		data_sp = fsp(np.linspace(1, data.shape[0], num=data.shape[0]*5, endpoint=False))

		data_estimation = np.zeros_like(data_sp)
		remain = data_sp - data_estimation

		# == Indicator == #
		entropy_change = [ self.entropy(data_estimation) ]

		for n in range(5): 
			SPR = self.single_peak_representacion(data=remain)
			data_estimation += SPR['function']
			remain -= SPR['function']
			entropy_change.append(self.entropy(data_estimation))
			plt.figure(2), plt.plot(data_sp, lw=3)
			plt.figure(2), plt.plot(data_estimation)
			plt.show()
		
		plt.plot(entropy_change, 'o')
		plt.show()



def example():
	def cromatograma(x, shifting, warping, scale):
		return scale * np.e**( -(x-50+shifting)**2/(10*warping) ) 
	x = np.linspace(0,100, 100)
	y = cromatograma(x, 0, 1, 1) + cromatograma(x, 20, 1, 1) + cromatograma(x, 28, 1, 1)
	for n in range(5):
		y += cromatograma(x, np.random.rand()*50-25, 1, 1)
	base = BASE()
	base.generate_parameters_base(	functional='Gaussian', 
									mu=np.arange(2,100), 
									sigma=np.arange(2,10))

	base.generate_discrete_base(dimention=100)

	base.proyection(data = y)

	base.PEM(data = y)

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