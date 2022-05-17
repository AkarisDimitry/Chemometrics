# *** numeric libraries
import numpy as np #pip3 install numpy

# *** graph libraries
try:
	import matplotlib.pyplot as plt
except: 
	print('WARNNING :: PARAFAC.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by "pip3 install matplotlib" ')

class COW(object):
	def __init__(self, X=None, D=None, L=None, La=None, f=None, A=None, a=1,
				Y=None, y=None, S=None,	N=None, Nc=None, Na=None, Ni=None, Nt=None,
				shift_range=None, warp_range=None ):

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

	def aling(self, ):

		self.aling_COW_N()

		return None
		
	def aling_COW_N(self, D=None, X=None, a=None, shift_range=None, warp_range=None):
		if not type(self.D) is np.ndarray and type(self.X) is np.ndarray: 			self.D = self.X
		if not type(self.X) is np.ndarray and type(self.D) is np.ndarray: 			self.X = self.D

		if not type(D) is np.ndarray:
			if type(self.D) is np.ndarray: 		D = self.D
			else:	print('ERROR :: code a050 :: COW.aling_COW_N() :: can NOT get D({}) or self.D({})')

		if not type(X) is np.ndarray:
			if type(self.X) is np.ndarray: 		X = self.X
			else:	print('ERROR :: code a050 :: COW.aling_COW_N() :: can NOT get X({}) or self.X({})')

		if not self.isnum(a) and self.isnum(self.a): a = self.a
		elif not self.isnum(a) and not self.isnum(self.a):
			print('ERROR :: code a051 :: COW.aling_COW_N() :: can NOT get a({}) or self.a({})'.format(a, self.a))

		if not type(shift_range) is np.ndarray:
			print('ERROR :: code a050 :: COW.aling_COW_N() :: can NOT get shift_range or self.shift_range')

		if not type(warp_range) is np.ndarray:
			print('ERROR :: code a050 :: COW.aling_COW_N() :: can NOT get warp_range or self.warp_range')

		dim = D.shape
		O = D.shape

		
		for nc in range(dim[0]): 	# iteration in sample channel
			for i, ln in enumerate(dim[1:]):	# iteration in i channels
				if i >= a:	# check if channel i is supose to be aling
					for l in range(ln):  	# iteration for each l sensor of channel i

						print(l)


		return None
	
	def interpolation(self, mu, sigma, n, ):
			return np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2) / (np.linalg.norm(np.e**(-1.0/2 * ((mu-np.arange(n))/sigma)**2)))

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




