# *** warning supresion
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries *** #
import numpy as np
import scipy.io

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

class KFOLD(object): # generador de datos
	def __init__(self, X=None, D=None, Y=None, y=None, 
					   f=None, N=None, Nc=None, Nv=None, Nt=None, Na=None, Ni=None,):

		# === DATA === #
		self.X = np.array(X) if not type(X) == type(None) else None
		self.Y = np.array(Y) if not type(Y) == type(None) else None
		self.D = D
		self.y = y

		self.N = N # INT-8 || dataset length 
		self.Nc = Nc # INT-8 || number of training sample
		self.Nv = Nv # INT-8 || number of validation sample
		self.Nt = Nt # INT-8 || number of test-samples 

		self.f = f # INT-8 || total number of factors   
		self.Na = Na # INT-8 || total number of analites 
		self.Ni = Ni # INT-8 || total number of interferents 

		# === PARAMETERS === #
		self.index_calibration = []
		self.index_test = []
		self.index_validation = []


	def split():
		pass

	def split_data():
		pass
		
	def split_test():
		pass
		
	def split_calibration(self, n_splits=5, independence=True, shuffle=False, random_state=None, x=None ):
		if random_state != None and type(random_state) == int:
			np.randomseed(random_state)
		elif random_state != None and not type(random_state) == int:
			print('WARNNING :: KFOLD.split_calibration() :: can NOT asing randomseed')

		try:
			x = x if type(x) != None else self.X
			x = x if type(x) != None else self.D
		except:
			# print('WARNNING :: KFOLD.split_calibration() :: can NOT asing data set (self.X or self.D)')

		Nc = self.Nc if type(self.Nc) != None else None 
		#Nc = x.shape[0]	
		
		if independence == True:
			if x.shape[0] != Nc and type(Nc) != None:  
				print('WARNNING :: KFOLD.split_calibration() :: Calibration number inconsistency')

			
			

	def split_validation():
		pass
		
	def split_val_cal():
		pass
		

if __name__ == '__main__':
	kfold = KFOLD()
