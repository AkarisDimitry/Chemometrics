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

class FUNCTIONAl(object): # generador de datos
	def __init__(self, X=None, D=None, Y=None, y=None, base=None):

		self.X = np.array(X) if not type(X) == type(None) else None
		self.D = np.array(D) if not type(D) == type(None) else None

		self.base_set
		self.base = None





