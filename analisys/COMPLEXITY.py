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
	
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# ==================== Obj  ====================  Obj  ==================== # # ==================== Obj ====================  Obj ==================== #
# ==================== ==================== ==================== ==================== === # # ==================== ==================== ==================== ==================== === #
# *** python common libraries
import logging, operator, pickle, os

class COMPLEXITY(object): # generador de datos
	def __init__(self, 	X=None, D=None, Y=None, y=None, S=None, A=None, a=None,
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

		self.N = N # INT-8 || dataset length 
		self.Nc = Nc # INT-8 || number of training sample
		self.Nv = Nv # INT-8 || number of validation sample
		self.Nt = Nt # INT-8 || number of test-samples 

		self.f = f # INT-8 || total number of factors   
		self.Na = Na # INT-8 || total number of analites 
		self.Ni = Ni # INT-8 || total number of interferents 

		self.ways = ways # toltal data tensor order


		self.colors = [ '#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',
						'#d0a64b', '#39beb9', '#1579c5', '#7b3786', '#F90061',] 
		self.font = {
				'family': 'serif',
        		'color':  (0.2, 0.2, 0.2, 1.0),
        		'weight': 'normal',
        		'size': 16,
        		}

	def self_vector_estimation(self, save=True):
	  	# --- Z -- #
		Ze = np.zeros( (Na, l1, l2) )
		for n in range(Na):	Ze[n,:,:] = np.tensordot( S[0][n,:], S[1][n,:] , axes=0)
		Z = Ze[:,0,:]
		for n in range(1, Ze.shape[1]): Z = np.concatenate( (Z ,  Ze[:,n,:]), axis=1)

	  	# --- Y --- #
		Ye = D[0,:,:,:]
		for n in range(1, N): Ye = np.concatenate( (Ye ,  D[n,:,:,:]), axis=2)
		Y = Ye[0,:,:]
		for n in range(1, Ye.shape[0]): Y = np.concatenate( (Y ,  Ye[n,:,:]), axis=0)

	  	# --- X(Lc) --- #
		Lc=np.dot(Y.T, np.linalg.pinv(Z) )
		T = np.zeros( (N, Na, l3) )
		for n in range(N): T[n,:,:] = Lc[n*l3:(n+1)*l3,:].T

		if save: self.T = T

		return T


	def W4A2( self, 	D=None, X=None, S=None, L=None, Nc=None, Na=None, Ni=None, a=None,
						save=True, v=True, 
						include_test=True, compress_nonalifned_channel=False, normalize=True ):
		'''
		Requieres S[2][f, l1/l2] and L[N][f, l3]
		
		S = <L[N] | L[N]>
		O = <S[N] | S[N]>

		'''

		if v: print('=== Complexity analysis ===')

		D = D if type(D) is np.ndarray else self.D
		X = X if type(X) is np.ndarray else self.X

		if type(D) is np.ndarray and type(X) is np.ndarray and not (X==D).all(): D = X
		S = S if type(S) is list else self.S

		Nc = Nc if type(Nc) in [np.ndarray, int, float] else self.Nc
		Na = Na if type(Na) in [np.ndarray, int, float] else self.Na
		Ni = Ni if type(Ni) in [np.ndarray, int, float] else self.Ni
		a  = a  if type(a ) in [np.ndarray, int, float] else 2
		F = Na + Ni

		N, l1, l2, l3  =  D.shape
		N = Nc if not include_test else N

		L = self.L if type(self.L) == np.ndarray else self_vector_estimation(save=False) 

		complexity_warping = np.zeros( (N, N, F) )
		for f in range(F):
			var = []
			for n1 in range(N):
				tensor_fn1 = np.tensordot( np.tensordot(S[0][f, :], S[1][f, :].T, axes=0), L[n1 ,f, :], axes=0)
				if compress_nonalifned_channel: tensor_fn1 = np.sum(tensor_fn1, axis=-1)
				tensor_fn1 = tensor_fn1/np.linalg.norm(tensor_fn1) if np.linalg.norm(tensor_fn1) != 0 and normalize else tensor_fn1 

				for n2 in range(N):
					tensor_fn2 = np.tensordot( np.tensordot(S[0][f, :], S[1][f, :], axes=0), L[n2 ,f, :] , axes=0)
					if compress_nonalifned_channel: tensor_fn2 = np.sum(tensor_fn2, axis=-1)
					tensor_fn2 = tensor_fn2/np.linalg.norm(tensor_fn2) if np.linalg.norm(tensor_fn2) != 0 and normalize else tensor_fn2 

					complexity_warping[n1, n2, f] =  np.sum(tensor_fn1*tensor_fn2)
					#if v: print( 'sample {} sample {} factor {} {}'.format(n1, n2, f,  np.sum(tensor_fn1*tensor_fn2)  ))

		complexity_overlap = np.zeros( (F, F, N) )
		for n in range(N):
			for f1 in range(F):
				tensor_f1n = np.tensordot( np.tensordot(S[0][f1, :], S[1][f1, :], axes=0), L[n ,f1, :] , axes=0)
				tensor_f1n = tensor_f1n/np.linalg.norm(tensor_f1n) if np.linalg.norm(tensor_f1n) != 0 and normalize else tensor_f1n
				for f2 in range(F):
					tensor_f2n = np.tensordot( np.tensordot(S[0][f2, :], S[1][f2, :], axes=0), L[n ,f2, :] , axes=0)
					tensor_f2n = tensor_f2n/np.linalg.norm(tensor_f2n) if np.linalg.norm(tensor_f2n) != 0 and normalize else tensor_f2n

					complexity_overlap[f1, f2, n] = np.sum( tensor_f1n*tensor_f2n )
					if np.isnan(complexity_overlap[f1, f2, n]):
						print(f1,f2,n, type(complexity_overlap[f1, f2, n]), tensor_f2n)

					#if v: print( 'sample {} factor {} factor {} {}'.format(n, f1, f2, np.sum(tensor_f2n*tensor_f2n) ))
		
		print( 'Cal-Cal samples warping', np.mean(complexity_warping[:Nc, :Nc, 0]) )
		print( 'Cal-Test samples warping',np.mean(complexity_warping[14:, :14, 0]) )
		print( 'All samples warping',np.mean(complexity_warping[:, :, 0]) )

		print(complexity_warping.shape, complexity_overlap.shape)
		complexity_overlap[0,1,:] = np.sum(S[0][0, :]*S[0][1, :])
		print(self.L.shape, self.S[0].shape, self.S[1].shape)
		plt.figure(10), plt.plot(self.L[:,0,:].T)
		plt.figure(11), plt.plot(self.L[:,1,:].T)
		plt.figure(12), plt.plot(self.S[0].T)

		plt.matshow(complexity_warping[:,:,0] / np.max(complexity_warping[:,:,0]) )
		plt.matshow(complexity_warping[:,:,1] / np.max(complexity_warping[:,:,1]) )

		plt.matshow( np.mean(complexity_overlap[:,:,:], axis=2)/ np.max( np.mean(complexity_overlap[:,:,:], axis=2))  )
		plt.figure(20), plt.plot( complexity_overlap[0,1,:]  )
		plt.figure(20), plt.plot( complexity_overlap[0,0,:]  )
		plt.show()

		if save:
			self.complexity_warping = complexity_warping
			self.complexity_overlap = complexity_overlap

		return complexity_warping, complexity_overlap

	def complexity_analysis(self, 	a=None, L=None, S=None, D=None, X=None,
									verbosity=False, save=True):

		a = a if type(a) != type(None) else self.a
		X = X if type(X) != type(None) else self.X
		D = D if type(D) != type(None) else self.D
		L = L if type(L) != type(None) else self.L
		S = S if type(S) != type(None) else self.S

		C = self.W4A2(  D=D, X=X, L=L, S=S, a=a )
		
		if save: 
			self.a = a
			self.X = X
			self.D = D
			self.L = L
			self.S = S

			self.C = C

		return C

	def plot(self, ax=None, save=None):

		plt.matshow( self.complexity_warping[:,:,0])
		plt.matshow( self.complexity_overlap[:,:,0])