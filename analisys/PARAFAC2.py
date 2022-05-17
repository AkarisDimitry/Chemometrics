# *** warning supresion *** #
import warnings
warnings.filterwarnings("ignore")

# *** numeric libraries *** #
import numpy as np #pip3 install numpy
from functools import reduce
try:        import scipy
except:     print('WARNNING :: PARAFAC2.py :: can NOT correctly load "scipy" libraries')

# *** graph libraries *** #
try:
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pl
except: 
    print('WAENNING :: main_simulation.py :: can NOT correctly load "matplotlib" libraries')
    print('Install by: ( pip3 install matplotlib )')
#Kiers, H. A., Ten Berge, J. M., & Bro, R. (1999). PARAFAC2-Part I. A direct fitting algorithm for the PARAFAC2 model. Journal of Chemometrics, 13(3-4), 275-294.

class PARAFAC2(object): # generador de datos
    def __init__(self, X=None, D=None, S=None, 
                    C=None, A=None,
                    Y=None, y=None, 
                    N=None, Nc=None, Nt=None,
                    f=None, Na=None, Ni=None):

        if type(X) != type (None):
            try: self.X = np.array(X)   
            except: print('WARNING :: code 400 PARAFAC2.PARAFAC2() :: can not load X data in to self.X')
        else: self.X = X

        if type(D) != type (None):
            try: self.D = np.array(D)
            except: print('WARNING :: code 400 PARAFAC2.PARAFAC2() :: can not load X data in to self.D')
        else: self.D = self.X

        if type(self.X) is np.ndarray and type(self.D) is np.ndarray and  self.X.shape != self.D.shape:
            self.D = self.X
            print('WARNING :: code 401 PARAFAC2.PARAFAC2() :: self.X.shape({}) != self.D.shape({})'.format(self.X.shape, self.D.shape) ) 
            print('WARNING :: making self.D = self.X')

        # ---- PARAFACII exclusive parameters ---- #
        self.F = None
        self.A = None
        self.P = None
        self.conv = None
        self.niters = None
        self.MSE = None
        
        self.xp = None
        self.xpc = None
        self.xc = None

        self.Y = Y
        self.y = y

        self.N  = N
        self.Nc = Nc
        self.Nt = Nt

        self.f  = f
        self.Na = Na
        self.Ni = Ni

        self.L = []
        self.loadings = []          # loadings 
        self.Nloadings = []         # normalized loadings 
        self.model_mse = None       # MSE error 

        self.constraints = None

        self.font = {
                'family': 'serif',
                'color':  (0.2, 0.2, 0.2, 1.0),
                'weight': 'normal',
                'size': 16,
                }

    def train(self, x=None, nfactors=None, max_iter=10, inicializacion='random', constraints={}, v=False, save=True):
        if type(x) != type(None):   x = x
        elif type(self.X) != type(None): x = self.X
        else: print()

        if len(x.shape) == 4: 
            xc = self.compress_3WAY(x=x)
        
        if len(x.shape) == 3: 
            xc = x
        
        if save: self.xc = xc
        F, D, A = self.parafac2(X=xc, constraints=constraints, r=self.f, tol=1e-5, verbose=v, save=save)
        if save:
            self.F = F # eigen tensor NON-aligned channel
            self.D = np.flip(D.T, 0) # eigenvalue 
            self.A = A  # eigen tensors well-aligned channel

        if len(x.shape) == 4: 
            xe1, xe2 = self.decompress_3WAY(xc=A, x=x) # descomprime los modos alineados comprimidos durante el trinning en PARAFAC2.parafac2()
            if save: self.loadings = [self.D, xe1, xe2, self.F]

        if len(x.shape) == 3: 
            if save: self.loadings = [self.D, self.A, self.F]

        if 'normalized' in constraints and constraints['normalized']:
            s, self.loadings = self.normalized_loadings(self.loadings)
            
        return (self.loadings, self.MSE)

    def compress_3WAY(self, x=None, verbose=False, save=True):
        if type(x) == type(None): x = self.x
        else: pass

        if len( x.shape ) != 4: print('WARNING :: PARAFAC2.compress32 :: incorrect data shape ')
        
        N, l1, l2, l3 = x.shape
        
        xp = np.transpose(x, (0,3,1,2))
        xpc = xp.reshape( (N, l3, l1*l2))
        if save: self.xp, self.xpc = xp, xpc
        return xpc
    
    def decompress_3WAY(self, xc=None, x=None, Ns=None, verbose=False, save=True):
        if type(xc) == type(None): xc = self.xc
        else: pass

        if type(x) == type(None): x = self.x
        else: pass

        N, l1, l2, l3 = x.shape
        Ns = xc.shape[-1]

        xe1 = np.mean(np.transpose(xc.reshape((l1, l2, Ns)), (1,0,2)), axis=1)
        xe2 = np.mean(xc.reshape((l1, l2, Ns)), axis=1)

        return xe1, xe2 

    def parafac2(self, X, constraints={}, r=2, tol=1e-5, verbose=True, save=True):
        m = len(X)
        F = np.identity(r)
        D = np.ones((m, r))
        A = np.linalg.eigh(reduce(lambda A, B: A + B, map(lambda Xi: Xi.T.dot(Xi), X)))
        A = A[1][:, np.argsort(A[0])][:, -r:]
       
        H = [np.linalg.qr(Xi, mode='r') if Xi.shape[0] > Xi.shape[1] else Xi for Xi in X]
        G = [np.identity(r), np.identity(r), np.ones((r, r)) * m]

        err = 1
        conv = False
        niters = 0
        while not conv and niters < 100:
            P = [np.linalg.svd((F * D[i, :]).dot(H[i].dot(A).T), full_matrices=0) for i in range(m)]
            P = [(S[0].dot(S[2])).T for S in P]
            T = np.array([P[i].T.dot(H[i]) for i in range(m)])
            
            F = np.reshape(np.transpose(T, (0, 2, 1)), (-1, T.shape[1])).T.dot( self._KhatriRao(D, A)).dot(np.linalg.pinv(G[2] * G[1]))
            G[0] = F.T.dot(F)
            A = np.reshape(np.transpose(T, (0, 1, 2)), (-1, T.shape[2])).T.dot( self._KhatriRao(D, F)).dot(np.linalg.pinv(G[2] * G[0]))
            G[1] = A.T.dot(A)
            D = np.reshape(np.transpose(T, (2, 1, 0)), (-1, T.shape[0])).T.dot( self._KhatriRao(A, F)).dot(np.linalg.pinv(G[1] * G[0]))
            G[2] = D.T.dot(D)        
            err_old = err
            err = np.sum(np.sum((H[i] - (P[i].dot(F) * D[i, :]).dot(A.T)) ** 2) for i in range(m))
            niters += 1        
            conv = abs(err_old - err) < tol * err_old
            if verbose: print("Iteration {0}; error = {1:.6f}".format(niters, err))

        P = [np.linalg.svd((F * D[i, :]).dot(X[i].dot(A).T), full_matrices=0) for i in range(m)]
        F = [(S[0].dot(S[2])).T.dot(F) for S in P]

        if save:
            self.F = F
            self.D = D
            self.A = A
            self.G = G
            self.P = P
            self.conv = conv
            self.MSE = conv
            self.niters = niters

        return F, D, A
        
    def _KhatriRao(self, A, B):
        return np.repeat(A, B.shape[0], axis=0) * np.tile(B, (A.shape[0], 1))

    def normalized_loadings(self, loadings=None):
        if not loadings == None: loadings = self.loadings 
        mags = np.asarray([np.apply_along_axis(np.linalg.norm, 1, mode)
        for mode in loadings])

        norm_loadings = [loadings[mi] / mags[mi].reshape(-1, 1) 
        for mi in range(len(loadings))]

        mags = np.prod(mags, axis=0)
        order = np.argsort(mags)[::-1]
     
        return mags[order], [np.asarray([mode[fi] for fi in order]) for mode in norm_loadings]

    def predic(self, Nc=None, Na=None, Ni=None, N=None, f=None, v=0):
        if not type(N) is np.ndarray and not type(N) is int and N == None and (type(self.N) is np.ndarray or type(self.N) is int): N = self.N
        else: print('ERROR :: code 050 :: PARAFAC2.predic() :: can NOT get N({}) or self.N({})'.format(N, self.N))
        if not type(Nc) is np.ndarray and not type(Nc) is int and Nc == None and (type(self.Nc) is np.ndarray or type(self.Nc) is int): Nc = self.Nc
        else: print('ERROR :: code 050 :: PARAFAC2.predic() :: can NOT get Nc({}) or self.Nc({})'.format(Nc, self.Nc))

        if not type(Na) is np.ndarray and not type(Na) is int and Na == None and (type(self.Na) is np.ndarray or type(self.Na) is int): Na = self.Na
        else: print('ERROR :: code 050 :: PARAFAC2.predic() :: can NOT get Na({}) or self.Na({})'.format(Na, self.Na))
        if not type(Ni) is np.ndarray and not type(Ni) is int and Ni == None and (type(self.Ni) is np.ndarray or type(self.Ni) is int): Ni = self.Ni
        else: print('ERROR :: code 050 :: PARAFAC2.predic() :: can NOT get Ni({}) or self.Ni({})'.format(Ni, self.Ni))

        if not type(f) is np.ndarray and not type(f) is int and f == None and (type(self.f) is np.ndarray or type(self.f) is int): f = self.f
        else: print('ERROR :: code 050 :: PARAFAC2.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))

        if not type(f) is np.ndarray and not type(f) is int and type(Na) is int and type(Ni) is int: f = Na + Ni
        else: print('ERROR :: code 050 :: PARAFAC2.predic() :: can NOT get f({}) or self.f({})'.format(f, self.f))
        
        if not type(self.y) is np.ndarray or not self.y.shape == (self.f, self.X.shape[0]): 
            if type(self.y) is np.ndarray:  print('WARNING :: code 1000 PARAFAC2.predic() :: Previuos data store in self.y will be erased. ')
            self.y = np.zeros((self.f, self.X.shape[0]))

        if v >= 1:  print(' (1) Predic tensor coeficients.')
        self.A = self.loadings[0]

        if v >= 2:  
            print(' \t\t * coeficients : {}'.format(self.A.shape))
            for n in range(self.A.shape[1]):
                vec_str = '\t {} '.format( int(n+1) )
                for m in range(self.A.shape[0]):    
                    vec_str += '\t\t {:<.3f}'.format(self.A[m][n])
                    if v >= 3: vec_str += '\t\t {:<.3f}'.format(self.A[m][n]/np.linalg.norm(self.A[m,:]))

        if v >= 1:  print(' (2) Predic calibration, validation and test samples.')
        if self.Na > 1: self.MSE = np.ones((self.Na))*99**3
        else: self.MSE = np.array([99**3])
        ref_index = np.zeros((self.Na))

        for n in range(self.Na):
            for m in range(self.Na+self.Ni):
                try:
                    z = np.polyfit( self.A[int(m), :self.Nc] , self.Y[:self.Nc, int(n)], 1)
                    self.model_prediction = self.A[int(m), :]*z[0]+z[1]
                    predic = self.model_prediction
                    MSE_nm = np.sum(( predic[:self.Nc] - self.Y[:self.Nc, int(n)])**2)
                except: MSE_nm = 99**3
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
            if fig == None: fig = plt.figure()
            else: fig=fig
        except:
            print('ERROR :: code 020b PARAFAC.plot_loadings() :: can NOT generate fig obj')
        
        #try:   ax = fig.add_subplot(100*len(self.Nloadings)+11+100)
        #except:    print('ERROR :: code 020c PARAFAC.plot_loadings() :: can NOT generate axis, maybe fig argument is NOT what except ')
                
        for i, loading in enumerate(self.Nloadings): 
            # ***** PLOT loading ***** #
            try:    ax = fig.add_subplot(len(self.Nloadings)*100+11+i)
            except: print('ERROR :: code 020c PARAFAC.plot_loadings() :: can NOT generate axis, maybe fig argument is NOT what except ')
            for j, vector in enumerate(loading):
                ax.plot(vector, color=self.colors[j], lw=1.5, alpha=0.6, marker='o', markersize=3, label='Factor {}'.format(j+1) )


            try:
                ax.set_xlabel('Variable')
                ax.set_ylabel('Intencity')
                #ax.set_title('Factor plot')
                #ax.set_xlabel('variable', fontsize=12);        ax.set_ylabel('Absorbance (a.u.)', fontsize=12)
                ax.spines["right"].set_linewidth(2);        ax.spines["right"].set_color("#333333")
                ax.spines["left"].set_linewidth(2);         ax.spines["left"].set_color("#333333")
                ax.spines["top"].set_linewidth(2);          ax.spines["top"].set_color("#333333")
                ax.spines["bottom"].set_linewidth(2);       ax.spines["bottom"].set_color("#333333")
            except:     print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not configure axis')


            try:
                if names != None:
                    legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
                    legend.get_frame().set_facecolor('#FFFFFF')
            except:     print('ERROR :: code 002c FPAV21.plot_spectra() :: Can not configure legend')

        return None

'''
TEST N = 21
mode00 = np.arange(5)
mode01 = np.e**(  -(np.arange(N)-5)**2/2  )
mode02 = np.e**(  -(np.arange(N)-10)**2/2  )
mode03 = np.e**(  -(np.arange(N+5)-15)**2/2  )

A = np.dot(mode00[:,np.newaxis], mode01[:,np.newaxis].T)
B = np.dot( A[:,:,np.newaxis], mode02[:,np.newaxis].T )
C = np.dot( B[:,:,:,np.newaxis], mode03[:,np.newaxis].T )

mode00 = np.arange(5)*5
mode01 = np.e**(  -(np.arange(N)-10)**2/2  )
mode02 = np.e**(  -(np.arange(N)-5)**2/2  )
mode03 = np.e**(  -(np.arange(N+5)-5)**2/2  )

A = np.dot(mode00[:,np.newaxis], mode01[:,np.newaxis].T)
B = np.dot( A[:,:,np.newaxis], mode02[:,np.newaxis].T )
D = np.dot( B[:,:,:,np.newaxis], mode03[:,np.newaxis].T )


C = C + D
parafac2 = PARAFAC2()

parafac2.Y = np.array([ [1,2,3,4,5], [1,2,3,4,5]]).T
parafac2.Nc = 3
parafac2.Nt = 2
parafac2.N = [3,0,2]

parafac2.f = 2
parafac2.Na = 2
parafac2.Ni = 0
parafac2.X = C

parafac2.train(v=True)
parafac2.plot_loadings()

parafac2.predic(v=True)
'''