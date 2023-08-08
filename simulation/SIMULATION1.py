# *** warning supresion *** #
import warnings
warnings.filterwarnings("ignore")
try:
	import os, functools
except:
	print('WARNNING :: SIMULATION.py :: can NOT import os. ( needed for SIMULATION.recursive_read() )')

import numpy as np

try:
	import matplotlib.pyplot as plt
	from matplotlib.pyplot import cm
except: 
	print('WARNNING :: SIMULATION.py :: can NOT correctly load "matplotlib" libraries')
	print('Install by: "pip3 install matplotlib" )')
	
import itertools, copy

# *** chemometric libraries *** #
try:
	from analisys import MCR, PARAFAC, PARAFAC2, GPV
	from alig import FAPV32, COW,  FAPV21, ICOSHIFT, MCR_ICOSHIFT
	from data import DATA
except: 
	print('WARNNING :: SIMULATION.py :: can NOT correctly load chemometric libraries !'
		)
class SIMULATION(object):
	def __init__(self, model_trainning=None, model_aling=None, data=None, generator=None, verbose=0 ):

		self.hyperparameters = None  			# dic with all hyperparameters
		self.hyperparameters_exploration = None 
		
		self.model_trainning = None				# list with analisis models obj
		self.model_aling = None					# list with alignement models obj
			
		self.data = None						# DATA obj
		self.generator = None					# data set generator obj

		self.repeat = None
		self.save_path = None				# path + filename 
		self.save_metadata = 	{			# wich data/metadata save 
								'Y':			False,		# (0) 'Y' signal response 
								'y':			False,		# (1) 'y' signal response 
								'X':			False,		# (2) 'X' data trainning/validation/test samples 
								'RMSD':			True,		# (3) 'RMSD' RMSD in prediction
								'hp':			True,		# (4) 'hp' hiperparemeters of simulation.
								'summary':		False,		# (5) complete summary of results
								'DataSet':		False,		# (6) Completely save  data set 
								'code':			0,			# (7) Save slot identification
								'Maling':		True,			# (8) Save aling model
								'Mtrainning':	True,			# (9) Save trannig model

								}

		self.save_fulldata = {'mode':False, 'code':0} # save fulldata set

		self.verbose = verbose

	def file_exist(self, filename, ):
		filename = filename if type(filename) == str else  None

		try:
			f = open( str(filename) )
			f.close()
			return True
		except IOError:
			print("File not accessible")
			return False

	def exploration(self, hyperparameters_exploration):
		dic_hp = hyperparameters_exploration
		lista_names = list(dic_hp.keys())
		lista = [ dic_hp[k] for k in dic_hp.keys() ]
		lista_len = [ list(range(len(n))) for n in lista ]
		for i, m in enumerate(itertools.product(*lista_len, repeat=1)):
			for n in range(self.repeat):
				dic = {lista_names[i]:dic_hp[lista_names[i]][o] for i, o in enumerate(m)}

				if 	dic['ways'] == len(dic['dimentions']) and dic['ways'] == len(dic['deformations']) and dic['ways'] == len(dic['warping']):
					if 'code' in self.save_metadata and 'DataSet' in self.save_metadata and self.save_metadata['DataSet']:
						self.save_metadata['code'] = i * self.repeat + n

					self.evaluate(hp=dic)
				else:
					if self.verbose > 0:	print('WARNING :: SIMULATION.exploration() :: parameters inconsistency :: hyperparameters_exploration must be change.')
					else: 					pass

	def evaluate(self, hp=None, save_path=None):
		# -- evaluate the simulation hyper_parameters -- # 
		#	hp 	||	dict 	||  hyper_parameters  eg:
		'''
		hp = {
		# ----------------- volume ----------------- #
		'ways' 			:	[3]								,	# number of ways # 				eg: ways=3
		'dimentions' 	:	[[50,60,40]] 					,	# dimentionality of each way # 	eg: dimentions=[50,60,40]
		'N' 			:	[[6,7,5]] 						,	# [trainnig samples, validation samples, test samples] # eg: N=[6,7,5]
		# ----------------- Compose ----------------- #
		'factors' 		:	[2] 							,	# number of calibrated factors #		eg:  factors=3
		'interferentes' :	[2] 							,	# number of interferents #				eg:  interferentes=0 
		'aligned' 		:	[2] 							,	# number spected aligned channels #		eg:  aligned=2 
		# ----------------- complexity ----------------- #
		'functional' 	:	['Gaussian'] 					,	# functional basee form of pure vectors # eg: functional='Gaussian'
		'deformations'	:	[[[0,0,1],[0,0,1],[0,0,1]]] 	,	# deformations # 				eg: deformations=[[0.0,0.0,1],[0,0,1],[0,0,1]]
		'warping' 		:	[[[0,0],[0,0],[0,0]]]			,	# warping # 					eg: warping=[[0.0,0.0],[0,0],[0,0]] 
		'sd' 			:	[2]								,	# SD of pure vectors # 			eg: sd=2
		'overlaping' 	:	[0]								,	# overlapping of pure vectors # eg: overlaping=0
		# ----------------- noise ----------------- #
		'noise_type' 	:	['random']						,	# noise structure #				eg: noise_type='random'
		'intencity' 	:	[0.0001]						,	# noise intensity # 			eg: intencity=0.001
		}
		'''
		# - (1) Variable setter - #
		save_path = save_path if type(save_path) == str else self.save_path

		# - (2) USE class instance - #
		sample 	= self.generator
	
		# - (3) define problem parameters - #
		# ----------------- volume ----------------- #
		ways 			=	hp['ways']				# number of ways # 				eg: ways=3
		dimentions 		=	hp['dimentions'] 		# dimentionality of each way # 	eg: dimentions=[50,60,40]
		N 				=	hp['N'] 				# [trainnig samples, validation samples, test samples] # eg: N=[6,7,5]
		# ----------------- Compose ----------------- #
		factors 		=	hp['factors'] 			# number of calibrated factors #		eg:  factors=3
		interferentes 	=	hp['interferentes'] 	# number of interferents #				eg:  interferentes=0 
		aligned 		=	hp['aligned'] 			# number spected aligned channels #		eg:  aligned=2 
		# ----------------- complexity ----------------- #
		functional 		=	hp['functional'] 		# functional basee form of pure vectors # eg: functional='Gaussian'
		deformations	=	hp['deformations'] 		# deformations # 				eg: deformations=[[0.0,0.0,1],[0,0,1],[0,0,1]]
		warping 		=	hp['warping'] 			# warping # 					eg: warping=[[0.0,0.0],[0,0],[0,0]] 
		sd 				=	hp['sd']				# SD of pure vectors # 			eg: sd=2
		overlaping 		=	hp['overlaping'] 		# overlapping of pure vectors # eg: overlaping=0
		# ----------------- noise ----------------- #
		noise_type 		=	hp['noise_type']		# noise structure #				eg: noise_type='random'
		intencity 		=	hp['intencity']			# noise intensity # 			eg: intencity=0.001

		
		# - (4) generate problem - #
		X, Y, metadata = sample.generate(factors=factors, N=N, Ni=interferentes, 
										functional=functional, deformations=deformations, warping=warping, sd=sd, overlaping=overlaping,
										dimentions=dimentions, ways=ways, noise_type=noise_type, intencity=intencity)

		# - (5) inyect sample into data - #
		sample.inyect_data(self.data)
		self.data.a = aligned

		# - (6) solve aling model - #
		for model_aling_n, model_aling in enumerate(self.model_aling):
			
			if type(model_aling) != type(None) and type(model_aling['model']) != type(None):

				if model_aling['model'] == 'FAPV32':	
					model_aling = {'model':FAPV32(), 'init':'self-vector'}

				elif model_aling['model'] == 'ICOSHIFT':
					model_aling = {'model':ICOSHIFT()}
				
				elif model_aling['model'] == 'MCR-ICOSHIFT':
					model_aling = {'model':MCR_ICOSHIFT()}
				
				pv = [ l[-1,:,:] for l in sample.pure_vectors ]

				# initialization # 
				self.data.inject_data( model_aling['model'] )
				model_aling['model'].S = pv

				# aliniation # 
				Da, a, b = model_aling['model'].aling()
				self.data.D, self.data.X = Da, Da

			# - (7) solve prediction problem - #
			for t, model in enumerate(self.model_trainning):

				if model == 'PARAFAC':	
					model = PARAFAC()

				elif model == 'PARAFAC2':	
					model = PARAFAC2()

				elif model == 'MCR':
					model = MCR()

				elif model == 'GPV':
					model = GPV()

				self.data.inject_data(model)

				self.data.model = model  								# load data into the model 	#
				self.data.train( )										# train the model 			#
				self.data.predic(v=0)									# predic with the model 	#
				if self.verbose > 0:	self.data.summary()				# summary results 			#
				else: 					pass		
					
				# - (8) save partial results - #
				self.save(save_path=save_path, hp=hp, model_analisis=t, model_aling=model_aling, model_trainning=model) # save partial results each step #

	def save(self, save_path=None, hp=None, model_analisis=None, model_aling=None, save_metadata=None, model_trainning=None):
		# variables setter
		save_path = save_path if type(save_path) == str else self.save_path
		namefile = save_path+'/metafile'

		save_metadata = save_metadata if type(save_metadata) == dict else self.save_metadata

		if not type(save_metadata) == dict:
			print('ERROR :: code X :: SIMULATION.save() :: wrong type(save_metadata) :: except dict instead of {} '.format(type(save_metadata)) ) 

			# ----------- ----------- ----------- --------- #
			# *** ----------- FULL DATA SET ----------- *** #
		if 'DataSet' in save_metadata and 'code' in save_metadata and save_metadata['DataSet']: 
			self.data.save_text(file='{}/{}'.format(save_path, str(save_metadata['code'])), delimiter='\t')

			# ----------- ----------- ----------- --------- #
			# *** ----------- RMSD Erros    ----------- *** #
		if 'RMSD' in save_metadata and save_metadata['RMSD']: 			
			file_meanE 	= open( namefile+'.meanE', 'a') if self.file_exist(namefile+'.meanE') else open( namefile+'.meanE', 'w')
			file_RMSD 	= open( namefile+'.RMSD', 'a')  if self.file_exist(namefile+'.RMSD')  else open( namefile+'.RMSD', 'w')
			file_NRMSD 	= open( namefile+'.NRMSD', 'a')  if self.file_exist(namefile+'.NRMSD')  else open( namefile+'.NRMSD', 'w')

			for f in range(self.data.f):
				mean_e 	= np.mean(np.abs(self.data.Y[:, f]-self.data.y[f, :]))								# mean value of absolute error 											#
				RMSD 	= (np.sum((self.data.Y[:, f]-self.data.y[f, :])**2)/self.data.y.shape[1])**0.5		# root-mean-square deviation (RMSD) or root-mean-square error (RMSE) 	#
				NRMSD 	= RMSD/np.mean(self.data.y[f, :])													# Normalized root-mean-square deviation (Normalizing the RMSD) 			#
				file_meanE.write('\t\t {:<.6f} '.format(mean_e) )
				file_RMSD.write('\t\t {:<.6f} '.format(RMSD) )
				file_NRMSD.write('\t\t {:<.6f} '.format(NRMSD) )

			file_meanE.write('\n'); 	file_RMSD.write('\n');	file_NRMSD.write('\n')
			file_meanE.close();			file_RMSD.close();		file_NRMSD.close()

			# ----------- ----------- ----------- --------- #
			# *** ----------- hyperparameters ----------- *** #
		if 'hp' in save_metadata and save_metadata['hp']: 
			file_hp 	= open( namefile+'.hp', 'a')
			lista_names = list(hp.keys())
			lista = [ hp[k] for k in hp.keys() ]

			for n in lista_names:
				file_hp.write('{} : {} \t'.format(n, hp[n]) )
			file_hp.write('\n' )

			file_hp.close();

			# ----------- ----------- ----------- --------- #
			# *** ----------- aling model   ----------- *** #
		if 'Maling' in save_metadata and save_metadata['Maling']: 
			file_ma 	= open( namefile+'.Maling', 'a')
			file_ma.write('{} \n'.format(model_aling) )
			file_ma.close();

			# ----------- ----------- ----------- --------- #
			# *** ----------- trainning model   ----------- *** #
		if 'Mtrainning' in save_metadata and save_metadata['Mtrainning']: 
			file_ma 	= open( namefile+'.Mtrainning', 'a')
			file_ma.write('{} \n'.format(model_trainning) )
			file_ma.close();

			# ----------- ----------- ----------- --------- #
			# *** ----------- summary intro ----------- *** #
		if 'summary' in save_metadata and save_metadata['summary']:  
			try:
				file_summary 	= open( namefile+'.summary', 'a')
			except:
				print('ERROR :: code 010E SIMULATION.save() :: Can not open {} , can NOT save summary '.format(namefile+'.summary') )
			# *** ----------- summary intro ----------- *** #
			file_summary.write('{}  {} \n'.format('*'*15, '*'*15) )
			file_summary.write('{} SUMMARY {} \n'.format('*'*10, '*'*10) )
			file_summary.write('{}  {} \n\n'.format('*'*15, '*'*15) )
			# *** ----------- hyperparameters ----------- *** #
			try:
				lista_names = list(hp.keys())
				lista = [ hp[k] for k in hp.keys() ]
				for n in lista_names:	file_hp.write('{} : {} \t\n'.format(n, hp[n]) )
			except: file_summary.write('ERROR :: code 010a SIMULATION.save() :: Can not summarise hyperparameters(hp) \n')

			try: # *** ----------- X data analisys ----------- *** #
				if type(self.data.X) is np.ndarray:
					file_summary.write(' * X data loaded :: shape {}\n'.format(str(self.data.X.shape)) )
			except: file_summary.write('ERROR :: code 010b SIMULATION.save() :: Can not summarise self.data.X \n')


			try: # *** ----------- ways data analisys ----------- *** #
				if type(self.data.ways) is np.ndarray:
					file_summary.write(' * {} way data loaded\n'.format(str(self.data.ways)) )
			except: file_summary.write('ERROR :: code 010c SIMULATION.save() :: Can not summarise self.data.ways\n')
	 
			try: # *** ----------- Y data analisys ----------- *** #
				if type(self.data.Y) is np.ndarray and len(self.data.Y.shape)==2:
						file_summary.write(' * Y data loaded :: shape {}\n'.format(str(self.data.Y.shape)) )
						for i in range(self.data.Y.shape[0]): 
							temporal_text = ''
							for j in range(self.data.Y.shape[1]):	temporal_text += '{:<.6f} \t'.format(self.data.Y[i,j])
							file_summary.write(' ({}) \t {} \n'.format(i, temporal_text) )
				elif len(self.data.Y.shape)==2:
					file_summary.write('WARNING :: code 010d SIMULATION.save() :: Can not summarise self.data.Y\n')

			except: file_summary.write('ERROR :: code 010d SIMULATION.save() :: Can not summarise self.data.Y\n ')

			try: # *** ----------- y data analisys ----------- *** #
				if not type(self.data.y) is np.ndarray:
					if type(self.data.y) == list: self.data.y = np.array(self.data.y)
					else:
						try: 
							if type(self.data.model.y) is np.ndarray: self.data.y = self.data.model.y
						except: file_summary.write('WARNING :: code 010e SIMULATION.save() :: Can not summarise self.data.y can not get array from self.y or self.model.y \n')

				if type(self.data.y) is np.ndarray and len(self.data.y.shape)==2:
						file_summary.write(' * y data loaded :: shape {} \n'.format(str(self.data.y.shape)) )
						for i in range(self.data.y.shape[1]): 
							temporal_text = ''
							for j in range(self.data.y.shape[0]):	temporal_text += '{:<.6f} \t'.format(self.data.y[j,i])
							file_summary.write(' ({}) \t {} \n'.format(i, temporal_text) )
				elif len(self.data.Y.shape)==2:
					file_summary.write('WARNING :: code 010e SIMULATION.save() :: Can not summarise self.data.y can not get array from self.y or self.model.y\n')

			except: file_summary.write('ERROR :: code 010e SIMULATION.save() :: Can not summarise self.data.y can not get array from self.data.y or self.model.y\n')


			try: # *** model prediction analisys *** #
				if not type(self.data.y) is np.ndarray: # load propel signal response prediction (self.y) 
					try: 
						if type(self.data.model.y) is np.ndarray: self.data.y = self.data.model.y
					except: file_summary.write('WARNING :: code 010f SIMULATION.save() :: Can not summarise self.data.y can not get array from self.data.y or self.data.model.y\n')

				if type(self.data.y) is np.ndarray and len(self.data.y.shape)==2 and type(self.data.Y) is np.ndarray :

					for a in range(self.data.y.shape[0]): 
						file_summary.write('\n {} Factor {} {} \n'.format('*'*30, int(a+1), '*'*30) )
						file_summary.write('Sample \t\t\t Y \t\t\t y(stimation) \t\t\t Error \t\t\t e% \t\t\t \n')
						for n in range(self.data.y.shape[1]):
							if self.data.Nc == n:
								file_summary.write(' ---------------- '*5 +'\n')
							
							if self.data.Nc > n:
								file_summary.write('Cal{} \t\t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f}\n'.format(n+1, self.data.Y[n, a], self.data.y[a, n],  
																	(self.data.Y[n, a]-self.data.y[a, n]),  (self.data.Y[n, a]-self.data.y[a, n])*100/self.data.Y[n, a] 	) )
							else:
								file_summary.write('Test{} \t\t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f} \t\t {:<.4f}\n'.format(n+1, self.data.Y[n, a], self.data.y[a, n],  
																	(self.data.Y[n, a]-self.data.y[a, n]),  (self.data.Y[n, a]-self.data.y[a, n])*100/self.data.Y[n, a] ) )

						# --- some general descriptors of the trainning --- #
						mean_e 	= np.mean(np.abs(self.data.Y[:, a]-self.data.y[a, :]))						# mean value of absolute error 											#
						RMSD 	= (np.sum((self.data.Y[:, a]-self.data.y[a, :])**2)/self.data.y.shape[1])**0.5		# root-mean-square deviation (RMSD) or root-mean-square error (RMSE) 	#
						NRMSD 	= RMSD/np.mean(self.data.y[a, :])										# Normalized root-mean-square deviation (Normalizing the RMSD) 			#
						file_summary.write('mean(abs(e)) {:<.4f} \t\t RMSD(RMSE) : {:<.4f} \t\t NRMSD(NRMSE) {:<.4f}\n '.format( 	mean_e,	RMSD, NRMSD	,))

				elif len(self.data.y.shape)==2:	file_summary.write('WARNING :: code 010g SIMULATION.save() :: Can not summarise self.data.y (test response stimation)')

			except: file_summary.write('ERROR :: code 010g SIMULATION.save() :: Can not summarise self.data.y (test response stimation) \n')

			file_summary.write('{} /END_SUMMARY {} \n\n\n'.format('*'*10, '*'*10) )
			file_summary.write('\n' )
			file_summary.close();			

	def read(self, path, files_list=None, aling_model_dict=None, analysis_model_dict=None):

		def read_Maling(path=None, name=None, aling_model_dict=None):
			# --- reads the model aling('metafile.Maling') file ---
			path = path if type(path) == str else None
			name = name if type(name) == str else None
			aling_model_dict = { 'FAPV32':'FAPV32',  'None':'None', 'ICOSHIFT':'ICOSHIFT',} if aling_model_dict == None else aling_model_dict
		
			file_Maling = open( '{0}/{1}'.format(path, name), 'r' ) 

			# - get entire model name list - #
			# eg. ['<alig.ICOSHIFT.ICOSHIFTobjectat0x7fb45b03f4a8>}\n', '<alig.ICOSHIFT.ICOSHIFTobjectat0x7fb45b03f4a8>}\n', "<alig.FAPV32.FAPV32objectat0x7fb4598a8b38>,'init'", "<alig.FAPV32.FAPV32objectat0x7fb4598a8b38>,'init'", 'None}\n']
			aling_model = [ m.split(':')[1].replace(" ", "") for n in file_Maling for i, m in enumerate(list(filter(lambda x: (len(x) > 1), n.split('\t'))) )  ]

			# - get match list - #
			# eg. [['ICOSHIFT'], ['ICOSHIFT'], ['FAPV32'], ['FAPV32'], ['None']] 
			aling_model = [ list(filter(lambda x: ( type(x) == str ), [am if am in a else False for am in aling_model_dict])) for a in aling_model ]
			# - get first element of the match list - #
			# eg. ['ICOSHIFT', 'ICOSHIFT', 'FAPV32', 'FAPV32', 'None']
			aling_model = [ a[0] if len(a)>0 else 'Unknow model' for a in aling_model ]

			file_Maling.close()  
			
			# - get dict label - #
			file_Maling = open( '{0}/{1}'.format(path, name), 'r' )
			aling_model_name = [ [m.split(':')[0].replace(" ", "") for i, m in enumerate(list(filter(lambda x: (len(x) > 1), n.split('\t'))) )] for n in file_Maling ]
			file_Maling.close() 
			
			return aling_model, aling_model_name

		def read_hp(path=None, name=None, ):
			path = path if type(path) == str else None
			name = name if type(name) == str else None

			file_hp = open( '{0}/{1}'.format(path, name), 'r' ) 
			hyperparameters_configuration = [ [m.split(':')[1].replace(" ", "") for i, m in enumerate(list(filter(lambda x: (len(x) > 1), n.split('\t'))) )] for n in file_hp ]
			file_hp.close()  
			
			file_hp = open( '{0}/{1}'.format(path, name), 'r' )
			hyperparameters_names = [ [m.split(':')[0].replace(" ", "") for i, m in enumerate(list(filter(lambda x: (len(x) > 1), n.split('\t'))) )] for n in file_hp ]
			file_hp.close() 

			return hyperparameters_configuration, hyperparameters_names

		def read_meanE(path=None, name=None, ):
			path = path if type(path) == str else None
			name = name if type(name) == str else None

			try:
				file_meanE = open( '{0}/{1}'.format(path, name), 'r' )
				mean_error = [ [ float(number) for number in list(filter(lambda x: (len(x) > 0) and 1==1, line.split('\t')))] for line in file_meanE ]
				file_meanE.close()
			except:
				print('ERROR :: SIMULATION.read.read_meanE() :: Can NOT read file {0}'.format( '{0}/{1}'.format(path, name) ) )

			return mean_error

		def read_Mtrainning(path=None, name=None, analysis_model_dict=None):
			path = path if type(path) == str else None
			name = name if type(name) == str else None
			analysis_model_dict = { 'PARAFAC2':'PARAFAC2', 'PARAFAC':'PARAFAC',  'None':'None', 'MCR':'MCR', 'GPV':'GPV',} if analysis_model_dict == None else analysis_model_dict 

			file_Mtrainning = open( '{0}/{1}'.format(path, name), 'r' ) 

			# - get entire model name list - #
			# eg. ['<analisys.MCR.MCRobjectat0x7fb3d28f3e48>', '<analisys.PARAFAC.PARAFACobjectat0x7fb3d0a3a9e8>', '<analisys.MCR.MCRobjectat0x7fb3d28f3e48>', '<analisys.PARAFAC.PARAFACobjectat0x7fb3d0a3a9e8>', '<analisys.MCR.MCRobjectat0x7fb3d28f3e48>']
			analysis_model = [ n.replace(" ", "")[:-1] for n in file_Mtrainning ]
			# - get match list - #
			# eg. [['MCR'], ['PARAFAC'], ['MCR'], ['PARAFAC'], ['MCR']]
			analysis_model = [ list(filter(lambda x: ( type(x) == str ), [am if am in a else False for am in analysis_model_dict])) for a in analysis_model ]
			# - get the largest elemen (it is more likelly to be correct) - #
			analysis_model = [ max(matchs , key=len) if len(matchs)>0 else 'Unknow model' for matchs in analysis_model ]

			file_Mtrainning.close()  
			
			return analysis_model

		def read_NRMSD(path=None, name=None, file=None):
			path = path if type(path) == str else None
			name = name if type(name) == str else None

			try:
				file_NRMSD = open( '{0}/{1}'.format(path, name), 'r' )
				NRMSD_error = [ [ float(number) for number in list(filter(lambda x: (len(x) > 0) and 1==1, line.split('\t')))] for line in file_NRMSD ]
				file_NRMSD.close()
			except:
				print('ERROR :: SIMULATION.read.read_NRMSD() :: Can NOT read file {0}'.format( '{0}/{1}'.format(path, name) ) )

			return NRMSD_error
			
		def read_RMSD(path=None, name=None, ):
			path = path if type(path) == str else None
			name = name if type(name) == str else None

			try:
				file_RMSD = open( '{0}/{1}'.format(path, name), 'r' )
				RMSD_error = [ [ float(number) for number in list(filter(lambda x: (len(x) > 0) and 1==1, line.split('\t')))] for line in file_RMSD ]
				file_RMSD.close()
			except:
				print('ERROR :: SIMULATION.read.read_RMSD() :: Can NOT read file {0}'.format( '{0}/{1}'.format(path, name) ) )

			return RMSD_error

		files_list = ['metafile.Maling', 'metafile.hp', 'metafile.meanE', 'metafile.Mtrainning', 'metafile.NRMSD', 'metafile.RMSD'] if files_list == None else files_list
		aling_model_dict = { 'FAPV32':'FAPV32',  'None':'None', 'ICOSHIFT':'ICOSHIFT',} if aling_model_dict == None else aling_model_dict
		analysis_model_dict = {  'PARAFAC2':'PARAFAC2', 'PARAFAC':'PARAFAC',  'None':'None', 'MCR':'MCR', 'GPV':'GPV',} if analysis_model_dict == None else analysis_model_dict 

		aling_model, aling_model_name = read_Maling(name='metafile.Maling', path=path, aling_model_dict=aling_model_dict)
		hyperparameters_configuration, hyperparameters_names = read_hp(name='metafile.hp', path=path)
		mean_error = read_meanE(name='metafile.meanE', path=path)
		analysis_model = read_Mtrainning(name='metafile.Mtrainning', path=path, analysis_model_dict=analysis_model_dict)
		NRMSD_error = read_RMSD(name='metafile.NRMSD', path=path)
		RMSD_error = read_NRMSD(name='metafile.RMSD', path=path)
		
		return	aling_model, aling_model_name, hyperparameters_configuration, hyperparameters_names, mean_error, analysis_model, NRMSD_error, RMSD_error

	def recursive_read_pandas(self, path, required_files=None, store='all', v=True, save=None, columns_name=True ):
		try: import pandas as pn
		except: print('ERROR :: SIMULATION.recursive_read_pandas() :: Can NOT import pandas. \n install with : "pip3 install pandas" ') 

		save = { key:item if not key in save else save[key] for key, item in {'mode':False, 'path':path, 'name':'{}.csv'.format(path.split('/')[-1]), 'porcentaje':100}.items() } if type(save) == dict else {'mode':False, 'path':path, 'name':'{}.csv'.format(path.split('/')[-1]), 'porcentaje':100}

		def check_files(file_list, required_files, criterion='all', file_name=None):
			check_list = [ True if r in file_list else False for r in required_files]

			if criterion=='all': 			check = check_list.count(True) == len(check_list)
			if criterion=='any false':		check = check_list.count(False) > 0
			if criterion=='any true': 		check = check_list.count(True) > 0

			return check

		required_files = required_files if type(required_files) != type(None) else ['metafile.Maling', 'metafile.hp', 'metafile.meanE', 'metafile.Mtrainning', 'metafile.NRMSD', 'metafile.RMSD']
		fullpath, folders, files = [[ x[n] for x in os.walk(path) ] for n in range(3)]

		#print('{}/{}'.format(save['path'], save['name']))
		#errorbar
		data_frame = open('{}/{}'.format(save['path'], save['name']), 'a')

		for i, f in enumerate(files):
			if v: print( 'completion {0:2.1f}%'.format( 100*float(i)/len(files) ) ) # verbosity 
			if check_files(file_list=f, required_files=required_files ):
				if v: print( fullpath[i], folders[i], files[i] ) 
				try:	aling_model, aling_model_name, hyperparameters_configuration, hyperparameters_names, mean_error, analysis_model, NRMSD_error, RMSD_error = self.read( path=fullpath[i] )
				except: print('ERROR :: SIMULATION.recursive_read_pandas() :: miss data in {}'.format( fullpath[i] ) )
				print
				if columns_name:	data_frame.write( '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format( 'Analysis model', 'Alignation model','File order', 'Internal orden', 'Analite order','NRMSD_error', 'RMSD_error', 'mean_error', '\t'.join(hyperparameters_names[0])) ); columns_name=False

				for j, data in enumerate(NRMSD_error):
					for k, value in enumerate(NRMSD_error[j]):
						try: data_frame.write( '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format( analysis_model[j], aling_model[j], i, j, k, NRMSD_error[j][k], RMSD_error[j][k], mean_error[j][k], '\t'.join(hyperparameters_configuration[j])) )
						except:	print('ERROR :: SIMULATION.recursive_read_pandas() :: can not write {}'.format( fullpath[i] ) )

				# -- Data integrity -- check if every element in f_data_list have the same len and restore data integrity
				#if not all([len(fdl_2) == np.mean([ len(fdl_1) for key1, fdl_1 in f_data_dict.items() ]) for key2, fdl_2 in f_data_dict.items()]): 
				#	for key, fdl in f_data_dict.items():		f_data_dict[key] = fdl[:min( [ len(fdl) for key, fdl in f_data_dict.items()] )]

				# -- early ending -- #
				if 100*float(i)/len(files) > save['porcentaje']: 	break

			elif v: print( 'WARNING :: missing files in {}'.format(fullpath[i]) )	
			
		if v: print( 'completion {0:2.1f}%'.format(100) ) # verbosity 
		data_frame.close()

		return True

	def recursive_read(self, path, required_files=None, store='all', v=True, save=None ):

		save = { key:item if not key in save else save[key] for key, item in {'mode':False, 'path':'', 'name':'', 'porcentaje':100}.items() } if type(save) == dict else {'mode':False, 'path':'', 'name':'', 'porcentaje':100}

		def check_files(file_list, required_files, criterion='all', file_name=None):
			check_list = [ True if r in file_list else False for r in required_files]

			if criterion=='all': 			check = check_list.count(True) == len(check_list)
			if criterion=='any false':		check = check_list.count(False) > 0
			if criterion=='any true': 		check = check_list.count(True) > 0

			return check

		def save_dict(dictionary, file_name):
			import pickle

			pickle_file = open(file_name, "wb")
			pickle.dump(dictionary, pickle_file)
			pickle_file.close()
			return 0

		def load_dict(file_name):
			import pickle

			pickle_file = open(file_name, "rb")
			dictionary = pickle.load(pickle_file)
			return dictionary

		required_files = required_files if type(required_files) != type(None) else ['metafile.Maling', 'metafile.hp', 'metafile.meanE', 'metafile.Mtrainning', 'metafile.NRMSD', 'metafile.RMSD']
		fullpath, folders, files = [[ x[n] for x in os.walk(path) ] for n in range(3)]
		data_dict = {}
		hyperparameters_configuration_space = {}

		for i, f in enumerate(files):
			if v: print( 'completion {0:2.1f}%'.format( 100*float(i)/len(files) ) ) # verbosity 

			if check_files(file_list=f, required_files=required_files ):
				if v: print( fullpath[i], folders[i], files[i] ) 
				aling_model, aling_model_name, hyperparameters_configuration, hyperparameters_names, mean_error, analysis_model, NRMSD_error, RMSD_error = self.read( path=fullpath[i] )
				#print( list(zip(analysis_model, aling_model, hyperparameters_configuration)) )
				#print( set([functools.reduce(lambda x,y: x+y, n) for n in hyperparameters_configuration])  )
				# -- generate the hash -- #
				hash_name = [a + b + functools.reduce(lambda x,y: x+y, c) for a,b,c in list(zip(analysis_model, aling_model, hyperparameters_configuration))]
				# -- Reduce repited hash -- #
				hash_name_list = set([a + b + functools.reduce(lambda x,y: x+y, c) for a,b,c in list(zip(analysis_model, aling_model, hyperparameters_configuration))])
				
				#f_data_list = [hash_name, aling_model, aling_model_name, hyperparameters_configuration, hyperparameters_names, mean_error, analysis_model, NRMSD_error, RMSD_error]
				# -- Data dict -- # contein all data from one file. Many possible configurations
				f_data_dict = 	{
							'hash_name'						: hash_name	,						
							'aling_model'					: aling_model	,						
							'aling_model_name'				: aling_model_name	,						
							'hyperparameters_configuration'	: hyperparameters_configuration	,						
							'hyperparameters_names'			: hyperparameters_names	,						
							'mean_error'					: mean_error	,						
							'analysis_model'				: analysis_model	,						
							'NRMSD_error'					: NRMSD_error	,						
							'RMSD_error'					: RMSD_error	,											
								}

				f_data_keylist = [	'mean_error',   'NRMSD_error',	'RMSD_error', ]

				f_data_keylist_once = [	'analysis_model', 'aling_model',  'aling_model_name', 'hyperparameters_configuration', 'hyperparameters_names']

				# -- Data integrity -- check if every element in f_data_list have the same len and restore data integrity
				if not all([len(fdl_2) == np.mean([ len(fdl_1) for key1, fdl_1 in f_data_dict.items() ]) for key2, fdl_2 in f_data_dict.items()]): 
					for key, fdl in f_data_dict.items():		f_data_dict[key] = fdl[:min( [ len(fdl) for key, fdl in f_data_dict.items()] )]

				# -- Prepare data_dict format -- # data_dict[key_hash][key_data]
				for key_hash in hash_name_list:
					if not key_hash in data_dict: data_dict[key_hash] = { key:[] for key in f_data_keylist+f_data_keylist_once }
					
				# -- Store data in data_dict -- # data_dict[key_hash][key_data] += [f_data_dict[key][key_i]]  
				for key_i, key_hash in enumerate(f_data_dict['hash_name']):
					for key in f_data_keylist:
						data_dict[key_hash][key] += [f_data_dict[key][key_i]] # use key to data_dict[hash_name]
					for key in f_data_keylist_once:
						if len( data_dict[key_hash][key] ) == 0: data_dict[key_hash][key] = f_data_dict[key][key_i]

				# -- Compress hyperparameters_configuration -- #
				for name, conf in zip(hyperparameters_names, hyperparameters_configuration):
					for snn, sn in enumerate(name):
						try:
							if sn in hyperparameters_configuration_space:
								if not conf[snn] in hyperparameters_configuration_space[sn]:
									hyperparameters_configuration_space[sn].append(conf[snn])
							else: 
								hyperparameters_configuration_space[sn] = [conf[snn]]
						except:
							print(name, conf)

				if save['mode'] in ['pkl', 'pickle'] and 100*float(i)/len(files) >= save['porcentaje']: 
					break
			
			elif v: print( 'WARNING :: missing files in {}'.format(fullpath[i]) )	
			

			if save['mode'] in ['pkl', 'pickle']: 
				save_dict(file_name='{}/{}.pkl'.format(save['path'], save['name']) , dictionary=data_dict)
				save_dict(file_name='{}/{}_hyperparameters_configuration_space.pkl'.format(save['path'], save['name']) , dictionary=data_dict)

		if v: print( 'completion {0:2.1f}%'.format(100) ) # verbosity 
		return {
			'hyperparameters_configuration_space':hyperparameters_configuration_space,
			'data_dict':data_dict,
			}

	def simulation_result_summary(self, data, hcs=None):
		# hcs == hyperparameters_configuration_space
		# data == first you must load data with recursive_read
		print( '{0} {1} {0} \n'.format('*'*20, 'Summary'*4) )
		if type(hcs) != type(None):
			print( ' -0- \t {0} Hyperparameters_configuration_space {0}'.format('*'*10) )
			for i, (key, item) in enumerate(hcs.items()):
				if len(item) < 3:
					text = ''
					for j, n in enumerate(item): text+=n+'\t,'
					print( ' -0.{}- {} : {}'.format(i, key, text) )
				else:
					print( ' -0.{}- {} '.format(i, key,) )	
					for j, n in enumerate(item):print( ' -0.{0}.{1}- {2}'.format(i,j,n) ) 

		for i, (key_data, item_data) in enumerate(data.items()):
			print(key_data)
			for j, (key_conf, item_conf) in enumerate(item_data.items()):	
				print(key_conf)

	def simulation_result_summary_pandas(self, file_path=None, df=None):
		try:	import pandas as pd
		except: print('ERROR :: SIMULATION.simulation_result_summary_pandas() :: can NOT load PANDAS. \n install pandas with : "pip install pandas"')

		def load_dataframe(file_path):
			tp = pd.read_csv(file_path, '\t', iterator=True, chunksize=10000)
			df = pd.concat(tp, ignore_index=True)
			return df 

		if type(file_path) != type(None):	df = load_dataframe(file_path)
		elif type(df) == type(None):		print('ERROR :: SIMULATION.simulation_result_summary_pandas() :: need DF path or DF as input.')

		print(' DataFrame shape : {}'.format(df.shape))
		for c, col in enumerate(df.columns):
			print('=== === {} === ==='.format(col))
			print(' [0] unique values list: {} \n{}'.format( len(df[col].unique()), df[col].unique() ))


		# == evaluate all posible paremeters combination == #
		inner_list = ['dimentions', 'N', 'factors', 'interferentes', 'aligned', 'functional', 'deformations', 'warping', 'sd', 'overlaping', 'noise_type', 'intencity']
		def extend(inner_list, n, configuration_space, element, forbidden):
			if n == len(inner_list): return configuration_space+[element]
			else:
				if not inner_list[n] == forbidden:
					for il in df[inner_list[n]].unique():	
						configuration_space = extend(inner_list, n+1, configuration_space, element+[il], forbidden)
				else:	configuration_space = extend(inner_list, n+1, configuration_space, element, forbidden)
				return configuration_space

		all_analisys_models = df['Analysis model'].unique()
		all_aling_models = df['Alignation model'].unique()

		# === Explore all interesting parameters === #
		for il in inner_list:
			# === Use parameters with some variace === #
			if 	len(df[il].unique()) > 3:
				print('\n\n {0}Particularity recognized : {1} {0}'.format('='*7, il))
				configuration_space = extend(inner_list, 0, [], [], il)
				inner_list_filter = list(filter(lambda x: x != il, inner_list ))
				print('configuration space size : {}'.format(len(configuration_space)) )

				# === exporer the configuration espace for 1 parameters === #
				# (all posible permutation of the other parameters)
				for cs in configuration_space:
					# === Plot the ones with less than 5 posible combinations === #
					if len(configuration_space) < 10:
						fig, ax = plt.subplots(len(all_analisys_models), len(all_aling_models), sharex=True, sharey=True, figsize=(len(all_analisys_models)*4, len(all_aling_models)*4), dpi=300)

						# == PLOT MATPLOTLIB == #
						for analisys_n, analisys_model in enumerate(all_analisys_models): 	# iters in all posible analisys models #
							for aling_n, aling_model in enumerate(all_aling_models):		# iters in all posible aling models #
								# === Explore the posible values of the parameter of interest === #
								color = iter(cm.inferno(np.linspace(0, 1, 10)))
								for il_values in df[il].unique()[:8]:							# iters in all posible values #
									print('analisys model:{} || aling model:{} '.format(analisys_model, aling_model) )

									# === apply all filter === #
									df_hist = df[df['Analysis model']==analisys_model][df['Alignation model']==aling_model][df['Analite order']==1][df[il]==il_values]
									for filter_name, filter_value in zip(inner_list_filter, cs):
										df_hist = df_hist[ df_hist[filter_name] == filter_value]

									# === select variable to plot === #
									df_hist = df_hist['NRMSD_error']

									# === plot in selected axis === #
									ax_s = ax[aling_n] if len(all_analisys_models) == 1 else ax[analisys_n] if len(all_aling_models) == 1 else ax[analisys_n][aling_n]
									ax_s.set_title( '{} + {}'.format(analisys_model, aling_model) )
									
									# === PLOT histogram === #
									c = next(color)
									ax_s.hist(df_hist, bins=500, color=c, range=[0,0.2], alpha=0.5, label='{}:{}'.format(il, str(il_values).split(',')[-1][:10] ))
									
								# === Plot legend (when there are less than 12) === #
								if len(df[il].unique()) < 21:
									if len(all_analisys_models) == 1:	ax[aling_n].legend() # add
									elif len(all_aling_models) == 1:	ax[analisys_n].legend()
									else:								ax[analisys_n][aling_n].legend()

						# === Add title and save figure === #
						fig.suptitle( str( ','.join(['{}:{}'.format(filter_name, filter_value) for filter_name, filter_value in zip(inner_list_filter, cs)]) ))
						plt.savefig('{}/{}_{}.png'.format('/'.join(file_path.split('/')[:-1]), cs, il_values), bbox_inches='tight', dpi=400)


	def simulation_result_summary_pandas_scatter(self, file_path=None, df=None, plot={'errorbar':False, }):
		try:	import pandas as pd
		except: print('ERROR :: SIMULATION.simulation_result_summary_pandas() :: can NOT load PANDAS. \n install pandas with : "pip install pandas"')
		try:	
			import seaborn as sns
			#sns.set_theme(style="dark")
		except: print('ERROR :: SIMULATION.simulation_result_summary_pandas() :: can NOT load seaborn. \n install pandas with : "pip install seaborn"')

		def load_dataframe(file_path):
			tp = pd.read_csv(file_path, '\t', iterator=True, chunksize=10000)
			df = pd.concat(tp, ignore_index=True)
			return df 


		if type(file_path) != type(None):	df = load_dataframe(file_path)
		elif type(df) == type(None):		print('ERROR :: SIMULATION.simulation_result_summary_pandas() :: need DF path or DF as input.')

		'''
		a = df[df['Analysis model']=='MCR']
		a = a[a['Alignation model']=='None']
		a = a[a['warping']=='[[0,0],[0.0,0.0],[0.2,0.2]]'][['NRMSD_error','overlaping']] 

		b = df[df['Analysis model']=='MCR']
		b = b[b['Alignation model']=='FAPV32']
		b = b[b['warping']=='[[0,0],[0.0,0.0],[0.2,0.2]]'][['NRMSD_error','overlaping']]
		print(df)

		d = {'col1': a.reset_index(drop=True)['NRMSD_error'], 'col2': b.reset_index(drop=True)['NRMSD_error'], 'overlaping':a.reset_index(drop=True)['overlaping']}
		df2 = pd.DataFrame(data=d)
		print(a)
		print(b)
		print(df2)
		g = sns.jointplot(x="col1", y="col2", hue='overlaping', data=df2,
													                  xlim=(-.01, 0.3), ylim=(-.01, 0.3), kind='scatter',
													                  color="m", height=7, s=3, palette="flare" )
		plt.show()
		'''
		print(' DataFrame shape : {}'.format(df.shape))
		for c, col in enumerate(df.columns):
			print('=== === {} === ==='.format(col))
			print(' [0] unique values list: {} \n{}'.format( len(df[col].unique()), df[col].unique() ))


		# == evaluate all posible paremeters combination == #
		inner_list = ['dimentions', 'N', 'factors', 'interferentes', 'aligned', 'functional', 'deformations', 'warping', 'sd', 'overlaping', 'noise_type', 'intencity']
		def extend(inner_list, n, configuration_space, element, forbidden):
			if n == len(inner_list): return configuration_space+[element]
			else:
				if not inner_list[n] == forbidden:
					for il in df[inner_list[n]].unique():	
						configuration_space = extend(inner_list, n+1, configuration_space, element+[il], forbidden)
				else:	configuration_space = extend(inner_list, n+1, configuration_space, element, forbidden)
				return configuration_space

		all_analisys_models = df['Analysis model'].unique()
		all_aling_models = df['Alignation model'].unique()

		# === Explore all interesting parameters === #
		for il in inner_list:
			# === Use parameters with some variace === #
			if 	len(df[il].unique()) > 3:
				print('\n\n {0}Particularity recognized : {1} {0}'.format('='*7, il))
				configuration_space = extend(inner_list, 0, [], [], il)
				inner_list_filter = list(filter(lambda x: x != il, inner_list ))
				print('configuration space size : {}'.format(len(configuration_space)) )

				# === exporer the configuration espace for 1 parameters === #
				# (all posible permutation of the other parameters)
				for cs in configuration_space:
					# === Plot the ones with less than 5 posible combinations === #
					if len(configuration_space) < 30:
						fig, ax = plt.subplots(len(all_analisys_models)*len(all_aling_models), len(all_analisys_models)*len(all_aling_models), 
										sharex=False, sharey=False, figsize=(len(all_analisys_models)*16, len(all_analisys_models)*16), dpi=600)

						# == PLOT MATPLOTLIB == #
						for analisys_n1, analisys_model1 in enumerate(all_analisys_models): 	# iters in all posible analisys models #
							for aling_n1, aling_model1 in enumerate(all_aling_models):		# iters in all posible aling models #
								
								for analisys_n2, analisys_model2 in enumerate(all_analisys_models): 	# iters in all posible analisys models #
									for aling_n2, aling_model2 in enumerate(all_aling_models):		# iters in all posible aling models #
										# === Diagonal superior de los datos === #
										if aling_n1 + analisys_n1*len(all_aling_models) >= aling_n2 + analisys_n2*len(all_aling_models):
											# === Explore the posible values of the parameter of interest === #
											color = iter(cm.inferno(np.linspace(0, 1, len(df[il].unique()[:14]) )))
											mean_values = [ [], [] ]
											sd_values = [ [], [] ]

											for il_values_n, il_values in enumerate(df[il].unique()[:12] ):							# iters in all posible values #
												print('analisys model:{} || aling model:{} '.format(analisys_model1, aling_model1) )

												# === apply all filter === #
												df_hist1 = df[df['Analysis model']==analisys_model1][df['Alignation model']==aling_model1][df['Analite order']==1][df[il]==il_values]
												for filter_name, filter_value in zip(inner_list_filter, cs):
													df_hist1 = df_hist1[ df_hist1[filter_name] == filter_value]
												
												df_hist2 = df[df['Analysis model']==analisys_model2][df['Alignation model']==aling_model2][df['Analite order']==1][df[il]==il_values]
												for filter_name, filter_value in zip(inner_list_filter, cs):
													df_hist2 = df_hist2[ df_hist2[filter_name] == filter_value]

												# === select variable to plot === #
												df_hist1 = df_hist1['NRMSD_error']
												df_hist2 = df_hist2['NRMSD_error']

												# === plot in selected axis === #
												index1 = aling_n1 + analisys_n1*len(all_aling_models)
												index2 = aling_n2 + analisys_n2*len(all_aling_models)
												ax_s = ax[index2][index1] # triangulo superior #
												ax_i = ax[index1][index2] # triangulo inferior #

												# === PLOT histogram === #
												c = next(color)

												min_shape = min(df_hist1.shape[0], df_hist2.shape[0])
												df_hist1 = df_hist1[:min_shape]
												df_hist1.index = np.arange(min_shape)
												df_hist2 = df_hist2[:min_shape]
												df_hist2.index = np.arange(min_shape)
												
												if aling_n1 + analisys_n1*len(all_aling_models) == aling_n2 + analisys_n2*len(all_aling_models):
													ax_s.hist(df_hist1, density=True, bins=50, color=c, range=[0,0.3], alpha=0.5, label='{}:{}'.format(il, str(il_values).split(',')[-1][:10] ))

												else:
													df_row = pd.concat([df_hist1, df_hist2], keys=['x', 'y'], axis=1)
													#ax_i.boxplot(df_hist1, positions=[il_values_n], ms=0.2 )
													ax_s.scatter(df_hist1, df_hist2,  color=c, alpha=0.4, s=2 )
													ax_i.scatter(df_hist2, df_hist1,  color=c, alpha=0.4, s=2 )

													#sns.boxplot(x='x', y='y', data=df_row, ax=ax_s)
													#sns.scatterplot(x='x', y='y', data=df_row, s=5, color=".15", alpha=0.6, ax=ax_s)
													#sns.histplot(x='x', y='y', data=df_row, bins=50, pthresh=.1, cmap="mako", ax=ax_s)
													#sns.kdeplot(x='x', y='y', data=df_row, levels=5, color="w", alpha=0.6, linewidths=0.5, ax=ax_s)
													# == set limits == # 
													
													mean_values[0].append( np.mean(df_hist1) )
													mean_values[1].append( np.mean(df_hist2) )
													sd_values[0].append( np.std(df_hist1) )
													sd_values[1].append( np.std(df_hist2) )

												if index1-1==len(all_aling_models): ax_i.set_xlabel('{} + {}'.format(analisys_model2, aling_model2))
												if index2==0:						ax_i.set_ylabel('{} + {}'.format(analisys_model1, aling_model1))
											
											# == set labels == # 
											if aling_n1 + analisys_n1*len(all_aling_models) != aling_n2 + analisys_n2*len(all_aling_models):
												ax_s.plot([0,0.3],[0,0.3], c=(0,0,0), alpha=0.5, ls=':', lw=4.0)

												if plot['errorbar']:
													ax_s.errorbar( mean_values[0], mean_values[1], yerr=sd_values[1], xerr=sd_values[0], 
																errorevery=2, capsize=6, linewidth=2.6, c=(0.4, 1.0, 0.4), alpha=0.7, )

												ax_s.plot(     mean_values[0], mean_values[1], '-o', 
																c=(0.4, 0.4, 1.0), alpha=0.3, ms=4.0, lw=5.0 )
												ax_s.plot(     mean_values[0], mean_values[1], 'o', 
																c=(0.2, 0.2, 1.0), alpha=0.3, ms=4.0, lw=5.0 )

											#ax_s.axes.xaxis.set_visible(False)
											#ax_s.axes.yaxis.set_visible(False)

											if index1 != index2:
												# ==== Set superior triangular matrix axis parameters ==== #
												ax_s.set_xticks([])
												ax_s.set_yticks([])
												ax_s.grid()
												ax_s.set_xlim(-0.00, 0.3)
												ax_s.set_ylim(-0.00, 0.3)
												
												# ==== Set inferior triangular matrix axis parameters ==== #
												ax_i.set_xticks([])
												ax_i.set_yticks([])
												ax_i.grid()
												ax_i.set_xlim(-0.00, 0.3)
												ax_i.set_ylim(-0.00, 0.3)
											else:
												# ==== Set diagonal axis parameters ==== #
												ax_s.set_xticks([])
												ax_s.set_yticks([])
												ax_s.grid()
												ax_s.set_xlim(-0.00, 0.3)
												ax_s.legend()

											# === Plot legend (when there are less than 12) === #
											#if len(df[il].unique()) < 21:
											#	if len(all_analisys_models) == 1:	ax[aling_n].legend() # add
											#	elif len(all_aling_models) == 1:	ax[analisys_n].legend()
											#	else:								ax[analisys_n][aling_n].legend()

						# === Add title and save figure === #
						#fig.suptitle( str( ','.join(['{}:{}'.format(filter_name, filter_value) for filter_name, filter_value in zip(inner_list_filter, cs)]) ))
						plt.savefig('{}/{}_{}.png'.format('/'.join(file_path.split('/')[:-1]), cs, il_values), bbox_inches='tight', dpi=400)




#g = sns.PairGrid(penguins)
#g.map_upper(sns.histplot)
#g.map_lower(sns.kdeplot, fill=True)
#g.map_diag(sns.histplot, kde=True)
def quela():
	# ==== cookbook chapter: Simulation summary ==== #
	simulation = SIMULATION()

	# = READ = #
	#file_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/OV_WA_all03'
	#simulation.recursive_read_pandas(file_path)

	# = ANALISYS = #
	file_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/CSV/noise00.csv'
	file_path = '/home/akaris/Documents/code/Chemometrics/files/simulations/CSV/OV_WA_all02.csv'

	import pandas as pd 
	import seaborn as sns
	def load_dataframe(file_path):
		tp = pd.read_csv(file_path, '\t', iterator=True, chunksize=10000)
		df = pd.concat(tp, ignore_index=True)
		return df 

	df = pd.read_csv(file_path, '\t', iterator=True, chunksize=10000)
	df = pd.concat(df, ignore_index=True)
	print( df.columns )
	print( df['warping'].unique() )
	print( df['overlaping'].unique() )
	print( df['Analysis model'].unique() )
	print( df['Alignation model'].unique() )
	df2 = pd.DataFrame()
	colors = [(1.0*float(n)/25,0,0) for n in range(25,1,-1)]

	data1mean = [] 
	data2mean = [] 

	for model in ['PARAFAC' 'MCR' 'PARAFAC2']:
		for aling in ['None' 'FAPV32' 'ICOSHIFT']:
			for i, var2 in enumerate(df['overlaping'].unique()[0:] ):
				print(var2)

				data1 = [] 
				data2 = [] 
				data3 = [] 

				vec1 = []
				vec2 = []

				for j, var1 in enumerate(df['warping'].unique()[0:]):
					df2 = df[df['Analysis model'] == 'MCR']
					df2 = df2[df2['Alignation model'] == 'None']
					df2 = df2[df2['warping'] == var1]
					df2 = df2[df2['overlaping'] == var2]
					df2 = df2[df2['Analite order'] == 1]
					df2 = df2[df2['NRMSD_error'] < 0.7]
					df2 = df2[df2['NRMSD_error'] > 0.0]
					df2.set_index(df2['Internal orden'])
					df2 = df2['NRMSD_error']
					df2 = list(df2)

					df3 = df[df['Analysis model'] ==     'PARAFAC']
					df3 = df3[df3['Alignation model'] == 'None']
					df3 = df3[df3['warping'] == var1]
					df3 = df3[df3['overlaping'] == var2]
					df3 = df3[df3['Analite order'] == 1]
					df3 = df3[df3['NRMSD_error'] < 0.7]
					df3 = df3[df3['NRMSD_error'] > 0.0]
					df3.set_index(df3['Internal orden'])
					df3 = df3['NRMSD_error']
					df3 = list(df3)
				
					data1 += df2[:min([len(df2), len(df3)])] 
					vec1 += [ np.mean(df2) ]

					data2 += df3[:min([len(df2), len(df3)])] 
					vec2 += [ np.mean(df3) ]

					data3 += [j]*min([len(df2), len(df3)]) 



		df4 = pd.DataFrame( np.array([data1, data2, data3]).T, columns = ['A1', 'A2', 'A3'] )

		g = sns.jointplot(x="A1", y="A2", hue='A3', data=df4,
	                  xlim=(-.01, 0.7), ylim=(-.01, 0.7), kind='scatter',
	                  color="m", height=7, s=5, palette="flare" , alpha=0.7)

		data1mean.append( vec1 )
		data2mean.append( vec2 )

		#fig = g.fig
		#ax = fig.axes
		#ax[0].plot(vec1, vec2, '-o', color=colors[i] )

		#plt.tight_layout()

		#plt.savefig(f'fig_{var1}.png', dpi=500,)
		#plt.show()

	data1mean = np.array( data1mean )
	data2mean = np.array( data2mean )

	plt.figure(100); plt.plot(data1mean, data2mean, '-o', color=colors[i] )
	plt.figure(100); plt.plot(data1mean.T, data2mean.T, '-o', color=colors[i] )

	plt.figure(110) 
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	X = np.array([ i for i, n in enumerate(df['warping'].unique()) ])
	Y = np.array([ i for i, n in enumerate(df['overlaping'].unique()) ])
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, data1mean, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)


	plt.figure(111) 
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	X = np.array([ i for i, n in enumerate(df['warping'].unique()) ])
	Y = np.array([ i for i, n in enumerate(df['overlaping'].unique()) ])
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, data2mean, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()

quela()







	#print(var1, var2)
#simulation.simulation_result_summary_pandas_scatter(file_path)


'''
g = sns.jointplot(x="FAPV + PARAFAC", y="PARAFAC", hue='OVERLAP', data=df,
                  xlim=(-.01, 0.5), ylim=(-.01, 0.5), kind='scatter',
                  color="m", height=7, s=3, palette="flare", alpha=0.3 )
'''


'''		
# ====== cookbook chapter: read simulation results with PANDAS ====== #
simulation = SIMULATION()
path = '/home/akaris/Documents/code/Chemometrics/files/simulations/CSV'
list_name = [	'shift02', 'shift01', 'shift00', 'PARAFAC2_1', 'PARAFAC2_0', 
				'overlaping', 'overlaping01', 'overlaping00', 'noise01', 'noise00', 
				'MCR_ICO_01', 'MCR_ICO_00', 'files', 'error' ] OVERLAP02 OVERLAP03
for name in list_name:
	simulation.recursive_read_pandas(path='/home/akaris/Documents/code/Chemometrics/files/simulations/{}'.format(name), save={'mode':False, 'path':path, 'name':'{}.csv'.format(name), 'porcentaje':101})
'''


