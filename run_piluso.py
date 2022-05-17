import os

def make_cluster_bash(file_name, job_name, job_path,time):
	try:
		f = open(str(file_name), 'w')
	except:
		print('ERROR :: run_piluso.make_cluster_bash() :: File {0} not accessible'.format(str(file_name)) )

	list_string =[
			'#!/bin/bash'	,
			'#$ -S /bin/bash ' 	,
			'### Job Name' 	,
			'#$ -N {0}'.format('Chemometric-simulation-JML_{0}'.format(job_name) ) 	,
			'#'	,
			'###### Add the $ symbol after # only in the desired SGE option'	,
			'#'	,
			'# Setea HH:MM:SS tiempo de wall clock time, maximo 3 dias'	,
			'#'	,
			'#  Number of cores'	,
			'#$ -pe impi-node 1'	,
			'#'	,
			'### Write output files .oxxxx .exxxx in current directory'	,
			'#$ -cwd' 	,
			'#' 	,
			'### Merge \'-j y\' (do not merge \'-j n\') stderr into stdout stream:'	,
			'#$ -j y' 	,
			'#'	,
			'#  SGE environment variables'	,
			'#$ -V'	,
			'# seteamos el wall time'	,
			'#$ -l h_rt={0}'.format(time)	,
			'#'	,
			'#'	,
			'### Select a queue'	,
			'###$ -q  \'colisiones\''	,
			''	,
			'### module load intel-13'	,
			'### module load impi-4.0.0.028'	,
			'### IMPI_HOME=/share/apps/intel/impi/4.0.0.028/intel64/bin'	,
			''	,
			'module load  python3.8'	,
			''	,
			'# -------- SECTION print some infos to stdout ---------------------------------'	,
			''	,
			'python3 main_simulation.py {0}'.format(job_path)	,]
	
	for i, n in enumerate(list_string):
		f.write(n+'\n')

def make_job(job_name, save_path,time='248:00:00'):
	if not type(job_name) == str:
		print('ERROR :: run_piluso.make_job() :: job_name must be str.')
	else:
		make_cluster_bash(	file_name=	'run_{0}.sh'.format(job_name), 
							job_name=	'{0}'.format(job_name),
							job_path=	'{0}/{1}'.format(save_path, job_name), 
							time=		'600:00:00')

		os.system('mkdir {0}/{1}'.format(save_path, job_name) )
		os.system('qsub run_{0}.sh'.format(job_name) )
		
		#print('{0}/{1}'.format(save_path, job_name) )
		#os.system('python3 main_simulation.py {0}/{1}'.format(save_path, job_name))

for n in range(1):
	print( 'simulation0{0}'.format(str(n)))
	make_job(job_name='simulation0{0}'.format(str(n+100)), save_path='cases/files',time='600:00:00')