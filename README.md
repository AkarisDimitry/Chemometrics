Sure, here's the README.md content with appropriate formatting for GitHub:

---

```markdown
# 2D Ising Model Simulation with Metropolis Algorithm

## Installation Guide

### Install python 3.X

#### Linux

- **Check if you already have Python installed**
```bash
python --version
```

- **Step 1: Update and Refresh Repository Lists (optional)**
```bash
sudo apt update
```

- **Step 2: Install Supporting Software (optional)**
```bash
sudo apt install software-properties-common
```

- **Step 3: Add Deadsnakes PPA (optional)**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

- **Step 4: Install Python 3**
```bash
sudo apt install python3.8
```

- **Or FTP installation**
```bash
wget https://www.python.org/ftp/python/Python-3.9.1.tgz   
```

#### Windows

1. **Step 1: Select Version of Python to Install and Download Python Executable Installer**
- [Download Python for Windows](https://www.python.org/downloads/windows/)

2. **Step 2: Run Executable Installer**

3. **Step 3: Verify Python Was Installed On Windows (optional)**
- Navigate to the directory in which Python was installed on the system. In our case, it is `C:\Users\Username\AppData\Local\Programs\Python\Python37` since we have installed the latest version.
- Double-click python.exe.

4. **Step 4: Verify Pip Was Installed (optional)**
- Open the Start menu and type “cmd.”
- Select the Command Prompt application.
- Enter `pip -V` in the console. If Pip was installed successfully, you should see the appropriate output. If not, you might see the message: 
```
’pip’ is not recognized as an internal or external command, Operable program or batch file.
```

### INSTALLING NUMPY, scipy and matplotlib

- [Official Installation Guide](https://scipy.org/install.html)
  
Installation methods include:
- Distributions
- pip
- Package Manager
- Source
- Binaries

Methods differ in ease of use, coverage, maintenance of old versions, system-wide versus local environment use, and control. We recommend the Scientific Python Distribution [Anaconda](https://www.anaconda.com/products/distribution).

#### By CONDA (option 1)

```bash
# Best practice, use an environment rather than install in the base env
conda create -n my-env
conda activate my-env
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install numpy scipy matplotlib 
```

#### By PIP (option 2 – recommended for new python users)  
```bash
pip install --user numpy scipy matplotlib 
```

#### Install system-wide via a package manager (Ubuntu)
```bash
sudo apt-get install python-numpy python-scipy python-matplotlib
```

## Example Code

### Step 1 - Load python libraries:

```python
###########################################
# Step 1 || Load python libraries #
###########################################
# warning suppression
import warnings
warnings.filterwarnings("ignore")

# load numeric libraries 
import numpy as np

# load graph libraries 
import matplotlib.pyplot as plt

# load chemometric libraries (Juan Manuel Lombardi)
from alig import FAPV21
```

### Step 2 – Load your data (can be done in many different ways):

```python
###########################################
# Step 2 || Load data #
###########################################
# ******** DATA ******** #
# load calibration data
call_file = 'path/to/callfile'  # <span style="color:red">Specify your path here</span>
calibration_data =      np.array([ np.loadtxt( fname='/'.join(call_file.split('/')[:-1])+'/' +str(n),  delimiter=None) for i, n in enumerate([m[:-1] for m in open(call_file, 'r') if m != '' and m != ' ' and len(m) > 2 ]) ])
 
 # load test data
 call_file = 'path/to/testfile' # !!!!! Can be modified !!!!!
test_data = np.array([ np.loadtxt( fname='/'.join(call_file.split('/')[:-1])+'/' +str(n),  delimiter=None) for i, n in enumerate([m[:-1] for m in open(call_file, 'r') if m != '' and m != ' ' and len(m) > 2 ]) ])
 
 # join test and calibration data
 data = np.concatenate( (calibration_data, test_data), )
 print(data.shape)
# ******** LOAD spectra ******** #
 # load spectra 1 #
 spectra01 = np.loadtxt( fname='load your spectra estimation')[:,0] 	 # !!!!! Can be modified !!!!! it is not an strict requirement

# load spectra 2 #
spectra02 = np.loadtxt( fname='load your spectra estimation')[:,1] # !!!!! Can be modified !!!!! it is not an strict requirement
```

**Note**: In line 24, replace `'path/to/callfile'` with your actual path to the calibration samples.

### Step 3 – RUN FAPV21 (this is our algorithm):

```python
###########################################
# Step 3 || alignment process #
###########################################

 # ------ alignment proc ------ #
 # creating FAPV21 obj #
 FAPV21 = FAPV21() 
 S = [ np.array([spectra01, spectra02]) ] # !!!!! Can be modified !!!!! you can give FAPV21 spectra information
 
 # - Set system information - #
 dictionary = {				
		'D'		:	data,	
 		'Ni'		:	number_of_interferts  	,	
 		'Na'		:	number_of_analites  	,	
 		'Nc'		:	number_of_calibrationsample 	,	
 		'N'		:	number_of_samples 	,
		'Nt'		:	number_of_testsamples	,	
 		'a'		:	number_of_wellalignedchannels  	,	
 		'S'		:	S  	, 	
 		}					#################################################
 
 FAPV21.import_data_dict(dictionary=dictionary) # import data into the models
 
 # Aligned proc 
 Da, a, b = FAPV21.aling	( 	
 					area			= 'gaussian_coeficient'	,
 					mu_range		= np.arange(150, 300, 3)	, 
 					non_negativity	= False			,
 					sigma_range		= np.arange(4, 20, 0.7)	,
 					SD			= {'mode':'constant'}	,	
				)		 # solve the model

# after this the well aligned data will be in ‘Da’ you can process with parafac, mcr-als, etc							 									
```

Line 65 call the FAPV21.aling method ( with the parameters given in lines 66 to 71).

area : Mode in which the area is preserved. (default='gaussian_coeficient')

mu_range : default=None (very wide and highly dense space exploration can be time consuming in some cases.). It is highly recommended to conveniently define the region to be explored and increase the density of the grid.

sigma_range : default=None ( very wide and highly dense space exploration can be time consuming in some cases.). It is highly recommended to conveniently define the region to be explored and increase the density of the grid.

Variable space mu and sigma. This parameter contains the values that allow generating a bounded functional subspace to explore. These parameters are used in a Gaussian function space, other types of functional spaces may depend on other parameters.

SD : Defines a formal method to evaluate a good standard deviation reference (All aigen function will considerer ass a well aligned reference). (default : {'mode':'constant'})

All posible inputs to be cosiderer in FAPV21.aling() method:

D=None
S=None
Nc=None
Na=None
Ni=None
Nt=None
a=None
shape='gaussian',
area='gaussian_coeficient'
non_negativity=True
SD={'mode':'mean'}
interference_elimination=False
mu_range=None
sigma_range=None 
save=True
v=1, 

### Important Note

It is imperative to correctly set all the input parameters. The user must know their data and have enough mathematical knowledge to define all the input variables.
```

---

Please replace the placeholders like `'path/to/callfile'` with your actual paths and data. The colored text in markdown for GitHub is achieved using HTML tags; however, they may not render in all markdown renderers. For the best compatibility, you might want to use GitHub's diff syntax to highlight lines of code.
