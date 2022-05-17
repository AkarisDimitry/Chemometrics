import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import random, time, sys,  pickle

if sys.version_info[0] == 3:    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
else:   # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import matplotlib.animation as animation
from matplotlib.figure import Figure

class aling_model(tk.Frame):
    def __init__(self, data, parent):
        self.data = data

        # ----- LABEL organization ----- #
        self.label_top = tk.Label(parent, font=("Verdana", 15) )
        self.label_top.pack(side=TOP, expand=1, fill=X, pady=0, padx=0)

        self.label_bottom = tk.Label(parent, font=("Verdana", 15) )
        self.label_bottom.pack(side=BOTTOM, expand=1, fill=X, pady=0, padx=0)

        # -- tabs acting over rigth label -- #
        self.tab_parent = ttk.Notebook(self.label_bottom, )
        self.tab00 = ttk.Frame(self.tab_parent) # data visualization #
        self.tab01 = ttk.Frame(self.tab_parent) # data aligment #
        self.tab02 = ttk.Frame(self.tab_parent) # data aligment #

        self.tab_parent.add(self.tab00, text='Model configuration')         
        self.tab_parent.add(self.tab01, text='Aliniation analisys') 
        self.tab_parent.add(self.tab02, text='Result') 
        self.tab_parent.pack(side=TOP, expand=1, fill='both')

        # **** **** TAB00/ **** **** #
        self.label_tab00_model_top = tk.Label(self.tab00, )
        self.label_tab00_model_top.pack(fill=X, side=TOP, pady=1,padx=1)

        self.label_tab00_model_selection = tk.Label(self.label_tab00_model_top, font=("Verdana", 15) )
        self.label_tab00_model_selection.pack(side=TOP, fill=X, pady=0, padx=0)

        self.label_tab00_model_options = tk.Label(self.label_tab00_model_top, font=("Verdana", 15) )
        self.label_tab00_model_options.pack(side=TOP, fill=X, pady=0, padx=0)
        # **** **** /TAB00 **** **** #

        # **** **** TAB01/ **** **** #
        self.label_tab01_model_calibration = tk.Label(self.tab01, )
        self.label_tab01_model_calibration.pack(fill=X, side=TOP, pady=1,padx=1)

        self.label_tab01_model_calibration_top = tk.Label(self.label_tab01_model_calibration, font=("Verdana", 15) )
        self.label_tab01_model_calibration_top.pack(side=TOP, fill=X, pady=0, padx=0)

        self.label_tab01_model_calibration_bot = tk.Label(self.label_tab01_model_calibration, font=("Verdana", 15) )
        self.label_tab01_model_calibration_bot.pack(side=TOP, fill=X, pady=0, padx=0)
        # **** **** TAB01/ **** **** #

    def info_align_actualization(self, event=None, parent=None, a=2):
        for n in range(a):
            loadings_stimation_plot(mode=n)

    def loadings_stimation_plot(self, event=None, parent=None, mode=None): 

        def act_plot(self, ):
            try:
                fig.clear()
                data.plot_loadings(aligned=False, mode=[mode], fig=fig)
                canvas.draw()
            except: print('ERROR :: code V000 MAIN.data_aligment.plot_mode.act_plot() :: ')

        def loading_estimate( ):
            if combo_stimate.get() == 'PARAFAC':
                parafac = PARAFAC()
                data.inyect_data(parafac)
                data.L[mode] = parafac.loading_stimation()[mode].T
                act_plot(self,)

        def open_loading_file(event=None):
            global data
            input_file_name = tk.filedialog.askopenfilename(defaultextension=".txt",
                                                            filetypes=[("All Files", "*.*"), ("Text Documents", "*.txt")])
            if input_file_name:
                file_name = input_file_name

                data.L[mode] = np.loadtxt( file_name )
                act_plot(self, )

        def save_loading_file(event=None):
            try:
                global data
                input_file_name = tkinter.filedialog.asksaveasfilename(defaultextension=".txt",
                                                       filetypes=[("All Files", "*.*"), ("Text Documents", "*.txt")])
                if input_file_name:
                    figure_new, data_new = data.plot_loadings(aligned=False, mode=[mode], fig=None)
                    file_name = input_file_name
                    data.save_text(file=input_file_name, data=data_new, delimiter='\t')

            except: print('ERROR :: code V001 MAIN.data_aligment.plot_mode.save_loading_file() :: ')

        if type(data.X) is np.ndarray:
            #try:
                # ---- General FONT ---- #
                font00 = ("Verdana", 10)
                factor_selection = StringVar()
                mode_selection = StringVar()
                plot_selection = StringVar()
                
                # ****** Label to store all plots ****** #
                label_general = tk.Label(self.label_tab01_model_calibration_top )
                label_general.pack(pady=1,padx=1, side=LEFT)

                label_plot_top = tk.Label(label_general )
                label_plot_top.pack(pady=1,padx=1, side=TOP)

                label_bottom = tk.Label(label_general )
                label_bottom.pack(pady=1,padx=1, fill=X, side=BOTTOM)

                label_info_rigth = tk.Label(label_general )
                label_info_rigth.pack(pady=1,padx=1, side=RIGHT)

                label_info_left = tk.Label(label_general )
                label_info_left.pack(pady=1,padx=1, side=LEFT)



                # ****** remove buttom ****** #
                self.button_erase = tk.Button(label_bottom, text="Remove",
                                       command=label_general.destroy, bg='#AAAAAA', font=("Verdana", 10))
                self.button_erase.pack(fill=X, side=BOTTOM  ) 

                # ****** LOAD buttom ****** #
                self.button_load = tk.Button(label_bottom, text="Load",
                                       command=open_loading_file, bg='#AAAAAA', font=("Verdana", 10))
                self.button_load.pack(fill=X, side=BOTTOM  ) 

                # ****** SAVE buttom ****** #
                self.button_save = tk.Button(label_bottom, text="Save",
                                       command=save_loading_file, bg='#AAAAAA', font=("Verdana", 10))
                self.button_save.pack(fill=X, side=BOTTOM  ) 

                # ****** ESTIMATE buttom ****** #
                self.button_stimate = tk.Button(label_info_left, text="Estimate",
                                       command=loading_estimate, bg='#AAAAAA', font=("Verdana", 10))
                self.button_stimate.pack(fill=X, side=RIGHT ) 

                # ****** COMBO model buttom ****** #
                combo_stimate = ttk.Combobox(label_info_rigth, state="readonly")
                combo_stimate["values"] = [ 'PARAFAC', 'MCR', 'GPV', ]
                combo_stimate.bind("<<ComboboxSelected>>", )
                combo_stimate.current(0)
                combo_stimate.pack(pady=0,padx=1,fill=X, side=BOTTOM ) 

                # ****** PLT GRAPH ****** # in label_general
                fig = Figure(figsize=(5, 4), dpi=100) # figure size and resolution # 

                fig_plot = fig.add_subplot(111)
                fig.axes[0].format_coord = lambda x, y: ""

                canvas = FigureCanvasTkAgg(fig, label_plot_top)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.BOTTOM, expand=True)

                toolbar = NavigationToolbar2Tk(canvas, label_plot_top,)
                toolbar.update()
                canvas._tkcanvas.pack(fill=X, side=tk.TOP, expand=True )
                # *********************************************** #             

            #except:  print('ERROR :: code V000 MAIN.add_plot() :: can NOT plot')


    def run_model(self):
        global data 


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

    def exit(self, event=None):
        if tkinter.messagebox.askokcancel("Quit?", "Do you want to QUIT for sure?\n Make sure you've saved your current work."):
            self.controller.destroy()