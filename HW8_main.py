#HW8_main.py
#import relevant packages
import os
import numpy as np
import scipy.io as sio
#This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf #This will be used to create a PDF to store multiple plots in the same file
#You may not use this function for HW7, but when we start to move everything to .py file, you will use it.

from self_py_fun.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

#Global variables
bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

parent_dir = '/Users/gracebrewer/Documents/GitHub/BIOS-584'
parent_data_dir = f'{parent_dir}/data'
time_index = np.linspace(0, 800, 25) #This is a hypothetic time range up to 800 ms after each stimulus.
subject_name = 'K114'
session_name = '001_BCI_TRN'

eeg_trunc_obj = sio.loadmat(os.path.join(parent_data_dir, 'K114_001_BCI_TRN_Truncated_Data_0.5_6.mat'))

eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = eeg_trunc_obj['Type']
eeg_trunc_type = np.squeeze(eeg_trunc_type, axis=1)

E_val = electrode_num
input_signal = eeg_trunc_signal
input_type = eeg_trunc_type

#call produce_trun_mean_cov()
eeg_tar_mean, eeg_ntar_mean, eeg_tar_cov, eeg_ntar_cov, eeg_all_cov = produce_trun_mean_cov(input_signal, input_type, E_val)

#plot and save figures
plot_trunc_mean(eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls)
plot_trunc_cov(eeg_tar_cov, 'Target', time_index, subject_name, E_val, electrode_name_ls)
plot_trunc_cov(eeg_ntar_cov, 'Non-Target', time_index, subject_name, E_val, electrode_name_ls)
plot_trunc_cov(eeg_all_cov, 'All', time_index, subject_name, E_val, electrode_name_ls)