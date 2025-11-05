#HW8Fun.py
#import relevant packages
import os
import numpy as np
import scipy.io as sio
#This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf #This will be used to create a PDF to store multiple plots in the same file
#You may not use this function for HW7, but when we start to move everything to .py file, you will use it.

#produce_trun_mean_cov()
def produce_trun_mean_cov(input_signal, input_type, E_val):
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)
        E_val: integer, (number of electrodes)

    return:
    -----
        A list of 5 arrays:
            signal_tar_mean: (E_val, length_per_electrode)
            signal_ntar_mean: (E_val, length_per_electrode)
            signal_tar_cov: (E_val, length_per_electrode, length_per_electrode)
            signal_ntar_cov: (E_val, length_per_electrode, length_per_electrode)
            signal_all_cov: (E_val, length_per_electrode, length_per_electrode)
    """
    total_features = input_signal.shape[1]
    length_per_electrode = total_features // E_val

    signal_tar = input_signal[input_type == 1, :]
    signal_ntar = input_signal[input_type !=1, :]

    signal_tar_mean = np.zeros((E_val, length_per_electrode))
    signal_ntar_mean = np.zeros((E_val, length_per_electrode))
    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    for e in range(E_val):
        start = e * length_per_electrode
        end = (e + 1) * length_per_electrode

        sig_tar_e = signal_tar[:, start:end]
        sig_ntar_e = signal_ntar[:, start:end]
        sig_all_e = input_signal[:, start:end]

        signal_tar_mean[e, :] = np.mean(sig_tar_e, axis=0)
        signal_ntar_mean[e, :] = np.mean(sig_ntar_e, axis=0)

        signal_tar_cov[e, :, :] = np.cov(sig_tar_e, rowvar=False)
        signal_ntar_cov[e, :, :] = np.cov(sig_ntar_e, rowvar=False)
        signal_all_cov[e, :, :] = np.cov(sig_all_e, rowvar=False)

    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]

#plot_trunc_mean()
def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit=np.array([-5, 8]), fig_size=(12, 12)
):
    r"""
    Plot sample means for target and non-target ERPs for each electrode.

    :param eeg_tar_mean: array, (E_val, length_per_electrode)
    :param eeg_ntar_mean: array, (E_val, length_per_electrode)
    :param subject_name: string, name of the subject
    :param time_index: array, time points for x-axis
    :param E_val: integer, number of electrodes
    :param electrode_name_ls: list of electrode names
    :param y_limit: optional, y-axis limits [min, max]
    :param fig_size: optional, figure size
    :return: None (displays the plot)
    """
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"{subject_name}: Target vs Non-Target ERP means", fontsize=16, y=0.95)

    for e in range(E_val):
        ax = axes[e // 4, e % 4]
        ax.plot(time_index, eeg_tar_mean[e, :], color='red', label='Target')
        ax.plot(time_index, eeg_ntar_mean[e, :], color='blue', label='Non-Target')
        ax.set_ylim(y_limit)
        ax.set_title(electrode_name_ls[e], fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        if e == 0:
            ax.legend()

    plt.tight_layout()
    save_dir = os.path.join(os.getcwd(), "K114")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Mean.png"), dpi=300)
    plt.close()


#plot_trunc_cov()
def plot_trunc_cov(
        eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, fig_size=(14,12)
):
    X, Y = np.meshgrid(time_index, time_index)

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f"{subject_name}: {cov_type} Covariance", fontsize=16, y=0.95)

    for e in range(E_val):
        ax = axes[e // 4, e % 4]
        c = ax.contourf(X, Y, eeg_cov[e, :, :], cmap='viridis')
        ax.set_title(electrode_name_ls[e], fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Time (ms)')
        plt.colorbar(c, ax=ax, shrink=0.7)

    plt.tight_layout()
    save_dir = os.path.join(os.getcwd(), "K114")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"Covariance_{cov_type}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()