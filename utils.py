import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import medfilt, savgol_filter, wiener
import os

import config


def beerlamb_multi(molecules, wavelength_range, a_all, left_cut):
    """
        Computes Beer-Lambert Law for multiple molecules.
        molecules: molecule absorption spectra. use the output of preprocessing.read_molecules() here
        wavelength_range: available wavelengths
        a_all: parameters (molecule concentrations)
        left_cut: left side cutoff of wavelengths
    """
    y = []
    for i in wavelength_range:
        b_avg = 0.86
        g_avg = 0.9
        m_a = 0
        step = int(i - left_cut)

        for j in range(len(a_all)):
            m_a += a_all[j] * molecules[j][step]

        mu_s_avg = np.power(np.divide(i, 500), -b_avg)
        mu_s_avg = mu_s_avg / (1 - g_avg)

        exp_m = np.exp(-(m_a))  # + mu_s_avg

        sp = exp_m ** 2  # signal Power
        snr = 8000  # signal to noise ratio
        std_n = (sp / snr) ** 0.5  # noise std. deviation

        y.append(exp_m + np.random.normal(0, std_n, 1)[0])

    return y  # I = I_o * exp(-(m_a+m_s)x)


def beerlamb_multi_batch(molecules, wavelength_range, a_all, left_cut):
    """
    beerlamb_multi but with batched input
    """
    y = np.empty((len(wavelength_range), a_all.shape[1]))
    for idx, i in enumerate(wavelength_range):
        b_avg = 0.86
        g_avg = 0.9
        m_a = 0
        step = int(i - left_cut)

        for j in range(len(a_all)):
            m_a += a_all[j] * molecules[j][step]

        mu_s_avg = np.power(np.divide(i, 500), -b_avg)
        mu_s_avg = mu_s_avg / (1 - g_avg)

        exp_m = np.exp(-(m_a))  # + mu_s_avg

        sp = exp_m ** 2  # signal Power
        snr = 8000  # signal to noise ratio
        std_n = (sp / snr) ** 0.5  # noise std. deviation

        y[idx] = exp_m + np.random.normal(0, std_n, exp_m.shape)

    return y  # I = I_o * exp(-(m_a+m_s)x)


def get_mean_std(dataset):
    return torch.mean(dataset), torch.std(dataset)


def plot_pred(opti_coef, nn_coef, name):
    """
    Used during training to plot prediction results
    """
    opti_coef = opti_coef.cpu().detach().numpy()
    nn_coef = nn_coef.cpu().detach().numpy()

    torch.save(opti_coef, 'results/{}/coef_Opti.pt'.format(name))
    torch.save(nn_coef, 'results/{}/coef_NN.pt'.format(name))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    coef = ["HbO2", "Hbb", "oxyCCO", "redCCO"]
    for i in range(opti_coef.shape[1]):
        plt.plot(opti_coef[:, i], color=colors[i], label=f'Opti ' + coef[i], linewidth=3)
        plt.plot(nn_coef[:, i], color=colors[i], label=f'NN ' + coef[i], linewidth=0.5)

    plt.legend()
    plt.savefig("results/{}/fig.png".format(name), format='png')
    plt.clf()


def median_fil(data, window_size=17):
    window_size = 17  # Adjust as needed
    return medfilt(data, kernel_size=window_size)


def moving_avg_fil(data, window_size=17):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def savgol_fil(data, window_size=17, polyorder=1):
    return savgol_filter(data, window_length=window_size, polyorder=polyorder)


def wiener_fil(data, window_size=17):
    return wiener(data, mysize=window_size)


def save_optimization_data(ref_spectr, spectra_list, coef_list, folder):
    """
    Used after running the optimisation script/notebook to be used later by the Neural Network
    """
    spectra_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(spectra_list))), 0, 1)
    coef_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(coef_list))), 0, 1)
    path = config.dataset_path + 'piglet_diffs_v2/' + folder
    if not os.path.exists(path): os.mkdir(path)
    split_point = int(len(spectra_list) * 0.85)
    torch.save(ref_spectr, path + '/ref_spectr.pt')
    torch.save(spectra_list[:split_point], path + '/spectra_list_train.pt')
    torch.save(coef_list[:split_point], path + '/coef_list_train.pt')
    torch.save(spectra_list[split_point:], path + '/spectra_list_test.pt')
    torch.save(coef_list[split_point:], path + '/coef_list_test.pt')
    torch.save(spectra_list, path + '/spectra_list.pt')
    torch.save(coef_list, path + '/coef_list.pt')
    
def save_optimization_data2(ref_spectr, spectra_list, coef_list, folder):
    """
    Used after running the optimisation script/notebook to be used later by the Neural Network
    """
    spectra_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(spectra_list))), 0, 1)
    # coef_list = torch.swapaxes(torch.squeeze(torch.from_numpy(np.array(coef_list))), 0, 1)
    path = config.dataset_path + 'piglet_diffs_sim2/' + folder #config.dataset_path + 'piglet_diffs/' + folder
    if not os.path.exists(path): os.mkdir(path)
    torch.save(spectra_list, path + '/spectra_list.pt')
    torch.save(coef_list, path + '/coef_list.pt')
    
def save_optimization_data3(spectra, coef, folder):
    """
    Used after running the optimisation script/notebook to be used later by the Neural Network
    """
    path = config.dataset_path + 'mouse_diffs/' + folder
    if not os.path.exists(path): os.mkdir(path)
    torch.save(spectra, path + '/spectra.pt')
    torch.save(coef, path + '/coef.pt')
