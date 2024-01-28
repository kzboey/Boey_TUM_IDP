import os
import scipy.io
import cvxpy as cp
import seaborn as sns
import numpy as np
import time

import data_processing.preprocessing as preprocessing
from config import left_cut, right_cut
import utils

"""
Script the runs cvxpy optimisation, to find the param differences between two spectra
Last line saves the result for later use in NN training.
"""

sns.set()

# extract file names
folder = os.listdir("dataset/miniCYRIL-Piglet-Data")
folder = [i for i in folder if "lwp" in i or "LWP" in i]
dic = {}
for i in folder:    
    files = os.listdir('dataset/miniCYRIL-Piglet-Data/'+i)
    files = [i for i in files if ".mat" in i]
    files = [i for i in files if not any(excluded in i for excluded in 
                    ["Results", "DarkCount", "ref", "output", "markcope", "Abs", "lwp"])]
    if files == []: continue
    dic[i] = files

def optimisation(spectr1, spectr2, wavelengths):
    m = 4  # number of parameters (from 2 to 6)
    #np.random.seed(1)
    molecules, x = preprocessing.read_molecules(left_cut, right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa),
                                np.asarray(y_creda))))
    
    b = spectr2 / spectr1
    b = np.log(1 / np.asarray(b))  # see the writting above (we took log of I_R and there was also minus that went to the degree of the logarithm)
    X = cp.Variable((m, len(b)))
    b = np.swapaxes(b, 0, 1)
    
    print(M.shape, X.shape, b.shape)
     
    objective = cp.Minimize(cp.sum_squares(M @ X - b))
    constraints = [cp.abs(X[2]+X[3])<=0.01]
    prob = cp.Problem(objective, constraints)
    
    start = time.time()
    result = prob.solve()
    print("Time:", time.time() - start)

    return -X.value, b


def get_dataset(spectr, dark_full, white_full, wavelengths, minimum, maximum, pig, date):
    ref_spectr = (spectr[:, minimum] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
    ref_spectr[ref_spectr <= 0] = 0.0001

    spectra_list = []
    coef_list = []

    comp_spectr = np.array([(spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0]) for i in range(minimum+1,maximum+1)])
    comp_spectr[comp_spectr <= 0] = 0.0001

    #comp_spectr = np.array(comp_spectr.tolist() * 11)

    coef_diff, spect_diff = optimisation(ref_spectr, comp_spectr, wavelengths)

    spectra_list.append(spect_diff)
    coef_list.append(coef_diff)
    utils.save_optimization_data(ref_spectr, spectra_list, coef_list, str(pig)+'_'+str(date))
    
for pig in dic.keys():
    for index, date in enumerate(dic[pig]):  
        print(pig, date)
        img = scipy.io.loadmat('dataset/miniCYRIL-Piglet-Data/' + pig + '/' + date)

        wavelengths = img['wavelengths'].astype(float)
        white_full = img['refIntensity'].astype(float)
        dark_full = img['darkcount'].astype(float)
        spectr = img['spectralDataAll'].astype(float)

        idx = (wavelengths >= left_cut) & (wavelengths <= right_cut)
        wavelengths = wavelengths[idx]
        spectr = spectr[idx.squeeze()]
        white_full = white_full[idx.squeeze()]

        print(white_full.shape, dark_full.shape, wavelengths.shape, spectr.shape)  
        minimum, maximum = None, None
              
        for i in range(spectr.shape[1]):
            spectr_1 = (spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
            dif = max(spectr_1) - min(spectr_1)
            if dif >= 0.1:
                minimum = i
                break

        for i in reversed(range(spectr.shape[1])):
            spectr_2 = (spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
            dif = max(spectr_2) - min(spectr_2)
            if dif >= 0.1:
                maximum = i
                break
        
        print("Min and Max:", minimum, maximum)
        if minimum is not None and maximum-minimum >= 100: get_dataset(spectr, dark_full, white_full, wavelengths, minimum, maximum, pig, index)
        else: print("skipped this sample")
        print()