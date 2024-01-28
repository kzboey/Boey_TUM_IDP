# Boey_TUM_IDP
Explored the application of physics-based deep learning for fast inference of molecular composition of brain tissue from spectroscopy- based imaging modality such as hyperspectral imaging and near- infrared spectroscopy. This method is based on the paper "Shallow learning enables real-time inference of molecular composition changes from broadband-near-infrared spectroscopy of brain tissue" by IVAN EZHOV, LUCA GIANNONI, IVAN ILIASH, FELIX HSIEH, CHARLY CAREDDA, FRED LANGE, ILIAS TACHTSIDIS, AND DANIEL RUECKERT.

1. optimisation_diff_batch.ipynb - Demonstrate the method on the piglet dataset. Tissue concentration of "HbO2", "Hbb", "oxyCCO", "redCCO" are derived from the spectra using the cvpx library as ground truth for the neural network.

2. inference_time.ipynb - Various neural netork architecure are training, their performance is compared in this notebook to rank them.

3. optimisation_diff_batch_simulation.ipynb - Demonstrate our method on the mouse dataset that is simulated using monte carlo method. We employed a small neural network with 2 layers and 256 hidden units. The plotting shows the activation map of each of the 4 tissues concentration predicted, one predicted using the neural network and another the groud truth derived from the optimization method.

4. preprocessing.py - To generate the molecular coefficient of the 4 tissue molecules by interpolating the input spectrato generate the system matrix for the optimization method.

5. train_mouse.py - Training script for the mouse dataset demonstrated in optimisation_diff_batch_simulation.ipynb.

6. train_piglet.py - Training script for the piglet dataset demonstrated in optimisation_diff_batch.ipynb.
