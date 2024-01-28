import os
import time

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_processing.datasets import PigletDataset, SpectraDataset, MouseDataset
from neuralnet.model import SpectraMLP
import config
import wandb

# wandb.login(key=config.key)
# wandb.init(
#     project='Concentration of Molecules (NN model)',
#     config={
#         "epochs": config.epochs,
#         "batch_size": 128,
#         "lr": config.lr,
#     }
# )


def train_mouse(n_layers=4, layer_width=1024, epochs=20):
    """
    created to train on datasets in dataset/piglet_diffs
    results are saved in the results folder, for each n_layers/layer_width setup seperately
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = config.lr

    name = str(n_layers) + "_" + str(layer_width)
    save_model_path = "results_mouse/{}".format(name)
    if not os.path.exists(save_model_path): os.mkdir(save_model_path)
    save_path = 'results_mouse/{}/best_model.pth'.format(name)

    model = SpectraMLP(config.molecule_count, n_layers=n_layers, layer_width=layer_width)
    model.to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss = 0.0
    
    data_path = config.dataset_path+"mouse_diffs"
    _, dirs, _ = os.walk(data_path).__next__()
    dirs.remove('.ipynb_checkpoints') if '.ipynb_checkpoints' in dirs else dirs
    
    trainset_dirs = dirs[:int(len(dirs)*0.8)]
    valset_dirs = dirs[int(len(dirs)*0.8):]
    
    train_set = MouseDataset(data_path, trainset_dirs)
    val_set = MouseDataset(data_path, valset_dirs)

    train_dl = DataLoader(train_set, batch_size=128, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=128, shuffle=False)

    best_loss, best_age = 100, 0
    ground_truth, predictions = None, None
    start = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, config.epochs))
        print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(tqdm(train_dl)):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            inputs = torch.squeeze(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("output shape: ",outputs.shape)
            print("targets shape: ",targets.shape)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            
            # metrics = {"model name : {} - train/train_loss".format(name): loss}
            # wandb.log(metrics)

        val_loss = 0
        model.eval()
        ground_truth, predictions = None, None
        for i, (inputs, targets) in enumerate(tqdm(val_dl)):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            inputs = torch.squeeze(inputs)
            outputs = model(inputs)
            if ground_truth is None:
                ground_truth = targets
                predictions = outputs
            else:
                ground_truth = torch.cat((ground_truth, targets), 0)
                predictions = torch.cat((predictions, outputs), 0)

            val_loss += criterion(outputs, targets)
        val_loss = val_loss / len(val_dl)
        #Log    
        # metrics = {"model name : {} - val/val_loss".format(name): val_loss}
        # wandb.log(metrics)
        
        print("validation loss at epoch {}:".format(epoch), val_loss.item())
        if best_loss > val_loss:
            torch.save(model.state_dict(), save_path)
            # utils.plot_pred(ground_truth, predictions, name)
            best_loss = val_loss
            best_age = 0
        else:
            best_age += 1
            print("patience: {}/{}".format(best_age, config.patience))
            if best_age == int(config.patience / 2):
                print("50% patience reached, decrease learning rate")
                lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print("New learning rate: ", param_group['lr'])
            if best_age >= config.patience:
                print("patience reached, stop training")
                break


    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))

    # print(ground_truth[100])
    # print(predictions[100])


nn_params2 = [(2, 256)]

for p in nn_params2:
    train_mouse(n_layers=p[0], layer_width=p[1], epochs=20)
