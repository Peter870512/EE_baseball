import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

class Predicter(nn.Module):
    def __init__(self):
        super(Predicter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        out = self.fc(x)
        return out

class testDataset(Dataset):
    def __init__(self, x):
        self.x = torch.Tensor(x)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        return X

def prediction_of_hit(exit_velo, launch_angle, spray_angle):
    start_time = time.time()
    train_mean = [88.76291862, 11.53803822, -5.66975084]
    train_std = [13.87276049, 25.30113639, 27.11713013]
    model_path = 'predictor.pth'
    if torch.cuda.is_available():
        model = Predicter.cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model = Predicter()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    data = np.zeros((1,3))
    exit_velo = exit_velo / 1.6093
    data[0, 0] = (exit_velo - train_mean[0]) / train_std[0]
    data[0, 1] = (launch_angle - train_mean[1]) / train_std[1]
    data[0, 2] = (spray_angle - train_mean[2]) / train_std[2]
    data = testDataset(data)
    batch_size = 1
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for i, data in enumerate(test_loader):
        if torch.cuda.is_available():
            output = model(data.cuda())
        else:
            output = model(data)
    output = np.argmax(output.cpu().data.numpy(), axis=1)

    predict_result = None
    if output == 0:
        predict_result = 'single'
    elif output == 1:
        predict_result = 'double'
    elif output == 2:
        predict_result = 'triple'
    elif output == 3:
        predict_result = 'homerun'
    elif output == 4:
        predict_result = 'fly out'
    elif output == 5:
        predict_result = 'ground out'
    elif output == 6:
        predict_result = 'line out'
    end_time = time.time()
    #print(end_time - start_time)  
    print(predict_result)
    return predict_result

if __name__ == '__main__':
    prediction_of_hit(80, 1, 30)