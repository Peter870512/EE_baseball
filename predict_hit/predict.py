import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import math

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


def predict_distance(exit_velo, launch_angle):
    # km/h -> m/s
    exit_velo = exit_velo * 0.277778
    # 360 -> 2*pi
    launch_angle = launch_angle * math.pi / 180 
    g = 9.81
    x0 = 0
    y0 = 0.9
    # parameter of ball
    pa = 1.2            # kg / m^3
    C = 0.3             # coefficient
    r = 0.025           # m
    p = 2000            # kg / m^3
    A = math.pi * r * r
    D = pa * C * A / 2
    m = (4/3) * math.pi * math.pow(r, 3) * p   # kg 
    dt = 0.01

    v_x = exit_velo * math.cos(launch_angle)
    v_y = exit_velo * math.sin(launch_angle)
    if launch_angle > 0:
        a_x = -D/m * abs(v_x) * v_x
        a_y = -D/m * abs(v_y) * v_y - g
        x = x0
        y = y0
        while y >= 0:
            x = x + v_x * dt + a_x * dt * dt/2
            y = y + v_y * dt + a_y * dt * dt/2
            v_x = v_x + a_x * dt
            v_y = v_y + a_y * dt
            a_x = -D/m * abs(v_x) * v_x
            a_y = -D/m * abs(v_y) * v_y - g
    else:
        a_x = 0
        a_y = -g
        x = x0
        y = y0
        while y >= 0:
            x = x + v_x * dt
            y = y + v_y * dt + a_y * dt * dt/2
            v_y = v_y + a_y * dt
    hit_distance = x
    return hit_distance


def prediction_of_hit(exit_velo, launch_angle, spray_angle):
    start_time = time.time()
    hit_distance = predict_distance(exit_velo, launch_angle)
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
    exit_velo = exit_velo / 1.6093  # km / h -> mph
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
    return predict_result, hit_distance

if __name__ == '__main__':
    result, distance = prediction_of_hit(80, 1, 30)