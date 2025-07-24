import os
import json

import torch
from torch import nn
from scipy.fft import rfft
import numpy as np
import joblib


class Utils:
    DEVICE = "cpu"  # using cuda makes inference slower

    @staticmethod
    def get_examples():
        res = {}
        # sample requests are stored in files due to big size
        for i, file_name in enumerate(os.listdir("./examples")):
            with open("./examples/" + file_name, "r") as file:
                res[i] = json.load(file)

        return res

    @staticmethod
    def prepare_data(run):
        res = []
        # gettings spectrums for each sensor
        run = np.array(run.sensor1_sensor2)
        freq1 = rfft(run[:, 0], n=12500)[3:118]
        freq1 = np.abs(freq1)
        freq2 = rfft(run[:, 1], n=12500)[3:118]
        freq2 = np.abs(freq2)

        res = np.concatenate((freq1, freq2))

        ss = joblib.load("./artifacts/scaler.pkl")
        res = ss.transform([res])  # scaling resulting spectrums

        res = torch.from_numpy(res.astype(np.float32)).to(Utils.DEVICE)
        res = torch.stack([res[:, :115], res[:, 115:]], dim=1)
        res = res.view(res.shape[0], 1, 2, 115)  # setting proper input shape

        return res

    @staticmethod
    def get_model():
        model = CNN().to(Utils.DEVICE)
        model.load_state_dict(
            torch.load("./artifacts/model.pt", map_location=Utils.DEVICE)
        )

        return model


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 33, kernel_size=(2, 15), padding=(0, 7)),
            nn.ReLU(),
            nn.Conv2d(33, 1, kernel_size=(1, 1)),
            nn.Linear(115, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x):
        return self.layers(x)
