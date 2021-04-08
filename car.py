import Model, torch
import numpy as np
from torch.utils.data import Dataset

class Mydata(Dataset):
    def __init__(self, path1, path2):
        with open(path1, encoding='utf-8') as f:
            self.data = np.loadtxt(path1, dtype=str, delimiter=',', skiprows=1, usecols=(0,1,2,3,5,6,7))
        #self.data = data 
        for i in range(423544):
            self.data[i][0] = float(self.data[i][0])
            self.data[i][2] = float(self.data[i][2])
            self.data[i][3] = float(self.data[i][3])
            self.data[i][5] = float(self.data[i][5])
            self.data[i][6] = float(self.data[i][6])

        with open(path2, encoding='utf-8') as f:
            self.weather = np.loadtxt(path2, dtype=str, delimiter=',', skiprows=1)
        
        #bad_weather(0), good_weather(1)
        for i in range(61):
            if ('雷阵雨' in self.weather[i][1]) or ('大雨' in self.weather[i][1]):
                self.weather[i][1] = 0
            else:
                self.weather[i][1] = 1
        '''
        self.data = [ID, StartTime, Startlongitude, Startlatitude, StopTime, Stoplongitude, Stoplatitude]
        self.weather = [Date, Weather]
        '''
    def __getitem__(self, idx):
        
    def __len__(self):
        return 423544
        
if __name__ == '__main__':
