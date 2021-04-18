import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from itertools import istools
from geopy.distance import geodesic
import pandas as pd
from Model.model import DeepJMTModel, POI

def weatherIdx(month, day):
        if month == 8:
            return day - 1
        return (day - 1) + 31

class MyDataSet(Dataset):
    def __init__(self, path1, path2):
        '''
        with open(path1, encoding='utf-8') as f:
            #self.f = f
            self.data = np.loadtxt(
                f, 
                encoding='utf-8', 
                dtype=str, 
                skiprows=1, 
            )
        print(self.data[0:5])

        '''

        data = pd.read_csv(
            path1,
            names=[
                'ID',
                'startTime',
                'startLon',
                'startLat',
                'startPOS',
                'stopTime',
                'stopLon',
                'stopLat',
                'stopPOS'
            ],
            dtype={
                'ID': str,
                'startTime': str,
                'startLon': str,
                'startLat': str,
                'startPOS': str,
                'stopTime': str,
                'stopLon': str,
                'stopLat': str,
                'stopPOS': str
            },
            skiprows=1
        )

        self.data = [
            [row['ID'], row['startTime'], row['startLon'], row['startLat'], row['stopTime'], row['stopLon'], row['stopLat']] for index, row in data.iterrows()
        ]

        print(self.data[0:5])

        a = input("wait")

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
        if idx > self.end:
            self.load()

        user = self.data[idx][0]
        time1, time2 = self.data[idx][1].split(' ')
        startYear, startMonth, startDay = time1.split('/')
        startYear, startMonth, startDay = float(startYear), float(startMonth), float(startDay)
        startHour, startMiute, startSecond = time2.split(':')
        startHour, startMiute, startSecond = float(startHour), float(startMiute), float(startSecond)
        #上车year month day hour minute second

        time1, time2 = self.data[idx][4].split(' ')
        stopYear, stopMonth, stopDay = time1.split('/')
        stopYear, stopMonth, stoptDay = float(stopYear), float(stopMonth), float(stopDay)
        stopHour, stopMiute, stopSecond = time2.split(':')
        stopHour, stopMiute, stopSecond = float(stopHour), float(stopMiute), float(stopSecond)
        #下车year month day hour minute second

        startLocVector = torch.tensor([startYear, startMonth, startDay, startHour, startMiute, startSecond, self.data[idx][2], self.data[idx][3]])
        stopLocVector = torch.tensor([stopYear, stopMonth, stopDay, stopHour, stopMiute, stopSecond, self.data[idx][5], self.data[idx][6]])
        
        weaIdx = weatherIdx(startMonth, startDay)
        weather = torch.tensor([self.weather[weaIdx][1]])

        location = [self.data[idx][2], self.data[idx][3]]

        return user, startLocVector, stopLocVector, weather, location
        
    def __len__(self):
        return 400000
        #总数据 423544行
        
if __name__ == '__main__':
    path1, path2 = 'C:\\Users\\Lenovo\\Desktop\\Code\\car\\data\\train_new.csv', 'C:\\Users\\Lenovo\\Desktop\\Code\\car\\data\\weather.csv'
    dataSet = MyDataSet(path1, path2)
    dataLoader = DataLoader(dataset=dataSet)
    deepModel = DeepJMTModel(8, 10)

    for i, data in enumerate(dataLoader):
        user, startLocVector, stopLocVector, weather, location = data
        if (lastUser is None) or (user != lastUser):
            lastUser, nextHid, periodHid, qhh, aH = user, None, None, torch.zeros(10, 10), torch.zeros(10, 10) 
            nodes = [location[0], location[1], weather[0], 0]
        else:
            nodes.append([location[0], location[1], weather[0], 0])
        pois = POI(location[0], location[1])
        
        projectionMatrix = [[p['location']] for p in pois]
        Len = len(pois)
        for i in range(Len):
            temp = projectionMatrix[i][0]
            a1, a2 = temp.split(',')
            longitude, latitude = float(a1), float(a2)
            projectionMatrix[i] = [longitude, latitude, weather[0], 0]
        
        #print(projectionMatrix)
        Node = nodes.append(projectionMatrix)
        Len = len(Node)

        for i in range(Len):
            Node[i][3] = geodesic((location[0], location[1]), (Node[i][0], Node[i][1])).m
        
        #def forward(self, x, nextHid, user, location, periodHid, qhh, aH, pre, pois, nodes, edges):
        nextHid, periodHid, qhh, aH, index = deepModel(
            startLocVector,
            nextHid,
            location,
            periodHid,
            qhh,
            aH,
            len(nodes),
            pois,
            torch.tensor(Node),
            torch.ones(Len, Len)
        )

        if i == 1:
            a = input("test: ")
