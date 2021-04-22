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

        print("data")
        print(self.data[0:5])

        for i in range(len(self.data)):
            self.data[i][0] = float(self.data[i][0])
            self.data[i][2] = float(self.data[i][2])
            self.data[i][3] = float(self.data[i][3])
            self.data[i][5] = float(self.data[i][5])
            self.data[i][6] = float(self.data[i][6])

        '''
        with open(path2, encoding='utf-8') as f:
            self.weather = np.loadtxt(path2, dtype=str, delimiter=',', skiprows=1)
        '''
        weather = pd.read_csv(
            path2,
            names=[
                'date',
                'weather'
            ],
            dtype={
                'date': str,
                'weather': float
            },
            skiprows=1
        )
        #print('weather')
        #print(weather)
        self.Wea = [
            row['weather'] for index, row in weather.iterrows()
        ]
        
        '''
        for i in range(len(weather)):
            if weather[i][1] == 'bad':
                weather[i][1] = 0.0
            else:
                weather[i][1] = 1.0
        self.data = [ID, StartTime, Startlongitude, Startlatitude, StopTime, Stoplongitude, Stoplatitude]
        self.weather = [Date, Weather]
        '''

    def __getitem__(self, idx):
        user = self.data[idx][0]
        time1, time2 = self.data[idx][1].split(' ')
        startYear, startMonth, startDay = time1.split('/')
        startYear, startMonth, startDay = float(startYear), float(startMonth), float(startDay)
        startHour, startMiute = time2.split(':')
        startSecond = 0.0
        #临时加秒
        startHour, startMiute, startSecond = float(startHour), float(startMiute), float(startSecond)
        #上车year month day hour minute second

        time1, time2 = self.data[idx][4].split(' ')
        stopYear, stopMonth, stopDay = time1.split('/')
        stopYear, stopMonth, stoptDay = float(stopYear), float(stopMonth), float(stopDay)

        stopHour, stopMiute = time2.split(':')
        stopSecond = 0.0
        #临时加秒
        stopHour, stopMiute, stopSecond = float(stopHour), float(stopMiute), float(stopSecond)
        #下车year month day hour minute second

        startLocVector = [startYear, startMonth, startDay, startHour, startMiute, startSecond, self.data[idx][2], self.data[idx][3]]
        stopLocVector = [stopYear, stopMonth, float(stopDay), stopHour, stopMiute, stopSecond, self.data[idx][5], self.data[idx][6]]
        
        weaIdx = weatherIdx(startMonth, startDay)
        weather = [self.Wea[int(weaIdx)]]

        location = [self.data[idx][2], self.data[idx][3]]

        return torch.tensor([user]), torch.tensor(startLocVector), torch.tensor(stopLocVector), torch.tensor(weather), torch.tensor(location)
        
    def __len__(self):
        return 400000
        #总数据 423544行
        
if __name__ == '__main__':
    path1, path2 = 'C:\\Users\\Lenovo\\Desktop\\Code\\car\\data\\test.csv', 'C:\\Users\\Lenovo\\Desktop\\Code\\car\\data\\weather.csv'
    dataSet = MyDataSet(path1, path2)
    dataLoader = DataLoader(dataset=dataSet)
    deepModel = DeepJMTModel(8, 10)
    lastUser, lastTime = None, None
    for i, data in enumerate(dataLoader):
        user, startLocVector, stopLocVector, weather, location = data
        time = startLocVector[0:6]
        '''
        print("user {} and startLocVector {}".format(user.size(), startLocVector.size()))
        print("location {}".format(location.size()))
        '''
        
        if (lastUser is None) or (user != lastUser):
            lastUser, nextHid, periodHid, qhh, aH = user, None, None, torch.zeros(10, 10), torch.zeros(10, 10) 
            nodes = [[
                float(format(location[0][0], '.6f')),
                float(format(location[0][1], '.6f')),
                float(format(weather[0][0], '.6f')),
                0.0
            ]]
            lastTime = None
        else:
            nodes.append([[
                float(format(location[0][0], '.6f')),
                float(format(location[0][1], '.6f')),
                float(format(weather[0][0], '.6f')),
                0.0
            ]])

        pois = POI(format(location[0][0], '.6f'), format(location[0][1], '.6f'))

        '''
        print("location {} {}".format(format(location[0][0], '.6f'), format(location[0][1], '.6f')))
        print(pois)
        '''

        Node = nodes[:]
        projectionMatrix = [[p['location'], p['distance']] for p in pois]
        Len = len(pois)

        for i in range(Len):
            temp = projectionMatrix[i][0]
            a1, a2 = temp.split(',')
            longitude, latitude, distance = float(a1), float(a2), float(projectionMatrix[i][1])
            projectionMatrix[i] = [longitude, latitude, float(format(weather[0][0], '.6f')), distance]
        
        for temp in projectionMatrix:
            Node.append(temp)

        '''

        for i in range(len(nodes)):
            try:
                Node[i][3] = geodesic(
                    (
                        float(format(location[0][1], '.6f')),
                        float(format(location[0][0], '.6f'))
                    ),
                    (Node[i][1], Node[i][0])
                ).m
            except ValueError:
                print(Node[i][0])
                print(Node[i][1])
                break
        '''
        user = float(format(user[0][0], '.6f'))

        #def forward(self, x, nextHid, user, location, periodHid, qhh, aH, pre, pois, nodes, edges):
        nextHid, periodHid, qhh, aH, index = deepModel(
            x=startLocVector,
            nextHid=nextHid,
            lastTime=lastTime,
            user=user,
            location=[
                float(format(location[0][1], '.6f')),
                float(format(location[0][0], '.6f'))
            ],
            periodHid=periodHid,
            qhh=qhh,
            aH=aH,
            pre=len(nodes),
            pois=pois,
            nodes=torch.tensor(Node),
            edges=torch.ones([len(Node), len(Node)])
        )

        lastTime = time

        if i == 1:
            a = input("test: ")
