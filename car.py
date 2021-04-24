import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from itertools import istools
#from geopy.distance import geodesic
from math import radians,sin,cos,asin,sqrt
import pandas as pd
from Model.model import DeepJMTModel, POI
from geopy.distance import geodesic
import platform

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

        '''
        print("data")
        print(self.data[0:5])
        '''

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
        '''
        print("idx {}".format(idx))
        print(len(self.data))
        '''
        user = self.data[idx][0]
        time1, time2 = self.data[idx][1].split(' ')
        #print("time1 {} time2 {} unpack{}".format(time1, time2, time1.split('-')))
        startYear, startMonth, startDay = time1.split('-')
        startYear, startMonth, startDay = float(startYear), float(startMonth), float(startDay)
        startHour, startMiute, startSecond = time2.split(':')
        startHour, startMiute, startSecond = float(startHour), float(startMiute), float(startSecond)
        #上车year month day hour minute second

        time1, time2 = self.data[idx][4].split(' ')
        stopYear, stopMonth, stopDay = time1.split('-')
        stopYear, stopMonth, stoptDay = float(stopYear), float(stopMonth), float(stopDay)

        stopHour, stopMiute, stopSecond = time2.split(':')
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

def haversine_dis(lon1, lat1, lon2, lat2):
    #将十进制转为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    #haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = sin(d_lat/2)**2 + cos(lat1)*cos(lat2)*sin(d_lon/2)**2
    c = 2 * asin(sqrt(aa))
    r = 6371 # 地球半径，千米
    return c*r*1000

def correctIndex(pois, stopLocVector):
    longitude, latitude, idx = stopLocVector[0][-2], stopLocVector[0][-1], 0
    idx, MM, distance = 0, 0, [0 for x in range(len(pois))]
    for j in range(len(pois)):
        poi = pois[j]
        location = poi['location']
        lon, lat = location.split(',')
        lon, lat = float(lon), float(lat)
        
        value = haversine_dis(
            min(lon, longitude),
            min(lat, latitude),
            max(lon, longitude),
            max(lat, latitude)
        )
        #value = geodesic((lat, lon), (latitude, longitude)).m
        if value < 0:
            value = value * -1
        distance[j] = value
        #print(value)

        if j == 0:
            MM = value
        else:
            if value < MM:
                idx, MM = j, value
    
    return idx, MM, distance

def run(train=False):
    path1, path2 = './data/train_new.cav', './data/weather.csv'
    modelPath = './DeepModel/save.pt'
    
    if platform.system().lower = 'windows':
        path1, path2 = 'C:\\Users\\Lenovo\\Desktop\\car\\car_trace\\data\\train_new.csv', 'C:\\Users\\Lenovo\\Desktop\\car\\car_trace\\data\\weather.csv'
        modelPath = 'C:\\Users\\Lenovo\\Desktop\\car\\car_trace\\DeepModel\\save.pt'
    
    dataSet = MyDataSet(path1, path2)
    dataLoader = DataLoader(dataset=dataSet)
    #deepModel = DeepJMTModel(8, 10)
    deepModel = torch.load(modelPath)
    lastUser, lastTime = None, None
    lossFun, optimizer = torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=deepModel.parameters(), lr=0.0001)
    #modelPath = 'C:\\Users\\Lenovo\\Desktop\\Code\\car\\DeepModel\\save.pt'
    #deepModel = torch.load(modelPath)

    if train:
        deepModel.train()
        torch.autograd.set_detect_anomaly(True)
    else:
        deepModel.eval()


    total, correct = 0, 0

    for t in range(100):
        #epoch 100
        All, right = 0, 0
        for i, data in enumerate(dataLoader):
            #print(i)
            user, startLocVector, stopLocVector, weather, location = data
            time = startLocVector[0:6]

            
            if (i == 0) or (lastUser is None) or (user != lastUser):
                lastUser, nextHid, periodHid, qhh, aH = user, None, None, torch.zeros(10, 10), torch.zeros(10, 10) 
                nodes = [[
                    float(format(location[0][0], '.6f')),
                    float(format(location[0][1], '.6f')),
                    float(format(weather[0][0], '.6f')),
                    0.0
                ]]
                lastTime = None
            else:
                nodes.append([
                    float(format(location[0][0], '.6f')),
                    float(format(location[0][1], '.6f')),
                    float(format(weather[0][0], '.6f')),
                    0.0
                ])

            pois = POI(format(location[0][0], '.6f'), format(location[0][1], '.6f'))


            Node = nodes[:]
            projectionMatrix = [[p['location'], p['distance']] for p in pois]
            Len = len(pois)

            for j in range(Len):
                temp = projectionMatrix[j][0]
                a1, a2 = temp.split(',')
                longitude, latitude, distance = float(a1), float(a2), float(projectionMatrix[j][1])
                projectionMatrix[j] = [longitude, latitude, float(format(weather[0][0], '.6f')), distance]
            
            for temp in projectionMatrix:
                Node.append(temp)

            user = float(format(user[0][0], '.6f'))

            #def forward(self, x, nextHid, user, location, periodHid, qhh, aH, pre, pois, nodes, edges):
            #print(Node)
            nextHid, periodHid, qhh, aH, index, raw = deepModel(
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

            
            
            #print("poi {}".format(len(pois)))
            #print("raw {}".format(raw.shape))
            correctIdx, MM, distance = correctIndex(pois=pois, stopLocVector=stopLocVector)
            print("correctIdx {}".format(correctIdx))
            target, add = torch.zeros(len(pois), dtype=torch.long), False

            target[correctIdx] = 1
            for idx in range(len(pois)):
                left = distance[correctIdx] - distance[idx]
                if left < 0:
                    left = left * -1
                if left < 10:
                    target[idx] = 1
            if train:
                loss = lossFun(input=raw, target=target)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            All = All + 1

            for idx in index:
                left = distance[correctIdx] - distance[idx]
                if left < 0:
                    left = left * -1
                if left < 10:
                    add = True
            
            if add:
                right = right + 1


            if i % 10 == 0:
                if train:
                    print("All {}  right {} loss is {}  当前epoch训练{}个样本 当前正确率{}".format(All, right, loss, i, right / All))
                else:
                    print("All {}  right {}  当前epoch训练{}个样本 当前正确率{}".format(All, right, i, right / All))
                torch.save(deepModel, modelPath)
                total, correct = total + All, correct + right
                All, right = 0, 0
               # break
        
        total, correct = total + All, correct + right
        print("total {} correct {} 当前epoch {} 总正确率{}".format(total, correct, t, correct / total))
        
        torch.save(deepModel, modelPath)
        #break
        #a = input("wait")


if __name__ == '__main__':
    run(train=False)
    

