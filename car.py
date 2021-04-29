import torch, argparse
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
    def __init__(self, path1, path2, useGPU=False):
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
        self.useGPU = useGPU

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
        idx = idx + 320000

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

        if self.useGPU:
            user = torch.tensor([user], device='cuda:0')
            startLocVector = torch.tensor(startLocVector, device='cuda:0')
            stopLocVector = torch.tensor(stopLocVector, device='cuda:0')
            weather = torch.tensor(weather, device='cuda:0')
            location = torch.tensor(location, device='cuda:0')
            return user, startLocVector, stopLocVector, weather, location
        else:
            return torch.tensor([user]), torch.tensor(startLocVector), torch.tensor(stopLocVector), torch.tensor(weather), torch.tensor(location)
        
    def __len__(self):
        return 400000
        #return 400000
        #总数据 423544行

def haversine_dis(lon1, lat1, lon2, lat2): #经纬度计算距离
    #将十进制转为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    #haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = sin(d_lat/2)**2 + cos(lat1)*cos(lat2)*sin(d_lon/2)**2
    c = 2 * asin(sqrt(aa))
    r = 6371 # 地球半径，千米
    return c*r*1000

def correctIndex(pois, stopLocVector): #正确地点下标及各地点距离计算
    longitude, latitude, idx = stopLocVector[0][-2], stopLocVector[0][-1], 0
    idx, MM, distance = 0, 0, [0 for x in range(len(pois))]
    for j in range(len(pois)):
        poi = pois[j]
        lon, lat = poi[0], poi[1]
        #新图
        
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

def run(train=True, maxNodes=20):
    '''
    ./DeepModel/newModel.pt 新模型, GPU训练
    '''

    useGPU = torch.cuda.is_available()

    path1, path2 = './data/train_new.csv', './data/weather.csv'
    modelPath = './DeepModel/newModel.pt'
    
    if platform.system() == 'Windows': #跨系统运行
        path1, path2 = 'C:\\Users\\Lenovo\\Desktop\\car\\car_trace\\data\\train_new.csv', 'C:\\Users\\Lenovo\\Desktop\\car\\car_trace\\data\\weather.csv'
        modelPath = 'C:\\Users\\Lenovo\\Desktop\\car\\car_trace\\DeepModel\\newModel.pt'
    
    dataSet = MyDataSet(path1, path2, useGPU=useGPU)
    dataLoader = DataLoader(dataset=dataSet)
    deepModel = DeepJMTModel(8, 10, useGPU=useGPU)
    deepModel.load_state_dict(torch.load(modelPath))
    if useGPU:
        deepModel = deepModel.cuda()
    #模型准备

    lastUser, lastTime = None, None
    if useGPU:
        lossFun = torch.nn.CrossEntropyLoss().cuda()
    else:
        lossFun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=deepModel.parameters(), lr=0.0001)
    
    '''
    if useGPU:
        lossFun = lossFun.cuda()
    '''
    #损失函数，优化器准备


    if train: #训练 or 测试
        deepModel.train()
        #torch.autograd.set_detect_anomaly(True)
    else:
        deepModel.eval()

    total, correct = 0, 0

    for t in range(10):
        #epoch 100
        All, right = 0, 0
        for i, data in enumerate(dataLoader):
            #print(i)
            user, startLocVector, stopLocVector, weather, location = data
            time = startLocVector[0:6]
            
            if (i == 0) or (lastUser is None) or (user != lastUser): #用户切换状态改变
                if useGPU:
                    lastUser = user
                    nextHid = torch.randn(1, 10, device='cuda:0')
                    periodHid = torch.randn(1, 10, device='cuda:0')
                    qhh = torch.zeros(10, 10, device='cuda:0')
                    aH = torch.zeros(10, 10, device='cuda:0')
                else :
                    lastUser = user
                    nextHid = torch.randn(1, 10)
                    periodHid = torch.randn(1, 10)
                    qhh = torch.zeros(10, 10)
                    aH = torch.zeros(10, 10)
                
                nodes = [[
                    float(format(location[0][0], '.6f')),
                    float(format(location[0][1], '.6f')),
                    float(format(weather[0][0], '.6f')),
                    0.0
                ]]
                lastTime = None
            else:
                #input('wait')
                if len(nodes) == maxNodes:
                    nodes.pop(0)
                nodes.append([
                    float(format(location[0][0], '.6f')),
                    float(format(location[0][1], '.6f')),
                    float(format(weather[0][0], '.6f')),
                    0.0
                ])

            #print(len(nodes))
            #a = input('wait')

            pois = POI(format(location[0][0], '.6f'), format(location[0][1], '.6f'))

            for j in range(len(pois)): #poi距离计算
                Loc = pois[j]['location']
                lon, lat = Loc.split(',')
                lon, lat = float(lon), float(lat)
                pois[j]['distance'] = haversine_dis(lon, lat, startLocVector[0][0], startLocVector[0][1])
            #poi 修改距离

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

            #def forward(self, x, nextHid, user, location, periodHid, qhh, aH, pre, pois, nodes, edges, useGPU):
            #print(Node)
            #print(location)

            if useGPU:
                newNodes = torch.tensor(Node, device='cuda:0')
                newEdges = torch.tensor([len(Node), len(Node)], device='cuda:0')
            else:
                newNodes = torch.tensor(Node)
                #print(type(newNodes))
                newEdges = torch.ones([len(Node), len(Node)])
            #print('newNodes {}'.format(newNodes.shape))

            #print('nodes {}'.format(newNodes.shape))

            nextHid, periodHid, qhh, aH, index, raw = deepModel( #调用模型
                x=startLocVector,
                nextHid=nextHid,
                lastTime=lastTime,
                user=user,
                location=[
                    float(format(location[0][0], '.6f')),
                    float(format(location[0][1], '.6f'))
                ],
                periodHid=periodHid,
                qhh=qhh,
                aH=aH,
                #pre=len(nodes),
                pre=0,
                pois=pois,
                nodes=newNodes,
                edges=newEdges,
                useGPU=useGPU
            )
            lastTime = time
            
            #print("poi {}".format(len(pois)))
            #print("raw {}".format(raw.shape))
            correctIdx, MM, distance = correctIndex(pois=Node, stopLocVector=stopLocVector)
            #print("correctIdx {}".format(correctIdx))
            add = False
            if useGPU:
                target = torch.zeros(len(Node), dtype=torch.long, device='cuda:0')
            else:
                target = torch.zeros(len(Node), dtype=torch.long)

            if useGPU:
                target = target.cuda()

            target[correctIdx] = 1
            for idx in range(len(nodes)):
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

            for idx in index: #正确地点计算
                left = distance[correctIdx] - distance[idx]
                if left < 0:
                    left = left * -1
                if left < 500:
                    add = True
            
            if add: #正确数据加1
                right = right + 1

            #print("test i is{}".format(i))
            if i % 100 == 0: #正确率计算和保存模型
                #print(i)
                if train:
                    print("All {}  right {} loss is {}  当前epoch训练{}个样本 当前正确率{}".format(All, right, loss, 100, right / All))
                else:
                    print("All {}  right {}  当前epoch测试{}个样本 当前正确率{}".format(All, right, 100, right / All))
                #torch.save(deepModel, modelPath)
                if i % 1000 == 0:
                    torch.save(deepModel.state_dict(), modelPath)
                total, correct = total + All, correct + right
                All, right = 0, 0
               #break
        '''
        total, correct = total + All, correct + right
        print("total {} correct {} 当前epoch {} 总正确率{}".format(total, correct, t, correct / total))
        '''
        if not train:
            break
        #torch.save(deepModel, modelPath)
        torch.save(deepModel.state_dict(), modelPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train or test')
    parser.add_argument('--train', type=str, help='传入的训练参数类型', default='True', nargs='?')
    parser.add_argument('--maxNodes', type=int, help='传入的最大历史数据', default=20, nargs='?')
    train, maxNodes = parser.parse_args().train, parser.parse_args().maxNodes
    #default train is True
    if train == 'False':
        train = False
    
    run(train=train, maxNodes=maxNodes)
    

