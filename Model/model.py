import torch, torchvision, requests 
import json, datetime, math, DynamicNet.GAT

def POI(longitude, latitude):
    URL = 'https://restapi.amap.com/v3/geocode/regeo?parameters'
    location = str(longitude) + ',' + str(latitude)
    parameters = {
        'location': location,
        'key': '8c9b55944852c98366d685d8894d74a0',
        'radius': '1000',
        'extensions': 'all',
        'batch': 'false',
        'roadlevel': '0',
        'output': 'JSON'
    }
    #print(location)
    result = requests.get(url = URL, params = parameters)
    #print(result.request)
    state = json.loads(result.text)
    return state['regeocode']['pois']

def timeCompare(time1, time2):
    lastTime = datetime(time1[0], time1[1], time1[2], time1[3], time1[4], time1[5])
    now = datetime(time2[0], time2[1], time2[2], time2[3], time2[4], time2[5])
    if (now - lastTime).hour < 8:
        return True
    return False

class DeepJMTModel(torch.nn.Module):
    def __init__(self, input_num, hidden_num):
        super(DeepJMTModel, self).__init__()
        '''
        input_num = 8
        year/month/day/hour/minute/second/longitude/latitude 
        hidden_num = 10
        '''
        self.hidden_size = hidden_num
        self.GRUCell1 = torch.nn.GRUCell(input_num, hidden_num)
        self.GRUCell2 = torch.nn.GRUCell(input_num + 1, hidden_num)
        self.GRUCell3 = torch.nn.GRUCell(input_num + 1, hidden_num)
        self.weight = torch.nn.Parameter(data = torch.randn(1, 1), requires_grad = True)
        self.Relu = torch.nn.ReLU()
        self.GAT = DynamicNet.GAT.GAT(4, 4, 2, 0.2, 0.2, 4)
        
        '''
        longitude, latitude, weather, distance
        '''
        self.scaling = 2
    
    """
    def changeGAT(self, nodes, edges):
        self.add_module('GAT', DeepJMT.GAT(nodes, edges))
    """
    def forward(self, x, nextHid, lastTime, user, location, periodHid, qhh, aH, pre, pois, nodes, edges):
        #传入参数都是tensor, location除外(list), user(float)
        #print("x {}".format(x.size()))
        if (nextHid is None) or ( (lastTime is None) or (compare(x[0][0:6], lastTime)) ):
            if nextHid is None:
                nextHid = torch.randn(1, self.hidden_size)
            nextHid = self.GRUCell1(x, nextHid)
        else:
            nextHid = self.GRUCell2(
                    torch.cat(
                        (
                            torch.tensor([[user]]),
                            x
                        ),
                        dim=1
                    ),
                    nextHid
                )

        
        '''
        for i in range(len(x)):
            timeLocVector = x[i]
            if i == 0 or compare(x[i - 1][0:6], x[i][0:6]):
                nextHid = self.GRUCell1(timeLocVector, nextHid)
                #hidLow = nextHid
            else:
                nextHid = self.GRUCell2(
                    torch.cat(
                        (
                            torch.tensor([user]),
                            timeLocVector
                        ),
                        dim=1
                    ),
                    nextHid
                )
                #hidHigh = nextHid
        #双层RNN编码历史轨迹
        '''
        

        #pois = POI(location[0], location[1])
        nextHidT = nextHid.t()
        #nextHidT(self.hidden_size * 1)
        dist = [1 for x in range(len(pois))]
        qLis = torch.randn(self.hidden_size, 2)
        allwe = torch.zeros(self.hidden_size, 2)
        for i in range(len(pois)):
            #poi = pois[i]
            longitude, latitude = pois[i]['location'].split(',')
            longitude, latitude = float(longitude), float(latitude)
            eLi = torch.tensor([longitude, latitude])
            qLis = nextHidT * self.weight * eLi
            distance = math.exp(-1 * (float(pois[i]['distance']) / self.scaling))
            allwe = allwe + torch.exp(qLis * distance)

        cL = torch.zeros(self.hidden_size, 1)
        for i in range(len(pois)):
            #poi = pois[i]
            longitude, latitude = pois[i]['location'].split(',')
            longitude, latitude = float(longitude), float(latitude)
            eLi = torch.tensor([longitude, latitude])
            qLis = nextHidT * self.weight * eLi
            #print(qLis)
            #distance = math.exp(-1 * (float(pois[i]['distance'] / self.scaling))
            distance = math.exp(
                -1 * (
                    float( pois[i]['distance'] ) / self.scaling
                )
            )
            #print("hello")
            a = qLis * distance
            t1 = torch.exp(a)
    
            t = torch.div(t1, allwe)
            #temp(self.hiddenze * 2)
            cL = cL + t * eLi.t()
        #空间上下文提取器  

        Length = len(x)
        if periodHid is None:
            periodHid = torch.randn(1, self.hidden_size)
        timeLocVector = x[-1]

        '''
        print(timeLocVector.shape)
        print(torch.unsqueeze(timeLocVector, 0).shape)
        '''

        periodHid = self.GRUCell3(
            torch.cat(
                (
                    torch.tensor([[user]]),
                    torch.unsqueeze(timeLocVector, 0)
                ),
                dim=1
            ),
            periodHid
        )
        qhi = torch.exp(nextHidT * periodHid)
        qhh = qhh + qhi
        aH = aH + torch.div(qhi, qhh)
        cP = aH * periodHid
        #cP(self.hiddenze, 1)
        #周期上下文提取器

        outGATState = self.GAT(nodes, edges)
        #GAT hidden_stata

        projectionMatrix = [[p['location']] for p in pois]
        for i in range(len(projectionMatrix)):
            temp = projectionMatrix[i][0]
            a1, a2 = temp.split(',')
            longitude, latitude = float(a1), float(a2)
            projectionMatrix[i] = [longitude, latitude]
        #print(projectionMatrix)
        proMatrixTensor = torch.tensor(projectionMatrix)

        '''
        print("nextHid {}".format(nextHid.shape))
        print("cL.t() {}".format(cL.t().shape))
        print("cP.t() {}".format(cP.t().shape))
        '''

        mL = torch.cat(
            (nextHid, cL.t(), cP.t()),
            dim=0
        )

        #mL(1, 3 * self.hidden_size)
        deepLoc = torch.randn(len(pois), 2, 3 * self.hidden_size)
        for i in range(len(proMatrixTensor)):
            predictionLoc = proMatrixTensor[i]
            print("pred {}".format(predictionLoc.t().shape))
            print("mL {}".format(mL.shape))
            deepLoc[i] = predictionLoc.t() * mL
        #DeepJMT 地点预测

        anw = torch.randn(1, len(pois))
        for i in range(len(pois)):
            temp = outGATState[i + pre] * deepLoc[i]
            anw[i] = torch.max(temp)
        anw = torch.nn.Softmax(anw)
        value, index = torch.max(anw, dim=1)
        #结合DeepJMT和GAT

        return nextHid, periodHid, qhh, aH, index


        

        
