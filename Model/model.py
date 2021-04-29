import torch, torchvision, requests 
import json, datetime, math, DynamicNet.GAT

def POI(longitude, latitude):
    URL = 'https://restapi.amap.com/v3/geocode/regeo?parameters'
    location = str(longitude) + ',' + str(latitude)
    parameters = {
        'location': location,
        'key': '8c9b55944852c98366d685d8894d74a0',
        'radius': '3000',
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
    #print(time1)
    #print(time2)
    lastTime = datetime.datetime(time1[0], time1[1], time1[2], time1[3], time1[4], time1[5])
    now = datetime.datetime(time2[0], time2[1], time2[2], time2[3], time2[4], time2[5])
    if ((now - lastTime).seconds / 3600) < 8:
        return True
    return False

class DeepJMTModel(torch.nn.Module):
    def __init__(self, input_num, hidden_num, useGPU=False):
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

        data = torch.randn(1, 1)
        if useGPU:
            data = data.cuda()
        
        self.weight = torch.nn.Parameter(data=data)
        self.Relu = torch.nn.ReLU()
        self.GAT = DynamicNet.GAT.GAT(4, 4, 2, 0.2, 0.2, 4, useGPU=useGPU)
        if useGPU:
            self.GRUCell1, self.GRUCell2 = self.GRUCell1.cuda(), self.GRUCell2.cuda()
            self.GRUCell3, self.GAT = self.GRUCell3.cuda(), self.GAT.cuda()
            #self.weight = self.weight.cuda()
        #self.softmax = torch.nn.Softmax()
        
        '''
        longitude, latitude, weather, distance
        '''
        self.scaling = 2
    
    """
    def changeGAT(self, nodes, edges):
        self.add_module('GAT', DeepJMT.GAT(nodes, edges))
    """
    def forward(self, x, nextHid, lastTime, user, location, periodHid, qhh, aH, pre, pois, nodes, edges, useGPU=False):
        #传入参数都是tensor, location除外(list), user(float)
        #print("x {}".format(x.size()))
        if (nextHid is None) or ( (lastTime is None) or (timeCompare(lastTime[0], x[0][0:6])) ):
            '''
            if nextHid is None:
                if useGPU:
                    nextHid = torch.randn(1, self.hidden_size, device=torch.device('cuda:0'))
                nextHid = torch.randn(1, self.hidden_size)
                if useGPU:
                    nextHid = nextHid.cuda()
            '''
            nextHid = self.GRUCell1(x, nextHid)
        else:
            if useGPU:
                userT = torch.tensor([[user]], device='cuda:0')
            else:
                userT = torch.tensor([[user]])
            '''
            temp = torch.tensor([[user]])
            if useGPU:
                temp = temp.cuda()
            '''
            nextHid = self.GRUCell2(
                    torch.cat(
                        (
                            userT,
                            x
                        ),
                        dim=1
                    ),
                    nextHid
                )
        #双层RNN编码历史
        
        newNextHid = nextHid.detach()
        #pois = POI(location[0], location[1])
        nextHidT = newNextHid.t()

        #nextHidT(self.hidden_size * 1)
        dist = [1 for x in range(len(pois))]
        #qLis = torch.randn(self.hidden_size, 2)
        if useGPU:
            allwe = torch.zeros(self.hidden_size, 2, device='cuda:0')
        else:
            allwe = torch.zeros(self.hidden_size, 2)
        '''
        if useGPU:
            allwe = allwe.cuda()
        '''
        
        for i in range(len(pois)):
            #poi = pois[i]
            longitude, latitude = pois[i]['location'].split(',')
            longitude, latitude = float(longitude), float(latitude)
            if useGPU:
                eLi = torch.tensor([[longitude, latitude]], device='cuda:0')
            else:
                eLi = torch.tensor([[longitude, latitude]])
            '''
            if useGPU:
                eLi = eLi.cuda()
            '''
            #print("eLi {}".format(eLi.shape))
            qLis = nextHidT @ self.weight @ eLi

            '''
            nextHidT = (self.hidden_size, 1)
            self.weight = (1, 1)
            eLi = (1, 2)
            qLis = (self.hidden_size, 2)
            self.hidden_size = 10
            '''

            distance = math.exp(-1 * (float(pois[i]['distance']) / self.scaling))
            if useGPU:
                allwe = allwe + torch.exp(qLis * distance)
            else:
                allwe = allwe + torch.exp(qLis * distance)
            '''
            allwe = (self.hidden_size, 2)
            '''
            #print("allwe {}".format(allwe.shape))

        if useGPU:
            cL = torch.zeros(self.hidden_size, 1, device='cuda:0')
        else:
            cL = torch.zeros(self.hidden_size, 1)
        
        '''
        if useGPU:
            cL = cL.cuda()
        '''

        for i in range(len(pois)):
            #poi = pois[i]
            longitude, latitude = pois[i]['location'].split(',')
            longitude, latitude = float(longitude), float(latitude)
            if useGPU:
                eLi = torch.tensor([[longitude, latitude]], device='cuda:0')
            else:
                eLi = torch.tensor([[longitude, latitude]])
            '''
            if useGPU:
                eLi = eLi.cuda()
            '''
            qLis = nextHidT @ self.weight @ eLi
            
            '''
            nextHidT = (self.hidden_size, 1)
            self.weight = (1, 1)
            eLi = (1, 2)
            qLis = (self.hidden_size, 2)
            '''
            #print("qLis {}".format(qLis.shape))
            #distance = math.exp(-1 * (float(pois[i]['distance'] / self.scaling))
            distance = math.exp(
                -1 * (
                    float( pois[i]['distance'] ) / self.scaling
                )
            )
            '''
            a = qLis * distance
            t1 = torch.exp(a)
            t = torch.div(t1, allwe)
            cL = cL + t * eLi.t()
            '''
            
            cL = cL + ( torch.div(torch.exp(qLis * distance), allwe) @ eLi.t() )
            #print("cL {}".format(cL.shape))
        #空间上下文提取器
        '''
        cL = (self.hidden_size, 1)
        '''  

        Length = len(x)
        '''
        if periodHid is None:
            periodHid = torch.randn(1, self.hidden_size)
            if useGPU:
                periodHid = periodHid.cuda()
        '''
        timeLocVector = x[-1]

        if useGPU:
            newUserTensor = torch.tensor([[user]], device='cuda:0')
        else:
            newUserTensor = torch.tensor([[user]])
        '''
        if useGPU:
            newUserTensor = newUserTensor.cuda()
        '''

        nextPeriodHid = self.GRUCell3(
            torch.cat(
                (
                    newUserTensor,
                    x
                ),
                dim=1
            ),
            periodHid
        )
        newNextPeriodHid = nextPeriodHid.detach()
        qhi = torch.exp(nextHidT @ newNextPeriodHid)
        '''
        if useGPU:
            qhi = qhi.cuda()
        '''
        qhh = qhh + qhi
        aH = aH + torch.div(qhi, qhh)
        cP = aH @ ( nextPeriodHid.t() )

        '''
        qhi = (self.hidden_size, self.hidden_size)
        qhh = (self.hidden_size, self.hidden_size)
        aH = (self.hidden_size, self.hidden_size)
        periodHid = (1, self.hidden_size)
        cP = (self.hidden_size, 1)

        print("aH {}".format(aH.shape))
        print("cP {}".format(cP.shape))
        print("peiodHid {}".format(periodHid.shape))
        '''
        #cP(self.hiddenze, 1)
        #周期上下文提取器

        #print('nodes {}'.format(nodes.shape))
        #print('nodes {}'.format(nodes.device))
        #print('edges {}'.format(edges.shape))
        outGATState = self.GAT(nodes, edges, useGPU=useGPU)
        #GAT hidden_stata

        '''
        projectionMatrix = [[p['location']] for p in pois]
        for i in range(len(projectionMatrix)):
            temp = projectionMatrix[i][0]
            a1, a2 = temp.split(',')
            longitude, latitude = float(a1), float(a2)
            projectionMatrix[i] = [[longitude, latitude]]
        #print(projectionMatrix)
        proMatrixTensor = torch.tensor(projectionMatrix)
        '''

        projectionMatrix = [[[n[0], n[1]]] for n in nodes]
        if useGPU:
            proMatrixTensor = torch.tensor(projectionMatrix, device='cuda:0')
        else:
            proMatrixTensor = torch.tensor(projectionMatrix)

        '''
        if useGPU:
            proMatrixTensor = proMatrixTensor.cuda()
        '''
        
        mL = torch.cat(
            (newNextHid, cL.t(), cP.t()),
            dim=1
        )

        '''
        if useGPU:
            mL = mL.cuda()
        '''
        
        '''
        nextHid = (1, self.hidden_size)
        cL = (self.hidden_size, 1)
        cP = (self.hidden_size, 1)
        mL(1, 3 * self.hidden_size)
        '''

        if useGPU:
            deepLoc = torch.randn(len(proMatrixTensor), 2, 3 * self.hidden_size, device='cuda:0')
        else:
            deepLoc = torch.randn(len(proMatrixTensor), 2, 3 * self.hidden_size)
        '''
        if useGPU:
            deepLoc = deepLoc.cuda()
        '''
        
        for i in range(len(proMatrixTensor)):
            predictionLoc = proMatrixTensor[i]
            #predictionLoc = torch.unsqueeze(proMatrixTensor[i], 0)
            ''''
            if useGPU:
                predictionLoc = predictionLoc.cuda()
            '''

            #print("pred {}".format(predictionLoc.shape))
            #print("mL {}".format(mL.shape))
            #print("pred {}".format(predictionLoc.shape))
            deepLoc[i] = predictionLoc.t() @ mL
            '''
            mL = (1, self.hidden_size)
            predictionLoc = (1, 2)
            dppeLoc[i] = (2, 3 * self.hidden_size)
            '''
        #DeepJMT 地点预测

        if useGPU:
            anw = torch.zeros(1, len(projectionMatrix), device='cuda:0')
            raw = torch.randn(len(projectionMatrix), 3 * self.hidden_size, device='cuda:0')
        else:
            anw = torch.zeros(1, len(projectionMatrix))
            raw = torch.randn(len(projectionMatrix), 3 * self.hidden_size)
        '''
        if useGPU:
            anw, raw = anw.cuda(), raw.cuda()
        '''
        for i in range(len(projectionMatrix)):
            outGAT = torch.unsqueeze(outGATState[i + pre], 0)
            #print('outGAT {}'.format(outGAT.device))
            #a = input('wait')
            '''
            if useGPU:
                outGAT = outGAT.cuda()
            '''
            
            temp = outGAT @ deepLoc[i] #测试无GAT
            #print('temp {}'.format(temp.shape))
            #print('deeploc[i] {}'.format(deepLoc[i].shape))
            
            #temp = torch.unsqueeze(deepLoc[i][0], 0)
            #temp = deepLoc[i].view(1, 3 * self.hidden_size)

            '''
            outGAT = (1, 2)
            temp = (1, 3 * self.hidden_size)
            '''
            anw[0][i] = torch.max(input=temp[0])
            raw[i] = temp[0]
            
        softAnw = torch.nn.functional.softmax(anw[0], dim=0)
        #print("anw {}".format(anw.shape))
        #print(anw)
        value, index = torch.topk(input=softAnw, k=7)
        '''
        if useGPU:
            #softAnw, index = sortAnw.cuda(), index.cuda()
            softAnw = softAnw.cuda()
            index = index.cuda()
        '''
        #结合DeepJMT和GAT
        #print("once end")

        return newNextHid, periodHid, qhh, aH, index, raw


        

        
