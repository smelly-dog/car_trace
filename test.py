import torch, string
from Model.model import POI

if __name__ == '__main__':
    pois = POI(114.120514, 22.810388)
    projectionMatrix = [[x['location']] for x in pois]
    for i in range(len(projectionMatrix)):
        temp = projectionMatrix[i][0]
        a1, a2 = temp.split(',')
        longitude, latitude = float(a1), float(a2)
        projectionMatrix[i] = [longitude, laatitude]
    print(projectionMatrix)