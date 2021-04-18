import pandas as pd 
import ../Model

'''
data = pd.read_csv("./data/data.csv")
#print(type(data))
data = data.sort_values(by=['ObjectID', 'StartTime', 'StopTime'], ascending=[True, True, True])
#print(data.head(5))
data.to_csv(path_or_buf='./data/train_new.csv', index=False)

'''

a = Model.POI(0.0, 1.0)