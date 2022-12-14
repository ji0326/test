import json
from collections import OrderedDict
import pprint

f=open("감정데이터.json",'rt', encoding='UTF8').read()
data = json.loads(f)

x_data=[]
y_data=[]

for i in range(len(data)):
    x_data.append([data[i]["word"]])
    y_data.append(data[i]["polarity"])

print(x_data)

#print(y_data)