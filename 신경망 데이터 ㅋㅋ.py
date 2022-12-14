# csv 파일 불러오기
import pandas as pd
import csv

# 파일명
file_name = 'student.csv'
data = pd.read_csv(file_name)

# 데이터 크기 확인 395개의 데이터 29개의 특징, 1개의 라벨데이터
# data.shape
# # 상위 5개 데이터 확인
# data.head(5)

# 필요한 데이터 전처리
X_data = data.loc[:, ['age','activities','famsup','paid','internet','Medu','Fedu','studytime','schoolsup','G3']]
Y_data = data['pass']

# 훈련데이터 크기 확인
# X_data.shape, Y_data.shape

# #X_data, Y_data 확인
# X_data.head(10)
# Y_data.head(10)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

# 토치 플롯 텐서로 변환 - 훈련 데이터
X_train = torch.FloatTensor(X_data.values[10:])
Y_train = torch.FloatTensor(Y_data.values[10:]).long()

# 토치 플롯 텐서로 변환 - 테스트 데이터
X_test = torch.FloatTensor(X_data.values[:10])
Y_test = torch.FloatTensor(Y_data.values[:10]).long()

# 데이터 크기 확인, 훈련데이터 385, 테스트 데이터 10
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# 학습용 딥러닝 모델 클래스 생성
class FC_NN(nn.Module):
    def __init__(self):
        super(FC_NN, self).__init__()
        self.input_layer = nn.Linear(10, 5)
        self.hidden_layer = nn.Linear(5, 2)
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        output = self.hidden_layer(x)
        
        return output

# 모델 생성, 모델 옵션 지정    
model = FC_NN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
epochs = 100


# 학습 단계
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X_train)
    loss = loss_fn(prediction, Y_train)
    loss = F.cross_entropy(prediction, Y_train) #
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss : .4f}')





# 학습 검증 단계
import numpy as np

# 테스트 데이터를 넣으면 합격할 확률을 알려준다.
def model_predict(data):
    # 모델에 값 예측
    pred = model(data)
    print(type(data))
    # softmax로 확률로 변경
    for i, j in pred:
        one = np.array([i.item(), j.item()])
        exp_one = np.exp(one)
        sum_exp = np.sum(exp_one)
        output = exp_one / sum_exp
    
    return 100*output[1]


#테스트 데이터로 합격할 확률과 결과 비교하기
# for i, j in zip(X_test, y_test):
#     i = i.view(-1,10)
#     print(f"라벨 데이터가 {j} 인 학생의 합격할 확률은 {model_predict(i) : .2f} % 입니다.")

list = [15,0,0,1,1,1,1,2,1,9]
new_var = torch.FloatTensor([list])

# list = [18,0,0,0,0,4,4,2,1,6]
# # print(f"라벨 데이터가 20 인 학생의 합격할 확률은 {model_predict(list) : .2f} % 입니다.")


print(f"라벨 데이터가 9인 학생의 합격할 확률은 {model_predict(new_var) : .2f} % 입니다.")