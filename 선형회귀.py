# 판다스 데이터 정리
import pandas as pd # 판다스 라이브러리
import csv # csv 라이브러리

tr = pd.read_csv("student.csv", encoding = 'utf-8')


student_x_data = tr.loc[:, ['age','activities','famsup','paid','internet','Medu','Fedu','studytime','schoolsup','G3']]
student_y_data = tr.loc[:,['pass']]

################################
### 데이터 확인을 위한 명령어###
################################

# tr.head()
# student_x_data.head(40)
# student_y_data.head()


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1) # 재 실행 시에도 동일한 데이터 생성을 위해 랜덤 시드 값 지정
x_data = student_x_data
y_data = student_y_data


############################
##########파이토치##########
############################

import torch
import torch.nn as nn
import torch.nn.functional as F

# 파이토치 텐서 훈련 데이터 생성
x_train = torch.FloatTensor(x_data.values).unsqueeze(dim=1)
# unsqueeze(dim=1): (1000,) => (1000, 1)

y_train = torch.FloatTensor(y_data.values).unsqueeze(dim=1)

# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Sequential(
    nn.Linear(10,8), # 입력계층 = 1, 은닉계층1 = 5
    nn.ReLU(), # ReLu 활성화 함수
    nn.Linear(8,3), # 은닉 계층2 = 3
    nn.ReLU(), # ReLu 활성화 함수
    nn.Linear(3,1) # 출력계층 = 1
)
print(list(model.parameters()))

# optimizer 설정. 경사 하강법 SGD를 사용하고 학습 속도를 의미하는 lr은 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 전체 훈련 데이터에 대해 경사 하강법을 100회 반복
nb_epochs=1000

ep_data = []
loss_data = []

for epoch in range(nb_epochs):
    # H(x) 계산
    prediction = model(x_train)
    
    # 손실함수 계산
    loss = F.mse_loss(prediction, y_train)

    
    # 역전파 수행
    optimizer.zero_grad() # 기울기(gradient)를 0으로 초기화
    loss.backward() # 손실함수를 미분하여 기울기 계산
    optimizer.step() # W와 b를 업데이트
    
    # 10번 반복마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} loss : {:.6}'.format(epoch, nb_epochs, loss.item()))

        ep_data.append([epoch])
        loss_data.append([loss.item()])


print(ep_data)
print(loss_data)

print("학습 종료")

list = [15,0,0,1,1,1,1,2,1,20]
new_var = torch.FloatTensor([list])

pred_y = model(new_var) # 순전파 연산


print("훈련 후 입력이 10일 때의 예측값 :", pred_y*100)