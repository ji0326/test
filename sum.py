import sqlite3
from flask import Flask, render_template, request
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd # 판다스 라이브러리
import csv # csv 라이브러리
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import base64
from io import BytesIO
from matplotlib.figure import Figure




#플라스크 객체 생성
app=Flask(__name__)

conn = sqlite3.connect("music.db")
cursor = conn.cursor()


## 시작 페이지 ##
@app.route('/home')
def home():
    return render_template('home.html')


## 멜론 웹 크롤링을 위한 웹 ##
@app.route('/melon',  methods=['POST'])
def melon():
    return render_template('크롤링.html')

## 멜론 차트 웹 크롤링 하기 ##
@app.route('/melon1', methods=['POST'])
def melon1():

    web_url = request.form['id']

    #### tablename에 table이 있는 경우 그냥 데이터 삭제후 다시 웹 크롤링
    def chart_balad_save():
        url = web_url # 멜론 발라드 차트 url 
        hdr = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36"}
        req = urllib.request.Request(url, headers=hdr)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()


        
        lst50 = soup.select('.lst50, .lst100')

        genre = []

        paging = soup.find('div', class_ = 'calendar_prid').get_text().replace("\n","")
        genre.append([paging])

        cursor.execute("DELETE FROM {}".format(genre[0]))

        melonList = []
        for i in lst50:
            ranking = (i.select_one("td > div > span.rank").text+"위") # 멜론 사이트에서 발라드 순위 가져와서 temp에 추가
            title = (i.select_one("td > div > div > div.ellipsis.rank01").a.text) # 멜론 사이트에서 노래 제목 가져오서 temp에 추가
            singer = (i.select_one('td > div > div > div.ellipsis.rank02').a.text) # 멜론 사이트에서 가수 이름 가져와서 temp에 추가
            album = (i.select_one('td > div > div > div.ellipsis.rank03').a.text) # 멜론 사이트에서 앨범 이름 가져와서 temp에 주가
            url_title = "https://www.youtube.com/results?search_query={}".format((i.select_one("td > div > div > div.ellipsis.rank01").a.text.replace(" ","+")))
            url_singer = "https://www.youtube.com/results?search_query={}".format((i.select_one('td > div > div > div.ellipsis.rank02').a.text.replace(" ","+")))
            melonList.append([ranking, title, singer, album]) # 멜론리스트에 temp에 있는 것들 추가
            
            inser='insert into {} (ranking, title, singer, album, youtube_title, youtube_singer) values(?,?,?,?,?,?)'.format(genre[0])
            cursor.execute(inser,(ranking, title, singer, album, url_title, url_singer))
        conn.commit()


    #### teblename에 없는 table일 경우 table을 생성하고 데이터 저장
    def chart_balad_save1():
        url = web_url # 멜론 발라드 차트 url 
        hdr = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36"}
        req = urllib.request.Request(url, headers=hdr)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()

        lst50 = soup.select('.lst50, .lst100')

        genre = []

        paging = soup.find('div', class_ = 'calendar_prid').get_text().replace("\n","")
        genre.append([paging])


        inser='insert into tablename (name) values(?)'
        cursor.execute(inser,(genre[0]))
        conn.commit()


        insert = 'create table {} (ranking REAL, title TEXT, singer TEXT, album TEXT, youtube_title TEXT, youtube_singer TEXT)'.format(genre[0])
        cursor.execute(insert)
        conn.commit()

        melonList = []
        for i in lst50:
            ranking = (i.select_one("td > div > span.rank").text+"위") # 멜론 사이트에서 발라드 순위 가져와서 temp에 추가
            title = (i.select_one("td > div > div > div.ellipsis.rank01").a.text) # 멜론 사이트에서 노래 제목 가져오서 temp에 추가
            singer = (i.select_one('td > div > div > div.ellipsis.rank02').a.text) # 멜론 사이트에서 가수 이름 가져와서 temp에 추가
            album = (i.select_one('td > div > div > div.ellipsis.rank03').a.text) # 멜론 사이트에서 앨범 이름 가져와서 temp에 주가
            url_title = "https://www.youtube.com/results?search_query={}".format((i.select_one("td > div > div > div.ellipsis.rank01").a.text.replace(" ","+")))
            url_singer = "https://www.youtube.com/results?search_query={}".format((i.select_one('td > div > div > div.ellipsis.rank02').a.text.replace(" ","+")))
            melonList.append([ranking, title, singer, album]) # 멜론리스트에 temp에 있는 것들 추가
            
            inser='insert into {} (ranking, title, singer, album, youtube_title, youtube_singer) values(?,?,?,?,?,?)'.format(genre[0])
            cursor.execute(inser,(ranking, title, singer, album, url_title, url_singer))
        conn.commit()


    url = web_url # 멜론 발라드 차트 url 
    hdr = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36"}
    req = urllib.request.Request(url, headers=hdr)
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')


    genre = []

    paging = soup.find('div', class_ = 'calendar_prid').get_text().replace("\n","")
    genre.append([paging])


    conn=sqlite3.connect('music.db')
    conn.row_factory=sqlite3.Row
    cursor=conn.cursor()

    cursor.execute('select*from tablename where name =? ',(paging,))
    row = cursor.fetchone()
    conn.close()

    if row!=None:
        if paging == row['name']:
            chart_balad_save()
    else:
        chart_balad_save1()


    conn=sqlite3.connect('music.db')
    conn.row_factory=sqlite3.Row
    cursor=conn.cursor()
    #계정 id 레코드 읽기
    #cursor.execute('select name from tablename')
    cursor.execute('select*from tablename')
    rows=cursor.fetchall()

    ins = []

    for row in rows:
        ins.append(row['name'])

    conn.close()
    return render_template('melon1.html', values = ins)

## 데이터 조회, 가수명으로 조회, 순위로 조회 웹 페이지 ##
@app.route('/melon11')
def melon11():
    conn=sqlite3.connect('music.db')
    conn.row_factory=sqlite3.Row
    cursor=conn.cursor()
    #계정 id 레코드 읽기
    #cursor.execute('select name from tablename')
    cursor.execute('select*from tablename')
    rows=cursor.fetchall()

    ins = []

    for row in rows:
        ins.append(row['name'])

    conn.close()
    return render_template('melon1.html', values = ins)



## 데이터 조회 출력 웹 페이지 ##
@app.route('/melon2', methods=['POST'])
def melon2():

    ch = request.form['tablename']
    table = []
    table.append([ch])

    conn = sqlite3.connect("music.db")
    # conn.row_factory=sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT * FROM {} '.format(table[0]))
    rows = cur.fetchall()
        
    conn.close()
    
    nametable = table[0][0]

    return render_template('data.html', rows=rows, nametable = nametable)




## 가수명으로 조회하기위한 웹 페이지
@app.route('/melon3', methods=['POST'])
def melon3():
    ch = request.form['tablename']
    table = []
    table.append([ch])
    nametable = table[0][0]
    return render_template('melon3.html', nametable=nametable)


## 가수명 조회 후 출력하기 위한 웹 페이지 ##
@app.route('/show1', methods=['POST'])
def show1():
    na = request.form['name']

    ch = request.form['tablename']
    print(ch)
    table = []
    table.append([ch])
    nametable = table[0][0]
    
    conn=sqlite3.connect('music.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # sele = 'select*from {} where singer=?'.format(table[0])
    # cursor.execute(sele,(na,))
    cursor = conn.cursor()
    cursor.execute('''select * from {} where singer=?'''.format(table[0]),(na,))
    row = cursor.fetchall()

    conn.close()

    return render_template('data1.html', rows=row, nametable=nametable)



## 순위로 조회하기 위한 웹 페이지
@app.route('/melon4', methods=['POST'])
def melon4():
    ch = request.form['tablename']
    table = []
    table.append([ch])
    nametable = table[0][0]
    return render_template('melon4.html', nametable = nametable)

## 순위로 조회 후 출력하기 위한 웹 페이지 ##
@app.route('/show2', methods=['POST'])
def show2():
    rank = request.form['ranking']
    ch = request.form['tablename']
    print(ch)
    table = []
    table.append([ch])
    nametable = table[0][0]

    conn=sqlite3.connect('music.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''select*from {} where ranking=?'''.format(table[0]),(rank,))
    row = cursor.fetchall()

    conn.close()

    return render_template('data2.html', rows=row, nametable = nametable)



## ============머신러닝 웹 페이지 ====================##

#-------------------------로그인 페이지-----------------------

#로그인 폼 렌더링
#/ 경로 요청 시 실행 함수
@app.route('/mach1', methods=['POST']) #요청 경로 지정
def login(): #실행 함수 작성
    return render_template('login.html') #login.html 파일 렌더링


#로그인 처리
#/login1 경로 요청 시 실행 함수
@app.route('/login1', methods=['POST']) #요청 경로 지정, form태그의 매개변수 값 전달 방식_POST
def login1(): #실행 함수 작성
    #폼 입력값 가져오기
    idn=request.form['id']
    pwd=request.form['passwd']
    
    #데이터베이스 연결
    conn=sqlite3.connect('member.db')
    conn.row_factory=sqlite3.Row
    cursor=conn.cursor()
    #계정 id 레코드 읽기
    cursor.execute('select*from 회원정보 where id=?',(idn,))
    row=cursor.fetchone()
    conn.close()
    
    #계정/비번 조사
    if row!=None:
        if idn==row['id']and pwd==row['passwd']:
            return render_template('registration.html')
        
    return("<h2>로그인 에러!!</h2>")



#회원가입 폼 렌더링
#/signup 경로 요청 시 실행 함수
@app.route('/signup') #요청 경로 지정
def signup(): #실행 함수 작성
    return render_template('signup.html') #signup.html 파일 렌더링

#회원가입 처리
#/register 경로 요청 시 실행 함수
@app.route('/register',methods=['POST']) ##요청 경로 지정, form태그의 매개변수 값 전달 방식_POST
def register(): #실행 함수 작성
    
    #폼 입력값 가져오기
    idn=request.form['id']
    pwd=request.form['passwd']
    name=request.form['name']
    age=request.form['age']
    act=request.form['act']
    fam=request.form['fam']
    pain=request.form['pain']
    inter=request.form['inter']
    Medu=request.form['Medu']
    Fedu=request.form['Fedu']
    study=request.form['study']
    sch=request.form['sch']

    #데이터베이스 연결
    conn=sqlite3.connect('member.db')
    cursor=conn.cursor()
    #데이터베이스 등록(삽입)
    cursor.execute('''
    insert into 회원정보 (id,passwd,name,age,activities,famsup,pain,internet,Medu,Fedu,studytime,schoolsup) values(?,?,?,?,?,?,?,?,?,?,?,?)'''
    , (idn,pwd,name,age,act,fam,pain,inter,Medu,Fedu,study,sch))
    conn.commit()
    conn.close()
    
    return render_template('login.html') #login.html파일 렌더링


# 판다스 데이터 정리


tr = pd.read_csv("student.csv", encoding = 'utf-8')

## 필요한 데이터 분리 ##
student_x_data = tr.loc[:, ['age','activities','famsup','paid','internet','Medu','Fedu','studytime','schoolsup','G3']]
student_y_data = tr.loc[:,['pass']]

################################
### 데이터 확인을 위한 명령어###
################################

# tr.head()
# student_x_data.head(40)
# student_y_data.head()



np.random.seed(1) # 재 실행 시에도 동일한 데이터 생성을 위해 랜덤 시드 값 지정
x_data = student_x_data
y_data = student_y_data


############################
##########파이토치##########
############################


# 파이토치 텐서 훈련 데이터 생성
x_train = torch.FloatTensor(x_data.values).unsqueeze(dim=1)
# unsqueeze(dim=1): (1000,) => (1000, 1)

y_train = torch.FloatTensor(y_data.values).unsqueeze(dim=1)

# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=10, output_dim=1.
model12 = nn.Sequential(
    nn.Linear(10,8), # 입력계층 = 1, 은닉계층1 = 5
    nn.ReLU(), # ReLu 활성화 함수
    nn.Linear(8,3), # 은닉 계층2 = 3
    nn.ReLU(), # ReLu 활성화 함수
    nn.Linear(3,1) # 출력계층 = 1
)
print(list(model12.parameters()))

# optimizer 설정. 경사 하강법 SGD를 사용하고 학습 속도를 의미하는 lr은 0.05
optimizer = torch.optim.SGD(model12.parameters(), lr=0.05)

# 전체 훈련 데이터에 대해 경사 하강법을 100회 반복
nb_epochs=5000

ep_data = []
loss_data = []

for epoch in range(nb_epochs):
    # H(x) 계산
    prediction = model12(x_train)
    
    # 손실함수 계산
    loss = F.mse_loss(prediction, y_train)
    
    # 역전파 수행
    optimizer.zero_grad() # 기울기(gradient)를 0으로 초기화
    loss.backward() # 손실함수를 미분하여 기울기 계산
    optimizer.step() # W와 b를 업데이트
    
    # 10번 반복마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} loss : {:.6}'.format(epoch, nb_epochs, loss.item()))

        ep_data.append([epoch])
        loss_data.append([loss.item()])



print("학습 종료")

########################
###간단한 신경망 모델###
########################


# csv 파일 불러오기


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

epoch_data = []
loss1_data = []

# 학습 단계
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X_train)
    loss = loss_fn(prediction, Y_train)
    loss = F.cross_entropy(prediction, Y_train) #
    loss.backward()
    optimizer.step()
    if (epoch) % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss : .4f}')

        epoch_data.append([epoch])
        loss1_data.append([loss.item()])





# 학습 검증 단계


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


## 회원 정보를 입력 받고 점수를 입력 받은 뒤 합격 확률과 loss 시각화 그래프를 출력하기 위한 웹 페이지 ##
@app.route('/model', methods=['POST'])
def model1():
    score = request.form['score']
    pl_name = request.form['name']
    score_data = []
    conn=sqlite3.connect('member.db')
    conn.row_factory=sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('select*from 회원정보 where name =?',(pl_name,))
    rows=cursor.fetchall()
    conn.close()

    print(pl_name)

    data_list=[int(rows[0]['age']), int(rows[0]['activities']), int(rows[0]['famsup']),int(rows[0]['pain']),int(rows[0]['internet']),
    int(rows[0]['Medu']),int(rows[0]['Fedu']),int(rows[0]['studytime']),int(rows[0]['schoolsup']),int(score)]

    print(data_list)

    
    new_var = torch.FloatTensor([data_list])
    pred_y = model12(new_var)
    XX_DA = model_predict(new_var)
    print(f"라벨 데이터가 9인 학생의 합격할 확률은 {model_predict(new_var) : .2f} % 입니다.")

    return render_template('model.html', play_name = pl_name, play_data1 = pred_y*100, play_data2 = XX_DA, map1=map)
   

## 선형 회귀분석 모델 loss 값 출력하기 위한 웹 페이지 ##
@app.route("/loss1", methods=['POST'])
def loss1():
    
    
    fig = Figure()
    ax = fig.subplots()
    ax.plot(ep_data, loss_data)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return f"<img src='data:image/png;base64,{data}'/>"

## 간단한 신경망 모델 loss 값 출력하기 위한 웹 페이지 ##
@app.route("/loss2", methods=['POST'])
def loss2():    
    fig = Figure()
    ax = fig.subplots()
    ax.plot(epoch_data, loss1_data)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return f"<img src='data:image/png;base64,{data}'/>"


# 간단한 신경망 모델 loss 값 확인
print(loss1_data)
print(epoch_data)

# 선형 회귀분석 모델 loss 값 확인
print("\n")
print(ep_data)
print(loss_data)



#플라스크 서버 구동
if __name__=='__main__':
    app.run()  #실행되는 컴퓨터 IP 주소, 포트번호 5000
    