# y = 2x + 3의 그래프를 그려보세요
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
a = 2
b = 3
x = np.linspace( -5, 5, 100)
y = a * x + b

plt.plot(x, y, color = "blue")
plt.axvline(x = 0, color = "black")
plt.axhline(y = 0, color = "black")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()

x = np.linspace(0, 8, 100)

a = 70
b = 10
y = a * x + b
house_train = pd.read_csv("./posts/house_price/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]]
my_df["SalePrice"] = my_df["SalePrice"]
plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])
plt.plot(x, y, color = "blue")
plt.tight_layout()
plt.show()
plt.clf()

# 테스트 집 정보 가져오기 
a = 70
b = 10
house_test = pd.read_csv("posts/house_price/test.csv")
house_test["SalePrice"] = (a * house_test["BedroomAbvGr"] + b)
sub_df = pd.read_csv("posts/house_price/sample_submission.csv")
sub_df["SalePrice"] = house_test['SalePrice']
sub_df.to_csv("posts/house_price/sample_submission3.csv", index = False)

# 직선 성능 평가
a = 70
b = 10

# y_hat
y_hat = (a * house_train["BedroomAbvGr"] + b)
# y는 어디에 있는가?
y = house_train["SalePrice"]
np.sum(np.abs(y - y_hat))  # 절댓값
np.sum((y-y_hat) ** 2)  # 제곱값 


# 회귀분석 직선 구하기 
# !pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌. 
model.coef_      # 기울기
model.intercept_  # 절편

# 회귀 직선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# 회귀모델을 통한 집값 예측 
# 필요한 데이터 불러오기 

house_train = pd.read_csv("./posts/house_price/train.csv")


x = house_train[["BedroomAbvGr"]]

# x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1)
y = house_train[["SalePrice"]]

model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")
y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

np.sum(np.abs(y - y_pred))

y_pred = model.predict(x)
test_x = house_test[["BedroomAbvGr"]]
y_hat = model.predict(test_x)  # test 셋에 대한 집값

sub_df = pd.read_csv("posts/house_price/sample_submission3.csv")
sub_df["SalePrice"] = y_hat   # SalePrice 바꿔치기 

# csv 파일로 내보내기 
sub_df.to_csv("posts/house_price/sample_submission3.csv", index = False)


# y는 어디에 있는가?
import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def line_perform(x):
    y_hat = (x[0] * house_train["BedroomAbvGr"] + x[1]) 
    y = house_train["SalePrice"]
    return np.mean(y-y_hat) ** 2)
# 초기 추정값 
initial_guess = [0, 0]

result = minimize(line_perform, initial_guess)


# 첫번째 x^2 + 3 
def my_f(x):
    return x ** 2 + 3 
my_f(3)
initial_guess = [0]
result = minimize(my_f, initial_guess)
print("최소값", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 두번째: x^2 + 3
def my_f2(x):
    return x[0] ** 2 + x[1] ** 2 + 3 
initial_guess = [1, 2]

result = minimize(my_f2, initial_guess)
print("최소값", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 세번째: (x-1)^2 + (x-2)^2 + (x-4)^2 + 7 
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7

initial_guess = [0, 0, 0]
result = minimize(my_f3, initial_guess)
print("최소값", result.fun)
print("최소값을 갖는 x 값:", result.x)



