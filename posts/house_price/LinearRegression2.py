import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

house_train = pd.read_csv("./posts/house_price/train.csv")
house_test = pd.read_csv("./posts/house_price/test.csv")
sub_df =  pd.read_csv("posts/house_price/sample_submission.csv")

# 이상치 탐색 및 제거 
quantitative = house_train.select_dtypes(include = ['number'])
quantitative.info()
x = quantitative.iloc[:,1:-1]
x.isna().sum()
# LotFrontage, MasVnrArea, GarageYrBlt


fill_values = {
    'LotFrontage': x['LotFrontage'].mean(),
    'MasVnrArea': x['MasVnrArea'].mean(),
    'GarageYrBlt' : x['GarageYrBlt'].mean()
}
x = x.fillna(fill_values)

y = quantitative.iloc[:, -1]

model = LinearRegression()
# 모델 학습
model.fit(x, y)

# 회귀 직선의 기울기와 절편 
slope = model.coef_
intercept = model.intercept_
y_pred = model.predict(x)


# 결측치 확인 
quantitative1 = house_test.select_dtypes(include = ['number'])
test_x = quantitative1.iloc[:,1:]
test_x.isna().sum()

test_x = test_x.fillna(test_x.mean())

y_hat = model.predict(test_x)
sub_df["SalePrice"] = y_hat
sub_df.to_csv("posts/house_price/sample_submission3.csv", index = False)



# 3차원 평면 그래프 그리는 코드입니다
from mpl_toolkits.mplot3d import Axes3D

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='Data points')

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope1 * GrLivArea_vals + slope2 * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5)

# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.legend()
plt.show()




