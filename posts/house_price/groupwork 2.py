import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

house_train = pd.read_csv("./posts/house_price/train.csv")

# 이상치 탐색 및 제거 
house_train = house_train.query("GrLivArea <= 4500")

model = LinearRegression()

x = house_train[["GrLivArea"]]
y = house_train["SalePrice"]
model.fit(x, y)
slope1 = model.coef_[0]
#slope2 = model.coef_[1]
intercept = model.intercept_
y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

house_test = pd.read_csv("posts/house_price/test.csv")
test_x = house_test[["GrLivArea"]]
y_hat = model.predict(test_x)

sub_df = pd.read_csv("posts/house_price/sample_submission.csv")
sub_df["SalePrice"] = y_hat
sub_df.to_csv("posts/house_price/sample_submission3.csv", index = False)
