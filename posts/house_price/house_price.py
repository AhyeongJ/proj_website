import pandas as pd
import numpy as np

house_train = pd.read_csv("posts/house_price/train.csv")
house_trian.info()
house_test = pd.read_csv("posts/house_price/test.csv")
house_test.shape
price_mean = house_train["SalePrice"].mean()
price_mean

# sub_df에 있는 SalePrice값 다 평균으로 바꾸기 
sub_df = pd.read_csv("posts/house_price/sample_submission.csv")
sub_df['SalePrice'] = price_mean
sub_df

# 기존 csv에 바뀐 csv로 바꿔치기 
sub_df.to_csv("posts/house_price/sample_submission.csv", index = False)



## YearBuilt기준으로 SalePrice 예측 
house_train = house_train[["Id", "YearBuilt", "To"]]

# 연도별 평균 
house_mean = house_train.groupby('YearBuilt', as_index = False)\
                        .agg(mean = ('SalePrice', 'mean'))

house_test = house_test[["Id", "YearBuilt"]]

house_test = pd.merge(house_test, house_mean, how ='left', on = 'YearBuilt')
house_test = house_test.rename(columns = {'mean' : 'SalePrice'})

house_test.isna().sum()  # na 값 세기 
house_test.loc[house_test['SalePrice'].isna()] # na값 보기 

 # na 값 채우기 
price_mean = house_train["SalePrice"].mean()
house_test['SalePrice'] = house_test['SalePrice'].fillna(price_mean) 

#SalePrice 바꿔치기 및 저장 
sub_df['SalePrice'] = house_test['SalePrice']
sub_df.to_csv("posts/house_price/sample_submission2.csv", index = False)


