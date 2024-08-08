house_train = pd.read_csv("posts/house_price/train.csv")
house_train.info()

import pandas as pd
# pairplot
df = house_train[['SalePrice', 'YearBuilt', 'LotFrontage', 'LotArea', 'OverallQual', \
                 'MasVnrArea', 'BsmtQual', 'TotalBsmtSF', 'GrLivArea', 'FullBath', \
                 'TotRmsAbvGrd', 'GarageArea', 'PoolArea']]

plt.figure(figsize=(12, 10))
sns.pairplot(df, plot_kws={'s': 5})
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()
plt.clf()


# histogram
house_train['BsmtQual'].value_counts()
# Ex	Excellent (100+ inches)	
# Gd	Good (90-99 inches)
# TA	Typical (80-89 inches)
# Fa	Fair (70-79 inches)
# Po	Poor (<70 inches
# NA	No Basement

house_train['SalePrice'].isna().sum()
house_train['MSZoning'].isna().sum()
house_train['Functional'].isna().sum()
df2 = house_train.groupby(['MSZoning', 'Functional'], as_index = False)['SalePrice'] \
                 .mean().sort_values('SalePrice', ascending = False)
sns.barplot(data = df2, x = "MSZoning", y = "SalePrice", hue = "Functional")
plt.show()           
plt.clf()


