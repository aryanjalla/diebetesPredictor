#lmaoo
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

import quandl
af = pd.read_csv('C://Users//Aryan Jalla//OneDrive//Documents//diabetes_csv.csv')
af.replace('tested_negative',0) 
af.replace('tested_positive',1) 
# af = pd.read_csv("C://Users//Aryan Jalla//OneDrive//Documents//DiabetesAtlasData.csv") 
# df = quandl.get('ODA/IND_LP')
# df = af.iloc[::-1]
print(af.head())


# af.hist()
# plt.show()

# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# msk = np.random.rand(len(af)) < 0.8
# train = af[msk]
# test = af[~msk]


# # LInear regression wuhu
# from sklearn import linear_model
# pavan = linear_model.LinearRegression()
# x = np.asanyarray(train[['preg','plas','skin','insu','mass','pedi','age']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# pavan.fit (x, y)
# # The coefficients
# print ('Coefficients: ', pavan.coef_)

# y_hat= pavan.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f"
#       % np.mean((y_hat - y) ** 2))

# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % pavan.score(x, y))