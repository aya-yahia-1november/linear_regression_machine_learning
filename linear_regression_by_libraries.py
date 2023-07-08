import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("real_data.csv")
x = data[['size', 'year']]
y = data['price']
linear_regression=LinearRegression()
linear_regression.fit(x,y)
print(linear_regression.score(x,y))
print(linear_regression.predict([[643.09,2023]]))