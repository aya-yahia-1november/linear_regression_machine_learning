import numpy as np
import pandas as pd
import matplotlib as plt
from multi_de_module import *
data = pd.read_csv("real_data.csv")
x = data[['size', 'year']]
y = data['price']
betas = find_betas(x, y)
y_hat=fit_module(x,betas)
SSE=square_error(y,y_hat)
sst=square_error(y,y.mean())
ssr=sst-SSE
print(SSE)
print(r_square(ssr,sst))

"""betas = find_betas(x, y)
print(betas)
print(predict([[650, 2017]], betas))"""