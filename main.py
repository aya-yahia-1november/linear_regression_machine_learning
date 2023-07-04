import numpy as np

import matplotlib.pylab as plt

import pandas as pd

def drow_line(x,b0,b1):
    n=len(x)
    y_hat=np.zeros(n)
    for i in range(n):
        y_hat[i]=b0+b1*x[i]
    return y_hat
def compute_sum_error(y,b0,b1,x):
    n=len(x)
    sum_error=0
    for i in range(n):
        sum_error+=(y[i]-(b0+b1*x[i]))**2
    return sum_error
def calculate_b1(x,x_bar,y,y_bar):
    n=len(x)
    sum1=0
    sum2=0
    for i in range(n):
        sum1+=(x[i]+x_bar)*(y[i]-y_bar)
    for i in range(n):
        sum2+=(x[i]-x_bar)**2
    return sum1/sum2
def calcolate_b0(y_bar,b1,x_bar):
    return y_bar-b1*x_bar
def predict(b0,b1,x):
    return (b0+b1*x)

data=pd.read_csv("real data.csv")
x=data["SAT"]
y=data["GPA"]
"""x=np.array([1,1.7,2,2.5,3,3.2])
y=np.array([250,300,480,630,730,820])"""
plt.scatter(x,y)

x_bar=np.mean(x)
y_bar=np.mean(y)

b1=calculate_b1(x,x_bar,y,y_bar)
b0=calcolate_b0(y_bar,b1,x_bar)
y_hat=drow_line(x,b0,b1)
print(predict(b0,b1,800))
print(compute_sum_error(y,b0,b1,x))
plt.plot(x,y_hat)
plt.show()