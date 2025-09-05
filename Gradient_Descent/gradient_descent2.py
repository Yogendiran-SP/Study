import numpy as np
import pandas as pd
from sklearn import linear_model
import math as m

def gradient_descent(x,y):
    m_curr = b_curr = 0
    n=len(x)
    iterations = 10000
    learning_rate = 0.0001
    prev_cost = float("inf")
    for i in range(iterations):
        y_pred = m_curr*x + b_curr
        curr_cost = (1/n)*sum([val**2 for val in (y-y_pred)])
        if m.isclose(prev_cost, curr_cost, rel_tol=1e-20, abs_tol=1e-20):
            print(f"Converged at iteration {i}")
            break
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr -= learning_rate*md
        b_curr -= learning_rate*bd
        print(f"m: {m_curr}, b: {b_curr}, cost: {curr_cost}, i: {i}")
        prev_cost = curr_cost

df = pd.read_csv("test_scores.csv")
x=df['math'].to_numpy()
y=df['cs'].to_numpy()
gradient_descent(x,y)
