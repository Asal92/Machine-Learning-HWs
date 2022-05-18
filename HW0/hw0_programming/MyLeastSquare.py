
"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. final weight vector w
    2. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyLeastSquare(X,y) function.

"""

# Header
import numpy as np

# Solve the least square problem by setting gradients w.r.t. weights to zeros
def MyLeastSquare(X,y):
    # placeholders, ensure the function runs
    w = np.array([1.0,-1.0])
    error_rate = 1.0
    pred = 0
    # calculate the optimal weights based on the solution of Question 1
    #w = (XT.X)-1 . (XT.y)
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    # compute the error rate
    for xt,yt in zip(X,y):
        if (np.dot(yt, (np.matmul(w , xt)))) <= 0:
            pred += 1
        
    error_rate = pred / len(X)

    return (w,error_rate)
