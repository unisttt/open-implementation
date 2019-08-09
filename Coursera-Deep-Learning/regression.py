import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myhelper as helper

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = helper.numerical_gradient(f, x)
        x -= lr * grad

def linear_regression(X, y, params):
    alpha = 0.1
    m = len(y)
    iteration = 100
    
    plt.subplot(121)
 
    ax2 = plt.subplot(1,2,2)
    ax2.plot(1/2, 19/14, "rx")

    # GD
    for i in range(iteration):
        predict = h_theta(X, params)
        cost1 = np.sum(predict - y)
        cost2 = np.sum((predict - y) * X[:,1])
        params = params - alpha * (1/m) * np.array([cost1, cost2])
        if i % (iteration / 10) == 0:
            #plt.plot(X[1, :], h_theta(X, params), label=str(i+1))
            continue
        ax2.plot(params[0], params[1], "s")
        

    ax1 = plt.subplot(1,2,1)
    ax1.plot(X[:, 1], y, "s")
    plt.plot(X[:, 1], h_theta(X, params))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid()

    ax2.set_xlabel("theta0")
    ax2.set_ylabel("theta1")
    #ax2.legend()
    
    
    plt.show()

def j_theta(X, y, params):
    predict = h_theta(X, params)
    sqrerr = (predict - y)**2
    
    return (1/2 * len(y)) * sum(sqrerr)

def j_theta2(params, xvar, y):
    sigma = np.sum((y - h_theta2(params, xvar))**2)
    
    return sigma / (2*y.size)

def h_theta(X, params):
    return np.dot(X, params)

def h_theta2(params, xvar):
    dot = np.dot(xvar.T, params)
    return np.sum(dot)

def main():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([2.0, 3.0, 6.0])
    theta0 = 0.0
    theta1 = 0.0
    theta2 = 0.0
    params = np.array([theta0, theta1])
    
    x0 = np.array([1.0, 1.0, 1.0])
    x1 = np.array([1.0, 2.0, 4.0])
    x2 = np.array([3.0, 4.0, 6.0])
    X = np.array([x0, x1]).T
    
    params2 = np.array([theta0, theta1, theta2])
    xvar2 = np.array([x0, x1, x2])

    linear_regression(X, y, params)
    #multivariate_linear_regression(params, xvar, y)
    #multivariate_linear_regression(params2, xvar2, y)
    

if __name__ == "__main__":
    main()