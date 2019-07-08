import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myhelper as helper

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = helper.numerical_gradient(f, x)
        x -= lr * grad

def multivariate_linear_regression(params, xvar, y):
    lr = 0.1
    m = len(y)
    h = 1e-4
    grad = np.zeros_like(params)
    step_num = 1000
    print(params.size)
    print(grad)
    print("-"*20)
    for i in range(step_num):
        for idx in range(params.size):
            tmp_val = params[idx]
            #for xi in range(xvar)
            params[idx] = tmp_val + h
            fxh1 = j_theta2(params, xvar, y)
            fxh1 = np.sum(fxh1)
            
            params[idx] = tmp_val - h
            fxh2 = j_theta2(params, xvar, y)
            fxh2 = np.sum(fxh2)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            params[idx] = tmp_val

        params -= lr * grad
        if i < 10:
            plt.plot(xvar[1], h_theta2(params, xvar), "-", label=str(i))
    
            print("{}: {}".format(i, params))
        elif i == 999:
            plt.plot(xvar[1], h_theta2(params, xvar), label=str(i))
            print("{}: {}".format(i, params))

    plt.plot(xvar[1], y, 's')
    plt.plot(xvar[1], h_theta2(params, xvar))
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    #print(h_theta2(params, xvar))
    #X0, X1, X2 = np.meshgrid(xvar[0], xvar[1], xvar[2])
    
    #Z = h_theta2(params, np.array([X0, X1, X2]))
    #print(Z.shape)
    #fig = plt.figure()  # 2d
    #ax = Axes3D(fig)    # 3d
    #ax.set_label('X')
    #ax.set_label("y")
    #ax.set_label("f(x, y")

    #ax.plot_wireframe(X1, X2, Z)

def linear_regression(x, y):
    theta0 = 0
    theta1 = 0
    alpha = 1
    m = len(x)
    jt = j_theta(theta0, theta1, x, y)
    # GD
    for i in range(100):
        for xi, yi in zip(x, y):
            theta0 = theta0 + alpha * (yi - h_theta(theta0, theta1, xi)) / m
            theta1 = theta1 + alpha * (yi - h_theta(theta0, theta1, xi)) * xi / m
            print(theta0)
            print(theta1)
        if i % 10 == 0:
            plt.plot(x, h_theta(theta0, theta1, x), label=str(i+1))

    plt.plot(x, y, "s")
    print(h_theta(theta0, theta1, x))
    #plt.plot(xi, h_theta(theta0, theta1, xi))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()

def j_theta(theta0, theta1, x, y):
    sigma = 0
    for xi, yi in zip(x, y):
        sigma += (yi - h_theta(theta0, theta1, xi))**2     

    return sigma / (2 * len(x))

def j_theta2(params, xvar, y):
    sigma = np.sum((y - h_theta2(params, xvar))**2)
    
    return sigma / (2*y.size)

def h_theta(theta0, theta1, xi):
    return theta0 + theta1 * xi

def h_theta2(params, xvar):
    return np.dot(xvar.T, params)

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
    xvar = np.array([x0, x1])
    
    params2 = np.array([theta0, theta1, theta2])
    xvar2 = np.array([x0, x1, x2])

    #linear_regression(x, y)
    multivariate_linear_regression(params, xvar, y)
    #multivariate_linear_regression(params2, xvar2, y)
    

if __name__ == "__main__":
    main()