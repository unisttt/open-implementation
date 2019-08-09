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