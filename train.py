# encoding:utf-8
import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# draw three dimensional pic
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

epsilon = 0.001
Learning_rate = 0.001

def main():
    x, y = utils.read_data()    # get the coloum 1st-x(population), 2nd -y(profits)
    # to see the relationship of the x and y
    x = x.reshape(1, len(x))
    y = y.reshape(1, len(y))
    fig = plt.figure()
    plt.scatter(x, y, color='blue', linewidth=1.0, linestyle='-', label='profits', alpha=0.5)
    plt.title('population & profits')

    w1, bias = linear_model(x, y)
    print (w1, bias)
    # draw the line
    xx = np.arange(4, 25, 1).reshape(21, 1)
    yy = xx * w1 + bias
    plt.plot(xx, yy, color='red')
    # result = linear_model(x, y)
    # print (result['intercept'])
    # print (result['coef'])

    # start make data
    # # ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)
    # w = np.arange(0, 1, 0.1).reshape(10, 1)
    # b = np.arange(-2, 0, 0.2).reshape(10, 1)
    # w, b = np.meshgrid(w, b)
    # loss = np.sum(np.square(np.dot(w, x) + b - y)) / (2 * x.shape[1])
    #
    # # Plot the surface.
    # surf = ax.plot_surface(w, b, loss, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    pass


def linear_model(x, y):
    w1 = np.zeros(shape=(1, 1), dtype=np.float32)
    bias = 0.5
    loss1 = 0
    # loss0 = 0
    iter = 0
    while True:
        # define the model
        iter += 1
        y1 = np.dot(w1, x) + bias
        dy1 = y1 - y
        dw1 = np.dot(x, dy1.T) / x.shape[1]
        db = np.sum(dy1) / x.shape[1]
        w1 = w1 - Learning_rate * dw1
        bias = bias - Learning_rate * db
        # define the loss fun
        loss1 = np.sum(np.square(y1 - y)) / (2 * x.shape[1])
        if (loss1) < 5:
            break
        # else:
        #     loss0 = loss1
        if iter % 500 == 0:
            print ('loss is %f' % (loss1))
    return w1, bias

# def linear_model(x, y):
#     regr = LinearRegression()
#     regr.fit(x, y)
#     predict = {}
#     predict['intercept'] = regr.intercept_
#     predict['coef'] = regr.coef_
#     return predict

if __name__ == '__main__':
    main()