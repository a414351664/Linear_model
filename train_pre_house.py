# encoding:utf-8
import numpy as np
import utils
Learning_rate = 0.01

def main():
    mean, divation, x, y = utils.read_data()  # get the coloum 1st-x(population), 2nd -y(profits)
    # X = np.hstack([x, np.ones((x.shape[0], 1))])
    x = x.T
    y = y.reshape(1, len(y)) / 10000
    # y = y.reshape(1, len(y))
    # print(x, y)
    w, bias = linear_model(x, y)
    print(mean, divation, w, bias)

    pass
def linear_model(x, y):
    w = 0.01 * np.random.randn(x.shape[0], 1)
    # w = np.zeros(shape=(x.shape[0], 1), dtype=np.float32)
    bias = 0
    iter = 0
    while True:
        iter += 1
        y1 = np.dot(w.T, x) + bias
        dy1 = y1 - y
        dw1 = np.dot(x, dy1.T) / x.shape[1]
        db = np.sum(dy1) / x.shape[1]
        w = w - Learning_rate/(np.sqrt(iter + 1)) * dw1
        bias = bias - Learning_rate/(np.sqrt(iter + 1)) * db
        loss1 = np.sum(np.square(y1 - y)) / (2 * x.shape[1])
        # loss1 = dy1.T.dot(dy1) / (2 * x.shape[1])
        if (iter) > 10000:
            break
            # else:
            #     loss0 = loss1
        if iter % 10000 == 0:
            print ('loss is %f' % (loss1))
    return w, bias

if __name__ == '__main__':
    main()