# encoding:utf-8

import numpy as np

# def read_data():
#     arr = []
#     with open('./data/ex1data2.txt', 'r') as input:
#         for line in input.readlines():
#             tmp1 = line.strip('\n')                 # del the last char -- '\n'
#             tmp2 = tmp1.split(',')                  # using the dot as the division
#             arr.append(tmp2)                        # add to the arr(kind--str)
#     # print (arr)
#     arr = np.array(arr, np.float32)
#     # print (arr)
#     arr_mean = np.average(arr, axis=0)
#     arr_stand_diviation = np.std(arr, axis=0)
#     # only to x
#     for i in range(arr.shape[1] - 1):
#         arr[:, i] = (arr[:, i] - arr_mean[i]) / arr_stand_diviation[i]
#     return arr_mean, arr_stand_diviation, arr[:, 0:(arr.shape[1] - 1)], arr[:, arr.shape[1] - 1]

def read_data():
    arr = []
    with open('./data/ex1data1.txt', 'r') as input:
        for line in input.readlines():
            tmp1 = line.strip('\n')                 # del the last char -- '\n'
            tmp2 = tmp1.split(',')                  # using the dot as the division
            arr.append(tmp2)                        # add to the arr(kind--str)
    # print (arr)
    arr = np.array(arr, np.float32)
    # print (arr)
    # arr_mean = np.average(arr, axis=0)
    # arr_stand_diviation = np.std(arr, axis=0)
    # for i in range(3):
    #     arr[:, i] = (arr[:, i] - arr_mean[i]) / arr_stand_diviation[i]
    return arr[:, 0], arr[:, 1]


