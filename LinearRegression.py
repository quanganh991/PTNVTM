import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import statistics
import math
from scipy import stats


def f(i, n, x, w, b):  # truyền vào 1 mảng 2 chiều
    sum = 0
    for j in range(0, n):
        sum = sum + w[j] * x[j][i]
    sum = sum + b
    return sum


def BPL(x, y, m, n, w, b, k):
    sum2 = 0
    for i in range(0, m):
        sum1 = 0
        for j in range(0, n):
            # print("x[", j, "][", i, "] = ", x[i][j])
            sum1 = sum1 + w[j] * x[i][j]
        sum2 = sum2 + (sum1 + b - y[i]) ** 2
    ParameterInterference = LA.norm(w)  # sqrt(w1^2 + w2^2 + w3^2 + ... + wn^2)
    ParameterInterference = (ParameterInterference ** 2 + b ** 2) * k
    sum2 = sum2 + ParameterInterference
    return sum2


def half_derivative_w(j, x, y, m, n, w, b, K):
    sum2 = 0
    for i in range(0, m):
        sum1 = 0
        for k in range(0, n):
            sum1 = sum1 + w[k] * x[i][k]
        sum2 = sum2 + x[i][j] * (sum1 + b - y[i])
    sum2 = sum2 + (K * w[j])
    return sum2


def half_derivative_b(x, y, m, n, w, b, k):
    sum2 = 0
    for i in range(0, m):
        sum1 = 0
        for j in range(0, n):
            sum1 = sum1 + w[j] * x[i][j]
        sum2 = sum2 + sum1 + b - y[i]
    sum2 = sum2 + (k * b)
    return sum2


def SolveEquation(x, y, m, n, k):  # Giải hệ phương trình:
    A = [row[:] for row in x]
    for i in range(0, m):
        A[i].append(1)
    AT = np.matrix(A).transpose()
    I = np.identity(len(AT * A))

    W = inv(np.matrix((AT * A) + k * I)) * AT * y
    # W = np.concatenate(((AT * A) + k * I, AT * y), axis=1).tolist()
    # W = Factorization(W)
    # W = [[-i] for i in W]

    print("W = ", W)
    print("(AT * A) + k * I = ", (AT * A) + k * I)
    print("AT * y = ", AT * y)
    w = np.matrix(W[0: len(W) - 1]).tolist()
    b = np.matrix(W[len(W) - 1]).tolist()
    for i in range(0, n):
        w[i] = w[i][0]
    b = b[0][0]

    dL_dw = [0] * n
    L = np.matrix(BPL(x, y, m, n, w, b, k)).tolist()[0][0]
    for j in range(0, n):
        dL_dw[j] = np.matrix(half_derivative_w(j, x, y, m, n, w, b, k)).tolist()[0][0]
    dL_db = np.matrix(half_derivative_b(x, y, m, n, w, b, k)).tolist()[0][0]
    print("Tổng bình phương lỗi = ", L)
    print("Các đạo hàm theo w = ", dL_dw)
    print("Các đạo hàm theo b = ", dL_db)
    #
    return w, b


def Factorization(equations):
    """
         VD: hệ pt
         2x+9y-3z+7w+8=0
         7x-2y+6z-1w-10=0
         -8x-3y+2z+5w+4=0
         0x+2y+z+w+0=0
         Tương đương với input: [[2,9,-3,7,8],[7,-2,6,-1,-10],[-8,-3,2,5,4],[0,2,1,1,0]]
    """
    try:
        lists = []
        for eq in range(len(equations)):
            index = 0
            for i in range(len(equations)):
                if equations[i][0] != 0:
                    index = i
                    break
            lists.append([-1.0 * i / equations[index][0] for i in equations[index][1:]])
            equations.pop(index)
            for i in equations:
                for j in range(len(lists[-1])):
                    i[j + 1] += i[0] * lists[-1][j]
                i.pop(0)

        lists.reverse()

        answers = [lists[0][0]]
        for i in range(1, len(lists)):
            tmpans = lists[i][-1]
            for j in range(len(lists[i]) - 1):
                tmpans += lists[i][j] * answers[-1 - j]
            answers.append(tmpans)
        answers.reverse()

        return answers
    except:
        print("Hệ vô nghiệm")


def Train(k):
    # Chỉ thay đổi 2 giá trị của x và y
    x = [[1, 34732, -1], [2, 21343, 9.9], [3, 12532, 0.4], [4, 36543, -4], [4.5, 29234.1, 1.5], [4.1, 31435.9, 7.3],
         [-1.2, 9.43, 3.29]]  # các điểm dữ liệu
    y = np.matrix([2, 4, 3, 1, 5, 2, 9]).transpose()  # y(k) = f(x(k)), 1 <= k <= m
    # Chỉ thay đổi 2 giá trị của x và y

    m = len(x)  # số điểm dữ liệu, len(x) = len(w) = len(y)
    n = len(x[0])  # số chiều dữ liệu, len(x[0]) = len(x[1]) = ... = len(x[m-2]) = len(x[m-1])

    return SolveEquation(x, y, m, n, k)


result = Train(k=1)
print("Kết quả hồi quy: ", result)