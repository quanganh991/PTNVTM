import math
import statistics

import numpy as np


def MAE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + abs(y_predict[i] - y_actual[i])
    return sum / m_test


def MEDAE(m_test, y_actual, y_predict):
    result = []
    for i in range(0, m_test):
        result.append(abs(y_predict[i] - y_actual[i]))
    return statistics.mean(result)


def SAE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + abs(y_predict[i] - y_actual[i])
    return sum


def MAPE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + (abs(y_predict[i] - y_actual[i]) / y_actual[i])
    return sum / m_test


def MSE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + (y_predict[i] - y_actual[i]) ** 2
    return sum / m_test


def MEDSE(m_test, y_actual, y_predict):
    result = []
    for i in range(0, m_test):
        result.append((y_predict[i] - y_actual[i]) ** 2)
    return statistics.mean(result)


def SSE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + (y_predict[i] - y_actual[i]) ** 2
    return sum


def RMSE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + (y_predict[i] - y_actual[i]) ** 2
    return math.sqrt(sum / m_test)


def MSLE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + math.log2(abs(y_predict[i] + 1)/(abs(y_actual[i] + 1))) ** 2
    return sum / m_test


def RMSLE(m_test, y_actual, y_predict):
    sum = 0
    for i in range(0, m_test):
        sum = sum + math.log2(abs(y_predict[i] + 1)/(abs(y_actual[i] + 1))) ** 2
    return math.sqrt(sum / m_test)


def RRSE(m_test, y_actual, y_predict):
    y_ngang = np.mean(y_actual)
    sum1 = 0
    sum2 = 0
    for i in range(0, m_test):
        sum1 = sum1 + (y_predict[i] - y_actual[i]) ** 2
        sum2 = sum2 + (y_actual[i] - y_ngang) ** 2
    return math.sqrt(sum1 / sum2)


def RAE(m_test, y_actual, y_predict):
    y_ngang = np.mean(y_actual)
    sum1 = 0
    sum2 = 0
    for i in range(0, m_test):
        sum1 = sum1 + abs(y_predict[i] - y_actual[i])
        sum2 = sum2 + abs(y_actual[i] - y_ngang)
    return sum1/sum2


def Coefficient_Of_Determination(m_test, y_actual, y_predict):
    y_ngang = np.mean(y_actual)
    sum1 = 0
    sum2 = 0
    for i in range(0, m_test):
        sum1 = sum1 + ((y_actual[i] - y_predict[i]) ** 2)
        sum2 = sum2 + ((y_actual[i] - y_ngang) ** 2)
    return 1 - sum1/sum2


def KendallsTau(m_test, y_actual, y_predict, threshold):
    so_cap_trung_hop = 0
    so_cap_ko_trung_hop = m_test
    for i in range(0, m_test):
        if (abs(y_predict[i] - y_actual[i]) < threshold):
            so_cap_trung_hop = so_cap_trung_hop + 1
            so_cap_ko_trung_hop = so_cap_ko_trung_hop - 1
    return abs(so_cap_trung_hop - so_cap_ko_trung_hop) / (0.5 * m_test * (m_test - 1))


def pearson_r(m_test, y_actual, y_predict):
    y_predict_ngang = np.mean(y_predict)
    y_actual_ngang = np.mean(y_actual)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(0, m_test):
        sum1 = sum1 + (y_actual[i] - y_actual_ngang) * (y_predict[i] - y_predict_ngang)
        sum2 = sum2 + (y_actual[i] - y_actual_ngang) ** 2
        sum3 = sum3 + (y_predict[i] - y_predict_ngang) ** 2
    return sum1 / math.sqrt(sum2 * sum3)


def spearman_rho(m_test, y_actual, y_predict):
    y_actual_ranking = [sorted(y_actual).index(x) for x in y_actual]
    y_predict_ranking = [sorted(y_predict).index(x) for x in y_predict]
    y_actual_ranking = [ele for ele in reversed(y_actual_ranking)]
    y_predict_ranking = [ele for ele in reversed(y_predict_ranking)]
    y_actual_ranking_ngang = np.mean(y_actual_ranking)
    y_predict_ranking_ngang = np.mean(y_predict_ranking)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(0, m_test):
        sum1 = sum1 + (y_actual[i] - y_actual_ranking_ngang) * (y_predict[i] - y_predict_ranking_ngang)
        sum2 = sum2 + (y_actual[i] - y_actual_ranking_ngang) ** 2
        sum3 = sum3 + (y_predict[i] - y_predict_ranking_ngang) ** 2
    return sum1 / math.sqrt(sum2 * sum3)


y_actual = [9.2, 1.4, 4.24, 7.58, 9.31]
y_predict = [-3, 2, 3, 5, -1.54]
m_test = len(y_actual)
print(MAE(m_test, y_actual, y_predict))
print(MEDAE(m_test, y_actual, y_predict))
print(SAE(m_test, y_actual, y_predict))
print(MAPE(m_test, y_actual, y_predict))
print(MSE(m_test, y_actual, y_predict))
print(MEDSE(m_test, y_actual, y_predict))
print(SSE(m_test, y_actual, y_predict))
print(RMSE(m_test, y_actual, y_predict))
print(MSLE(m_test, y_actual, y_predict))
print(RMSLE(m_test, y_actual, y_predict))
print(RRSE(m_test, y_actual, y_predict))
print(RAE(m_test, y_actual, y_predict))
print(Coefficient_Of_Determination(m_test, y_actual, y_predict))
print(KendallsTau(m_test, y_actual, y_predict, 0.01))
print(pearson_r(m_test, y_actual, y_predict))
print(spearman_rho(m_test, y_actual, y_predict))