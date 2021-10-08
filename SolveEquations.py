# import numpy as np
# def solve(equations):
#     """
#          VD: hệ pt
#          2x+9y-3z+7w+8=0
#          7x-2y+6z-1w-10=0
#          -8x-3y+2z+5w+4=0
#          0x+2y+z+w+0=0
#          Tương đương với input: [[2,9,-3,7,8],[7,-2,6,-1,-10],[-8,-3,2,5,4],[0,2,1,1,0]]
#     """
#     try:
#         lists = []
#         for eq in range(len(equations)):
#             index = 0
#             for i in range(len(equations)):
#                 if equations[i][0] != 0:
#                     index = i
#                     break
#             lists.append([-1.0 * i / equations[index][0] for i in equations[index][1:]])
#             equations.pop(index)
#             for i in equations:
#                 for j in range(len(lists[-1])):
#                     i[j + 1] += i[0] * lists[-1][j]
#                 i.pop(0)
#
#         lists.reverse()
#
#         answers = [lists[0][0]]
#         for i in range(1, len(lists)):
#             tmpans = lists[i][-1]
#             for j in range(len(lists[i]) - 1):
#                 tmpans += lists[i][j] * answers[-1 - j]
#             answers.append(tmpans)
#         answers.reverse()
#
#         return answers
#     except:
#         print("Hệ vô nghiệm")
#
#
# eq = [[2, 9, -3, 7, 8], [7, -2, 6, -1, -10], [-8, -3, 2, 5, 4], [0,2,1,1,0]]
# print(solve(eq))
# x = [[1],[2],[3],[4]]
# x = [i for j in x for i in j]
# x = [[-i] for i in x]
# print(x)
# x = [[2,9,-3,7],[7,-2,6,-1],[-8,-3,2,5],[0,2,1,1]]
# y = [[-8],[10],[-4],[0]]
# z = np.concatenate((x,y), axis=1).tolist()
# print(solve(z))

def largest_smallest_InColumn():
    mat = [[3, 4, -1, 8],
           [1, 4, 9, 11],
           [76, 34, 21, 1],
           [2, 1, 4, 5],
           [21, 41, -4, 75]]
    result_max = []
    result_min = []
    rows = len(mat)
    cols = len(mat[0])
    for i in range(cols):
        maxm = mat[0][i]
        for j in range(rows):
            if mat[j][i] > maxm:
                maxm = mat[j][i]
        result_max.append(maxm)

    for i in range(cols):
        minm = mat[0][i]
        for j in range(rows):
            if mat[j][i] < minm:
                minm = mat[j][i]
        result_min.append(minm)
    return result_max, result_min


print(largest_smallest_InColumn())

from datetime import datetime

date_time_str = '2018-09-19'

date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d')