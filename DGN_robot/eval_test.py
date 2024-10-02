import numpy as np

# 假设我们有一个二维矩阵
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 获取矩阵的行数和列数
rows, cols = matrix.shape

# 创建一个新的行向量，长度与矩阵的列数匹配
new_row = np.zeros((1, cols))

# 创建一个新的列向量，长度与矩阵的行数匹配
new_column = np.zeros((rows + 1, 1))

# 使用np.append()函数添加新的行
matrix = np.append(matrix, new_row, axis=0)

# 使用np.append()函数添加新的列
matrix = np.append(matrix, new_column, axis=1)

print("matrix:",matrix)