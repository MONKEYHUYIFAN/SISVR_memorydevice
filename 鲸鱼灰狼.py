from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import datasets, metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from mealpy.swarm_based.GWO import GWO_WOA


# 读入Excel文件到数据框
df = pd.read_excel('data.xlsx')

# 将数据框转换为NumPy数组
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 训练集和测试集
X_train = X[:125]
y_train = y[:125]
X_test = X[125:]
y_test = y[125:]

# 对训练集和测试集的输入特征进行归一化处理
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

def fitness_function(solution):
    c = solution[0]  # 第一个变量为C参数
    e = solution[1]  # 第二个变量为epsilon参数
    g = solution[2]  # 第三个变量为gamma参数
    regressor = SVR(kernel='rbf', C=c, epsilon=e,gamma=g)
    # 使用交叉验证计算模型的均方误差
    cv_scores = cross_val_score(regressor, X_train_norm, y_train, cv=5, scoring='neg_mean_squared_error')
    fitness = -np.mean(cv_scores)  # 取平均均方误差的负值作为适应度值
    return fitness

problem = {
   "fit_func": fitness_function,
   "lb": [0.01, 0.001, 0.01],
   "ub": [50, 1, 100],
   "minmax": "min",
   #"obj_weights": [1, 1, 1, 1, 1]  # Adjust the number of weights to match the number of objectives
}



epoch = 120
pop_size = 100
model = GWO_WOA(epoch, pop_size)
best_position, best_fitness = model.solve(problem)
print(f"Solution: {best_position}, Fitness: {best_fitness}")

regressor = SVR(kernel='rbf', C=best_position[0], epsilon=best_position[1])
regressor.fit(X_train_norm, y_train)

#预测测试集的目标值
y_pred = regressor.predict(X_test_norm)
#相关系数
corr_coef = np.corrcoef(y_test, y_pred)

#准确率
accuracy = sum(abs((y_test - y_pred) / y_test) <= 0.2) / len(y_test)

#计算测试集的均方误差和平均绝对误差
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 交叉验证评估模型性能
scores = cross_val_score(regressor, X_train, y_train, cv=5)

#输出到屏幕
print('GWOSVR')
print('最优的参数C:', best_position[0])
print('最优的参数epsilon:', best_position[1])
print("准确率:", accuracy)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print("相关系数:", corr_coef[0][1])
print('平均交叉验证:', np.mean(scores))



# 绘制预测结果和测试集真实结果的折线图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("True vs Predicted")
plt.legend()
plt.show()

# Instantiate the Support Vector Classifier (SVC)
# C in R+
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# C = [1, 10, 100, 1000]

# for c in C:
#     for kernel in kernels:
#         svc = SVC(C=c, random_state=1, kernel=kernel)
#
#         # Fit the model
#         svc.fit(X_train_std, y_train)
#
#         # Make the predictions
#         y_predict = svc.predict(X_test_std)
#
#         # Measure the performance
#         print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))