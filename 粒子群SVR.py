import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
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
X_norm = scaler.transform(X)

def svr_model(c, epsilon):
    model = SVR(C=c, epsilon=epsilon)
    model.fit(X_train_norm, y_train)
    return model

# 定义PSO的目标函数
def pso_objective_function(params):
    c = params[0][0]
    epsilon = params[1][0]
    model = svr_model(c, epsilon)
    y_pred = cross_val_predict(model, X_norm, y, cv=5)
    mse = mean_squared_error(y, y_pred)
    return mse

# 使用粒子群优化算法寻找最优参数
search_range = [(0.1, 100.0), (0.01, 1.0)]
options = {'c1': 2, 'c2':2, 'w': 0.9}
optimizer = GlobalBestPSO(n_particles=10000, dimensions=2, options=options, bounds=search_range)
best_params = optimizer.optimize(pso_objective_function, iters=1000)
# 训练最优模型
best_c = best_params[0]
best_epsilon = best_params[1][0]
best_model = svr_model(best_c, best_epsilon)
#SVR模型构建
regressor = SVR(kernel='rbf', C=best_c, epsilon=best_epsilon)
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


#输出到屏幕
print('最优的参数C:', best_c)
print('最优的参数epsilon:', best_epsilon)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print("准确率:", accuracy)
print("相关系数:", corr_coef[0][1])

# 绘制预测结果和测试集真实结果的折线图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("True vs Predicted")
plt.legend()
plt.show()

