import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

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

# 定义灰狼算法的参数
max_iter = 200  # 最大迭代次数
n_pop = 100  # 灰狼群数量
alpha = 0.9  # 用于计算每只灰狼位置的参数
beta = 0.9  # 用于计算每只灰狼位置的参数
delta = 0.5  # 用于计算每只灰狼位置的参数
lower_bound = np.array([0.1, 0.01])  # 惩罚参数C和核函数参数g的下界
upper_bound = np.array([100, 10]) # 惩罚参数C和核函数参数g的上界


# 定义灰狼的初始位置
wolves = np.random.uniform(size=(n_pop, 2)) * (upper_bound - lower_bound) + lower_bound

# 对灰狼进行迭代优化
for i in range(max_iter):
    # 计算每只灰狼的适应度
    fitness = np.zeros(n_pop)
    for j in range(n_pop):
        # 将灰狼的位置映射到惩罚参数C和核函数参数g的范围内
        cc = wolves[j, 0]
        ep = wolves[j, 1]
        # 训练支持向量回归模型
        regressor = SVR(kernel='rbf', C=cc, gamma=ep)
        # 使用交叉验证计算模型的均方误差
        cv_scores = cross_val_score(regressor, X_train_norm, y_train, cv=5, scoring='neg_mean_squared_error')
        fitness[j] =-np.mean(cv_scores)
    # 找到当前群体中适应度最好的灰狼
    best_index =-np.argmin(fitness)
    best_wolf = wolves[best_index]    
    # 更新每只灰狼的位置
    for j in range(n_pop):
        # 计算灰狼与最好灰狼的距离
        delta_position = np.abs(best_wolf - wolves[j])   
        # 计算每只灰狼的位置
        A1 = alpha * (2 * np.random.rand(2) - 1)
        C1 = 2 * np.random.rand(2)
        D_alpha = np.abs(C1 * best_wolf - wolves[j])
        X1 = best_wolf - A1 * D_alpha
        # 找到与当前灰狼距离最近的灰狼
        all_distances = np.sqrt(np.sum((wolves - wolves[j])**2, axis=1))
        nearest_index = np.argmin(all_distances)
        nearest_wolf = wolves[nearest_index]
        
        # 计算每只灰狼的位置，引入beta参数
        A2 = beta * (2 * np.random.rand(2) - 1)
        C2 = 2 * np.random.rand(2)
        D_beta = np.abs(C2 * nearest_wolf - wolves[j])
        X2 = nearest_wolf - A2 * D_beta
        
        # 找到当前群体中适应度最差的灰狼
        worst_index = np.argmax(fitness)
        worst_wolf = wolves[worst_index]
        
        # 计算每只灰狼的位置，引入delta参数
        A3 = delta * (2 * np.random.rand(2) - 1)
        C3 = 2 * np.random.rand(2)
        D_delta = np.abs(C3 * worst_wolf - wolves[j])
        X3 = worst_wolf - A3 * D_delta        
        # 计算每只灰狼的位置
        wolves[j] = (X1 + X2 + X3)/3
        new_position = (X1 + X2 + X3) / 3
        new_position = np.maximum(new_position, lower_bound)
        new_position = np.minimum(new_position, upper_bound)
        wolves[j] = new_position

#训练最终的支持向量回归模型
best_C = best_wolf[0]
best_epsilon = best_wolf[1]
regressor = SVR(kernel='rbf', C=best_C, gamma=best_epsilon)
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
print('最优的参数C:', best_C)
print('最优的参数epsilon:', best_epsilon)
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
