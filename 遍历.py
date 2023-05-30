import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

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

best_score = -np.inf
best_params = {}

# 循环遍历超参数组合
for C in np.arange(1,100,0.1):
    for epsilon in np.arange(0.01,10,0.01):
        # 创建SVR模型
        regressor = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma='scale')

        # 对拟合结果进行验证
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(regressor, X_train_norm, y_train, cv=cv)
        avg_score = np.mean(scores)

        # 比较当前超参数组合的性能与最佳性能
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'C': C, 'epsilon': epsilon}

# 使用最佳超参数组合进行模型训练
best_regressor = SVR(kernel='poly', **best_params)
best_regressor.fit(X_train_norm, y_train)

# 在测试集上进行预测
y_pred = best_regressor.predict(X_test_norm)

accuracy = sum(abs((y_test - y_pred) / y_test) <= 0.2) / len(y_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
corr_coef = np.corrcoef(y_test, y_pred)

# 绘制预测结果和测试集真实结果的折线图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("True vs Predicted")
plt.legend()
plt.show()

# 打印结果
print('SVR')
print("准确率:", accuracy)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print("相关系数:", corr_coef[0][1])
print('平均交叉验证:', best_score)
print("最佳超参数组合:", best_params)
