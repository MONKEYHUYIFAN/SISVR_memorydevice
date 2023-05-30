import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
# =============================================================================
# 输入数据并处理。第一列为气体种类，1为氮气，2为氧气，3为氩气，第二列是ALD温度，第三列是
# RTA温度，第四列是施加的电压。
# =============================================================================
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


# 进行模型训练，支持向量机回归模型linear，sigmoid,rbf,poly

regressor = SVR(kernel='rbf', C=10, epsilon=0.01)

# 对拟合结果进行验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(regressor, X_train_norm, y_train, cv=cv)

# 拟合模型并预测
regressor.fit(X_train_norm, y_train)
y_pred = regressor.predict(X_test_norm)

accuracy = sum(abs((y_test - y_pred) / y_test) <= 0.2) / len(y_test)
# 计算均方误差和平均绝对误差
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
corr_coef = np.corrcoef(y_test, y_pred)

# 已知的三个输入量
input_1 = 1
# 归一化第二个输入量
input_2_range = np.arange(100, 350, 1)
min_2 = np.min(X[:, 1])
max_2 = np.max(X[:, 1])
scale_2 = (input_2_range - min_2) / (max_2 - min_2)
# 归一化第三个输入量
input_3 = 450
min_3 = np.min(X[:, 2])
max_3 = np.max(X[:, 2])
scale_3 = (input_3 - min_3) / (max_3 - min_3)
# 要预测的第四个输入量
input_4 = 20
min_4 = np.min(X[:, 3])
max_4 = np.max(X[:, 3])
scale_4 = (input_4 - min_4) / (max_4 - min_4)

# 预测结果列表
predictions = []

# 预测
for input_2 in input_2_range:
    scale_2 = (input_2 - min_2) / (max_2 - min_2)
    prediction = regressor.predict([[input_1, scale_2, scale_3, scale_4]])
    predictions.append(prediction[0])

# 找出最大值及其对应的输入量
max_prediction = max(predictions)
max_index = predictions.index(max_prediction)
max_input_2 = input_2_range[max_index]

#绘制特征与输出变量之间的散点图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(X_train_norm[:, 0], y_train, s=5)
axs[0].set_xlabel('Gas Type')
axs[0].set_ylabel('RTA Time (min)')
axs[1].scatter(X_train_norm[:, 1], y_train, s=5)
axs[1].set_xlabel('ALD Temperature (℃)')
axs[1].set_ylabel('RTA Time (min)')
axs[2].scatter(X_train_norm[:, 2], y_train, s=5)
axs[2].set_xlabel('RTA Temperature (℃)')
axs[2].set_ylabel('RTA Time (min)')
plt.show()

# 绘制预测结果和测试集真实结果的折线图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("True vs Predicted")
plt.legend()
plt.show()
print('SVR')
print("准确率:", accuracy)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print("相关系数:", corr_coef[0][1])
print('平均交叉验证:', np.mean(scores))
