import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
# 读取数据
data = pd.read_excel('data.xlsx')

# 提取输入和输出变量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 对输入变量进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train = X[:125]
y_train = y[:125]
X_test = X[125:]
y_test = y[125:]

# 对训练集进行标准化
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 对测试集进行标准化
X_test = scaler.transform(X_test)

# 预测
y_pred = model.predict(X_test)


# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = sum(abs((y_test - y_pred) / y_test) <= 0.2) / len(y_test)
scores = cross_val_score(model, X_train, y_train, cv=5)

# 输出评估指标
print('Linear')
print("准确率:", accuracy)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print('相关系数：', r2)
print('平均交叉验证:', np.mean(scores))


# 可视化实际值和预测值
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs. Predicted')
plt.show()