import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
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

#将数据标准化
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_norm = scaler.transform(X)

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 拟合模型
rf.fit(X_train_norm, y_train)

# 预测测试集
y_pred = rf.predict(X_test_norm)

#误差参数
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
corr_coef = np.corrcoef(y_test, y_pred)
accuracy = sum(abs((y_test - y_pred) / y_test) <= 0.2) / len(y_test)

# 绘制预测结果和测试集真实结果的折线图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("True vs Predicted")
plt.legend()
plt.show()

# 交叉验证评估模型性能
scores = cross_val_score(rf, X_train, y_train, cv=5)

#输出
print('RF')
print("准确率:", accuracy)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print("相关系数:", corr_coef[0][1])
print('平均交叉验证:', np.mean(scores))