import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
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

# 多层感知模型
regressor = MLPRegressor(hidden_layer_sizes=(100,50,20), activation='relu', solver='adam', alpha=0.0001, max_iter=1000, random_state=42)

# 训练模型并预测
train_loss = []
test_loss = []
for i in range(regressor.max_iter):
    regressor.partial_fit(X_train_norm, y_train)
    train_loss.append(mean_squared_error(y_train, regressor.predict(X_train_norm)))
    test_loss.append(mean_squared_error(y_test, regressor.predict(X_test_norm)))
    if i % 50 == 0:
        print("Epoch:", i+1, "Training loss:", train_loss[-1], "Test loss:", test_loss[-1])

y_pred = regressor.predict(X_test_norm)
accuracy = sum(abs((y_test - y_pred) / y_test) <= 0.2) / len(y_test)

# 计算均方误差和平均绝对误差
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
corr_coef = np.corrcoef(y_test, y_pred)
# 对拟合结果进行验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(regressor, X_train_norm, y_train, cv=cv)

#输出
print('GWOSVR')
print("准确率:", accuracy)
print('均方误差：{:.2f}'.format(mse))
print('平均绝对误差：{:.2f}'.format(mae))
print("相关系数:", corr_coef[0][1])
print('平均交叉验证:', np.mean(scores))
# 可视化损失
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# 绘制预测结果和测试集真实结果的折线图
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("True vs Predicted")
plt.legend()
plt.show()
