import pandas as pd
"""
# 读取CSV文件
df = pd.read_csv('./datas/mnist_train.csv')

# 保留前42000行
df = df.head(42000)

# 保存到新文件或覆盖原文件
df.to_csv('./datas/mnist_train_new.csv', index=False)

print(f"原始数据量: 60000")
print(f"现在数据量: {len(df)}")
"""
train_data = pd.read_csv('./datas/mnist_train_new.csv')
y = train_data['label'].values
X = train_data.drop('label',axis=1).values
print(X.shape)