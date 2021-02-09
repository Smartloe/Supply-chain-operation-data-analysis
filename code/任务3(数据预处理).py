import pandas as pd
import matplotlib.pyplot as plt

file = '../data/380平台－数据源.csv'

# 读取数据
data = pd.read_csv(file,encoding='gbk')
print(data.describe())
print(data.info())

# 从data.info()得知，关联收支、经营费用1、市场费用(原）、财务费用（原）、运作年限均有少量缺失值,均填充为NOT PROVIDED
data.fillna(0,inplace=True)

# 删除无用列
data.drop(['导入模式'],axis=1, inplace=True)#删除多余列

#异常值，由于数据基数大,对异常值不处理
def yi_chang(data):
	data.boxplot()
	plt.show()
yi_chang(data[['经营利润1','财务费用（原）','管理费用1','业务利润1']])

# 重新存储
data.to_csv('../data/380平台(已处理).csv',index=False,encoding='gbk')