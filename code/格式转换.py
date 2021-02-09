import pandas as pd

#路径
file = '../data/380平台－数据源.xlsx'

#读取数据
data = pd.read_excel(file)
df = pd.DataFrame(data)

#转为csv格式
df.to_csv('../data/380平台－数据源.csv',index=False,encoding='gbk')