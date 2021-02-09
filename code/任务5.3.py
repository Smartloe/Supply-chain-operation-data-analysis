import pandas as pd
import numpy as np
file = '../data/380平台(已处理).csv'

# 读取数据
data = pd.read_csv(file,encoding='gbk')
data = data[data['行业']=='母婴']
data["日期"]=pd.to_datetime(data["日期"]).dt.year

# 透视数据
df_p = data.pivot_table(index=['项目简称','日期'],  # 透视的行，分组依据
			columns=[],# 透视的列，分组依据
        	values=['期末存货余额','销售成本'],  # 值
        	aggfunc='sum'  # 聚合函数
           )
data = pd.DataFrame(df_p)

#计算库存周转率
data['库存周转率'] = (data['销售成本']/data['期末存货余额'])*12


# data.to_csv('xx.csv')

# 透视数据
df_p = data.pivot_table(index=['项目简称'],  # 透视的行，分组依据
			columns=['日期'],# 透视的列，分组依据
        	values=['库存周转率'],  # 值
        	aggfunc='sum'  # 聚合函数
           )

#透视表转普通表
df_p.columns = df_p.columns.droplevel(0)
df = df_p.reset_index().rename_axis(None, axis=1)

df[2014].fillna(0,inplace=True)
df[2015].fillna(0,inplace=True)

a,b,c ='热销','正常','滞销'
tag = []

m = df[2014].to_list()
n = df[2015].to_list()

for i in range(0,65):
	if m[i]<n[i]:
		tag.append(a)
	elif m[i]==n[i]:
		tag.append(b)
	else:
		tag.append(c)
		
df['标签'] = tag

df.drop([2014,2015],axis=1, inplace=True)#删除多余列

data = pd.DataFrame(df)

data.to_csv('../data/任务5.3(母婴行业各品牌库存和周转情况).csv',encoding='gbk')