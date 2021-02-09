import pandas as pd

file = '../data/380平台(已处理).csv'

# 读取数据
data = pd.read_csv(file,encoding='gbk')
# date = data['日期'].unique()
# hang_ye = data['行业'].unique()
data["月份"]=data["日期"].map(lambda x:x[:7])
# print(data["月份"])
# 透视数据
df_p = data.pivot_table(index='月份',  # 透视的行，分组依据
			columns='行业',
        	values='经营利润1',  # 值
        	aggfunc='sum'  # 聚合函数
           )
data = pd.DataFrame(df_p)
data.to_csv('../data/任务4.3(380平台各行业每月盈利).csv',encoding='gbk')