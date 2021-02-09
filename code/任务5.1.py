import pandas as pd

file = '../data/380平台(已处理).csv'

# 读取数据
data = pd.read_csv(file,encoding='gbk')
data = data[data['行业']=='母婴']

# 透视数据
df_p = data.pivot_table(index=['项目简称'],  # 透视的行，分组依据
			columns=['日期'],# 透视的列，分组依据
        	values='经营利润1',  # 值
        	aggfunc='sum'  # 聚合函数
           )
data = pd.DataFrame(df_p)

print(data)
data.to_csv('../data/任务5.1(380平台母婴行业各品牌销售分布情况).csv',encoding='gbk')