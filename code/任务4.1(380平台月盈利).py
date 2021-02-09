import pandas as pd
#导入pyecharts模块及随机虚构数据模块
from pyecharts.faker import Faker
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType

file = '../data/380平台(已处理).csv'

# 读取数据
data = pd.read_csv(file,encoding='gbk')

# 查看月份
print('共计{}个月'.format(len(data['日期'].unique())))
data['日期'] = data["日期"].map(lambda x:x[:7])
date = data['日期'].unique()

# 380平台月盈利
x,y,z= ['月份'],['盈利'],[]
for m in date:
	df = data[data['日期']==m]
	x.append(m)
	y.append(df['经营利润1'].sum())

for i in y:
	z.append([i])

dict_data = dict(zip(x,z))
data = pd.DataFrame(dict_data)
data = data.T
data.to_csv('../data/任务4.1(380平台月盈利).csv',header=False,encoding='gbk')

data = pd.read_csv('../data/任务4.1(380平台月盈利).csv',encoding='gbk')
print(data['盈利'].sum())

def bar(x,y):
	bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
	bar.add_xaxis(x)
	bar.add_yaxis("盈利",y)
	bar.set_global_opts(title_opts=opts.TitleOpts(title="380平台月盈利"))
	bar.set_series_opts(
		label_opts=opts.LabelOpts(is_show=False))
	bar.render('../graph/4.1平台月盈利.html')

bar(data['月份'].to_list(),data['盈利'].to_list())