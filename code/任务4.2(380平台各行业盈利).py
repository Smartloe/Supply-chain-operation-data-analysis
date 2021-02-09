import pandas as pd
#导入pyecharts模块及随机虚构数据模块
from pyecharts.faker import Faker
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType
file = '../data/380平台(已处理).csv'

# 读取数据
data = pd.read_csv(file,encoding='gbk')

#查看行业
print('共涉及{}个行业'.format(len(data['行业'].unique())))
hang_ye = data['行业'].unique()
print( data['行业'].unique())

# data["日期"]=pd.to_datetime(data["日期"]).dt.year
data["日期"]=pd.to_datetime(data["日期"]).dt.quarter
# print(data["日期"])
# data["月份"]=data["日期"].map(lambda x:x[:7])
# 透视数据
df_p = data.pivot_table(index=['日期'],  # 透视的行，分组依据
			columns=['行业'],# 透视的列，分组依据
        	values='经营利润1',  # 值
        	aggfunc='sum'  # 聚合函数
           )

df_p.fillna(0,inplace=True)
data = pd.DataFrame(df_p)
print('O2O金融项目:',data['O2O金融项目'].sum())
print('家电:',data['家电'].sum(),'元')
print('日化:',data['日化'].sum(),'元')
print('母婴:',data['母婴'].sum(),'元')
print('酒饮:',data['酒饮'].sum(),'元')
print('食品:',data['食品'].sum(),'元')
data.to_csv('../data/任务4.2(380平台各行业盈利).csv',header=True,encoding='gbk')

data = pd.read_csv('../data/任务4.2(380平台各行业盈利).csv',encoding='gbk')
def bar():
	bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.PURPLE_PASSION ))
	bar.add_xaxis(data["日期"].to_list())
	bar.add_yaxis("O2O金融项目",data['O2O金融项目'].to_list())
	bar.add_yaxis("家电",data['家电'].to_list())
	bar.add_yaxis("日化",data['日化'].to_list())
	bar.add_yaxis("母婴",data['母婴'].to_list())
	bar.add_yaxis("酒饮",data['酒饮'].to_list())	
	bar.add_yaxis("食品",data['食品'].to_list())
	#标记点和标记线
	bar.set_series_opts(
		label_opts=opts.LabelOpts(is_show=False),#不显示数字
			markpoint_opts=opts.MarkPointOpts(
		data=[
			opts.MarkPointItem(type_="max",name="最大值"),
			# opts.MarkPointItem(type_="min",name="最小值"),
		]
	),
		)
	bar.render('../graph/4.2(平台各行业盈利).html')
bar()