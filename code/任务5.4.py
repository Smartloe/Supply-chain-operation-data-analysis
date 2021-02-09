import pandas as pd
from pyecharts import options as opts
from pyecharts.faker import Faker
from pyecharts.charts import Line
from pyecharts.charts import Pie
from pyecharts.charts import WordCloud
from pyecharts.globals import ThemeType

file = '../data/380平台(已处理).csv'
# 读取数据
data = pd.read_csv(file,encoding='gbk')
data = data[data['行业']=='母婴']
# data["日期"]=pd.to_datetime(data["日期"]).dt.month
data["月份"]=data["日期"].map(lambda x:x[:7])
# 透视数据
df_p = data.pivot_table(index=['月份'],  # 透视的行，分组依据
			columns=['行业'],# 透视的列，分组依据
        	values=['经营利润1'],  # 值
        	aggfunc='sum'  # 聚合函数
           )
#透视表转普通表
df_p.columns = df_p.columns.droplevel(0)
data = df_p.reset_index().rename_axis(None, axis=1)

data = pd.DataFrame(data)
# print(data)
# 折线图
def zhe_xian():
	line = Line(init_opts=opts.InitOpts(theme=ThemeType.PURPLE_PASSION))
	line.add_xaxis(data["月份"].to_list())
	line.add_yaxis("盈利",data["母婴"].to_list(),is_smooth=True,areastyle_opts=opts.AreaStyleOpts(
		opacity=0.2,
		color='#9bf6ff'))
	line.set_global_opts(title_opts=opts.TitleOpts(title="母婴:月盈利"))
	line.set_series_opts(
		label_opts=opts.LabelOpts(is_show=False),#不显示数字
		markpoint_opts=opts.MarkPointOpts(
			data=[
				opts.MarkPointItem(type_="max",name="最大值"),
				opts.MarkPointItem(type_="min",name="最小值"),
			]
	),
		markline_opts=opts.MarkLineOpts(
			data=[
				opts.MarkLineItem(type_="average",name="平均值"),
			]
	),
		)
	line.render('../graph/5.4(母婴月盈利).html')
zhe_xian()

# 读取数据
data = pd.read_csv(file,encoding='gbk')
data = data[data['行业']=='母婴']
# 透视数据
df_p = data.pivot_table(index=['项目简称'],  # 透视的行，分组依据
			columns=['行业'],# 透视的列，分组依据
        	values=['经营利润1'],  # 值
        	aggfunc='sum'  # 聚合函数
           )

#透视表转普通表
df_p.columns = df_p.columns.droplevel(0)
data = df_p.reset_index().rename_axis(None, axis=1)

data = pd.DataFrame(data)
# print(data)

def pie():
	pie = Pie(init_opts=opts.InitOpts(theme=ThemeType.PURPLE_PASSION,
	width="1280px",
	height="1080px"))
	pie.add("",[list(z) for z in zip(data["项目简称"].to_list(),data["母婴"].to_list())],
		radius=["40%","75"])
	pie.set_global_opts(title_opts=opts.TitleOpts(title="各类母婴产品",pos_bottom='50%'))
	pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{c}"))
	pie.render('../graph/5.4(各类母婴产品盈利).html')
pie()

# 读取数据
data = pd.read_csv(file,encoding='gbk')
data = data[data['行业']=='母婴']
# 透视数据
df_p = data.pivot_table(index=['省区平台'],  # 透视的行，分组依据
			columns=['行业'],# 透视的列，分组依据
        	values=['经营利润1'],  # 值
        	aggfunc='sum'  # 聚合函数
           )

#透视表转普通表
df_p.columns = df_p.columns.droplevel(0)
data = df_p.reset_index().rename_axis(None, axis=1)

k = []
for _ in data['母婴']:
	k.append(int(_))
# print(k)

def ci_yun():
	wordcloud = WordCloud()
	new = list(zip(data['省区平台'], k))
	words = new
	wordcloud.add("",words,word_size_range=[20,100],shape="diamond")
	wordcloud.set_global_opts(title_opts=opts.TitleOpts(title="热销平台"))
	wordcloud.render("../graph/5.4(热销平台).html")
ci_yun()

# 读取数据
data = pd.read_csv(file,encoding='gbk')
data = data[data['行业']=='母婴']
# 透视数据
df_p = data.pivot_table(index=['项目简称'],  # 透视的行，分组依据
			columns=['是否停牌'],# 透视的列，分组依据
        	values=['行业'],  # 值
        	aggfunc='count'  # 计数函数
           )
#透视表转普通表
df_p.columns = df_p.columns.droplevel(0)
data = df_p.reset_index().rename_axis(None, axis=1)
data.drop(['0'],axis=1, inplace=True)#删除多余列
data = pd.DataFrame(data)
data['停牌'].fillna(0,inplace=True)
data['正常'].fillna(0,inplace=True)
data['差值'] = data['停牌']-data['正常']
p,q,a,z= [],'推荐','正常','不推荐'
for x in data['差值']:
	if x>0:
		p.append(z)
	elif x==0:
		p.append(a)
	else:
		p.append(q)
data['是否推荐'] = p
data.drop(['差值'],axis=1, inplace=True)#删除多余列
data.to_csv('../data/任务5.4(需加大推广的品牌).csv',index=False)

		