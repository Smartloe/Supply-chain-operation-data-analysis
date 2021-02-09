import numpy as np
import pandas as pd
from sklearn import linear_model #导入机器学习库中的线性回归方法
from matplotlib import pyplot as plt
#导入pyecharts模块及随机虚构数据模块
from pyecharts.faker import Faker
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType

file = '../data/任务4.3(380平台各行业每月盈利).csv'
data = pd.read_csv(file,encoding='gbk')
data.fillna(0,inplace=True)
df = pd.DataFrame(data)
df["日期"] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# df["日期"]=pd.to_datetime(data["日期"]).dt.month

date = np.array(df['日期']).reshape([24,1]) 
mu_ying = np.array(df['母婴']).reshape([24,1])
shi_pin = np.array(df['食品']).reshape([24,1])
jiu_yin = np.array(df['酒饮']).reshape([24,1])
ri_hua = np.array(df['日化']).reshape([24,1])
jia_dian = np.array(df['家电']).reshape([24,1])
O2O = np.array(df['O2O金融项目']).reshape([24,1])
# print(y)
forecast = []
def function(x,y):
	plt.scatter(x,y)
	# plt.show()
	model=linear_model.LinearRegression()
	model.fit(x,y)
	#检验模型效果。
	coef=model.coef_ #获取自变量系数
	model_intercept=model.intercept_#获取截距
	R2=model.score(x,y) #R的平方
	print('线性回归方程为：','\n','y=’{}*x+{}'.format(coef[0],model_intercept))
	# 利用上面的结果进行回归预测
	new_x=[[25]]
	y_pre=model.predict(new_x)
	forecast.append(y_pre[0][0])
	# print(y_pre[0][0])

function(date,shi_pin)
function(date,mu_ying)
function(date,jiu_yin)
function(date,ri_hua)
function(date,jia_dian)
function(date,O2O)
print(forecast)
dict_data = {
	'行业':['食品','母婴','酒饮','日化','家电','O2O金融项目'],	
	'预测':forecast,
}
new_data = pd.DataFrame(dict_data)
# new_data.to_csv('sss.csv',index=False,encoding='gbk')

# 可视化
bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.PURPLE_PASSION))
bar.add_xaxis(new_data['行业'].to_list())
bar.add_yaxis("盈利",new_data['预测'].to_list())
#添加图标大标题，副标题
## set_global_opts
bar.set_global_opts(title_opts=opts.TitleOpts(title="预测盈利"))

bar.render('../graph/4.3盈利预测.html')