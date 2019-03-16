# coding: utf-8

# # 电影数据分析案例

# # 1.提出问题

# 电影公司制作一部新电影推向市场时，要想获得成功，通常要了解电影市场趋势，观众喜好的电影类型，电影的发行情况，改编电影和原创电影的收益情况，以及观众喜欢什么样的内容。
#
# 本案例来源于kaggle上的[TMDB 5000 Movie Dataset数据集](https://www.kaggle.com/tmdb/tmdb-movie-metadata)，为了探讨电影数据可视化，为电影的制作提供数据支持，主要研究以下几个问题：
# * **电影类型如何随着时间的推移发生变化的？**
# * **电影类型与利润的关系？**
# * **Universal和Paramount两家影视公司的对比情况如何？**
# * **改编电影和原创电影的发行对比情况如何？**
# * **电影时长与电影票房及评分的关系？**
# * **分析电影关键字**

# # 2.理解数据

# ## 2.1 采集数据

# [从kaggle上的TMDB 5000 Movie Dataset下载数据集](https://www.kaggle.com/tmdb/tmdb-movie-metadata)

# ## 2.2 导入数据

# In[38]:

import json
import warnings

import pandas as pd

warnings.filterwarnings('ignore')  # 忽略python运行过程中的警告错误

# 数据可视化
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # 导入词云包

# In[39]:

# 导入电影数据
credits_file = 'D:\\Python\\notebook\\tmdb_5000_credits.csv'
movies_file = 'D:\\Python\\notebook\\tmdb_5000_movies.csv'
credits = pd.read_csv(credits_file)
movies = pd.read_csv(movies_file)

# ## 2.3 查看数据集信息

# In[40]:

credits.head()

# In[41]:

movies.head()

# # 3.数据清洗

# ### （1）合并数据集

# In[42]:

# 合并数据集
fulldf = pd.concat([credits, movies], axis=1)

# 查看合并后的数据集信息
fulldf.info()

# ### （2）选取子集

# In[43]:

moviesdf = fulldf[['original_title', 'crew', 'release_date', 'genres', 'keywords', 'production_companies',
                   'production_countries', 'revenue', 'budget', 'runtime', 'vote_average']]
moviesdf.info()

# In[44]:

# 增加profit列
moviesdf['profit'] = moviesdf['revenue'] - moviesdf['budget']
moviesdf.head()

# ### （3）缺失值处理

# **通过上面的数据集信息可以知道：整个数据集缺失的数据比较少**
#
# **其中release_date（首次上映日期）缺失1个数据，runtime（电影时长）缺失2个数据，可以通过网上查询补齐这个数据**

# In[45]:

# 找出release_date（首次上映日期）缺失的数据
release_date_null = moviesdf['release_date'].isnull()
moviesdf.loc[release_date_null, :]

# In[46]:

# 填充指定日期，从网上查到这部电影上映日期为2014年6月1日
moviesdf['release_date'] = movies['release_date'].fillna('2014-06-01')
# 修改日期格式
moviesdf['release_date'] = pd.to_datetime(moviesdf['release_date'], format='%Y-%m-%d')
moviesdf.info()

# In[47]:

# 找出runtime（电影时长）缺失的数据
runtime_date_null = moviesdf['runtime'].isnull()
moviesdf.loc[runtime_date_null, :]

# In[48]:

# 填充runtime缺失值
'''
网上查询结果：
电影《Chiamatemi Francesco - Il Papa della gente》的时长为98分钟；
电影《To Be Frank, Sinatra at 100》的时长为81分钟
'''
values1 = {'runtime': 98.0}
values2 = {'runtime': 81.0}
moviesdf.fillna(value=values1, limit=1, inplace=True)
moviesdf.fillna(value=values2, limit=1, inplace=True)

moviesdf.loc[runtime_date_null, :]

# ### （4）数据格式转换

# **genres列数据处理**

# In[49]:

# genres列格式化，建立包含所有genre类型的列表
moviesdf['genres'] = moviesdf['genres'].apply(json.loads)


# 自定义函数解码json数据
def decode(column):
    z = []
    for i in column:
        z.append(i['name'])
    return ' '.join(z)


moviesdf['genres'] = moviesdf['genres'].apply(decode)
moviesdf.head(2)

# In[50]:

# 建立genres列表，提取电影的类型
genres_list = set()
for i in moviesdf['genres'].str.split(' '):
    genres_list = set().union(i, genres_list)
    genres_list = list(genres_list)
    genres_list

genres_list.remove('')

# **release_date列数据处理**

# In[51]:

# 保留日期中的年份
moviesdf['release_date'] = pd.to_datetime(moviesdf['release_date']).dt.year
columns = {'release_date': 'year'}
moviesdf.rename(columns=columns, inplace=True)
moviesdf['year'].apply(int).head()

# # 4.数据分析及可视化

# ## 问题一：电影类型如何随着时间的推移发生变化的？

# ### （1）建立包含年份与电影类型数量的关系数据框

# In[52]:

for genre in genres_list:
    moviesdf[genre] = moviesdf['genres'].str.contains(genre).apply(lambda x: 1 if x else 0)

# In[53]:

genre_year = moviesdf.loc[:, genres_list]

# In[54]:

# 把年份作为索引标签
genre_year.index = moviesdf['year']
# 将数据集按年份分组并求和，得出每个年份，各电影类型的电影总数
genresdf = genre_year.groupby('year').sum()
# 查看数据集,tail默认查看后5行的数据
genresdf.tail()

# In[55]:

# 汇总电影类型的数量
genresdfSum = genresdf.sum(axis=0).sort_values(ascending=False)
genresdfSum

# ### （2）数据可视化

# ### 绘制柱状图

# In[56]:

# 设置画板大小
fig = plt.figure(figsize=(12, 8))
# 创建画纸，这里只使用1张画纸绘图，图形直接输出在整张画纸上
ax1 = plt.subplot(111)
# 在画纸上绘图
# 电影类型的数量按降序排序
rects = genresdfSum.sort_values(ascending=True).plot(kind='barh', label='genres')
plt.title('各种电影类型的数量统计图')
plt.xlabel('电影数量（部）', fontsize=15)
plt.ylabel('电影类型', fontsize=15)
plt.show()

# ### 绘制饼状图

# In[57]:

genres_pie = genresdfSum / genresdfSum.sum()

# 设置other类，当电影类型所占比例小于%1时，全部归到other类中
others = 0.01
genres_pie_otr = genres_pie[genres_pie >= others]
genres_pie_otr['Other'] = genres_pie[genres_pie < others].sum()

# 所占比例小于或等于%2时，对应的饼状图往外长高一截
explode = (genres_pie_otr <= 0.02) / 20 + 0.05

# 设置饼状图的参数
genres_pie_otr.plot(kind='pie', label='', startangle=50, shadow=False,
                    figsize=(10, 10), autopct='%1.1f%%', explode=explode)

plt.title('各种电影类型所占的比例')

# 分析结论：
#
# 1.从上面的结果可以看出，在所有的电影类型中，Drama(戏剧)类型电影最多，占所有电影类型的18.9%，其次为Comedy(喜剧)，占所有电影类型的14.2%。
#
# 2.在所有电影类型中，电影数量排名前5的电影类型分别为：Drama(戏剧)、Comedy(喜剧)、Thriller(惊悚)、Action（动作）、Romance（冒险）。

# ### 电影类型随时间变化的趋势分析

# In[58]:

plt.figure(figsize=(12, 8))
plt.plot(genresdf, label=genresdf.columns)
plt.xticks(range(1910, 2018, 5))
plt.legend(genresdf)
plt.title('电影类型随时间的变化趋势', fontsize=15)
plt.xlabel('年份', fontsize=15)
plt.ylabel('数量（部）', fontsize=15)
plt.grid(True)
plt.show()

# 分析结论：从图中观察到，随着时间的推移，所有电影类型都呈现出增长趋势，尤其是1992年以后各个类型的电影均增长迅速，其中Drama(戏剧)和Comedy(喜剧)增长最快，目前仍是最热门的电影类型。

# ## 问题二：电影类型与利润的关系？

# In[59]:

# 把电影类型作为索引
mean_genre_profit = pd.DataFrame(index=genres_list)

# 求出每种电影类型的平均利润
newarray = []
for genre in genres_list:
    newarray.append(moviesdf.groupby(genre, as_index=True)['profit'].mean())
newarray2 = []
for i in range(len(genres_list)):
    newarray2.append(newarray[i][1])
mean_genre_profit['mean_profit'] = newarray2
mean_genre_profit.head()

# In[60]:

# 数据可视化
plt.figure(figsize=(12, 8))

# 对于mean_profit列数据按值大小进行降序排序
mean_genre_profit.sort_values(by='mean_profit', ascending=True).plot(kind='barh')

plt.title('各种电影类型的平均利润')
plt.xlabel('平均利润（美元）')
plt.ylabel('电影类型')
plt.grid(True)
plt.show()

# 分析结论：从图中观察到，拍摄Animation、Adventure、Fantasy这三类电影盈利最好，而拍摄Foreign、TV、Movie这三类电影会存在亏本的风险

#  ## 问题三：Universal Pictures和Paramount Pictures两家影视公司的电影发行对比情况如何？

# ### （1）查看 Universal Pictures和Paramount Pictures两家影视公司电影发行的数量

# In[61]:

# production_companies列数据格式化
moviesdf['production_companies'] = moviesdf['production_companies'].apply(json.loads)
# 调用自定义函数decode处理production_companies列数据
moviesdf['production_companies'] = moviesdf['production_companies'].apply(decode)
moviesdf.head(2)

# In[62]:

# 查询production_companies数据列中是否含有Universal Pictures、Paramount Pictures，有则标记为1
moviesdf['Universal Pictures'] = moviesdf['production_companies'].str.contains('Universal Pictures').apply(
    lambda x: 1 if x else 0)
moviesdf['Paramount Pictures'] = moviesdf['production_companies'].str.contains('Paramount Pictures').apply(
    lambda x: 1 if x else 0)

# In[63]:

# 统计Universal Pictures和Paramount Pictures的数据
a = moviesdf['Universal Pictures'].sum()
b = moviesdf['Paramount Pictures'].sum()
dict_company = {'Universal': a, 'Paramount': b}
company_vs = pd.Series(dict_company)
company_vs

# In[64]:

# 使用饼状图显示两家公司发行电影所占的比例
company_vs.plot(kind='pie', label='', startangle=50, shadow=False,
                autopct='%1.1f%%')
plt.title('Universal Pictures和Paramount Pictures两家公司电影发行数量对比', fontsize=13)

# ### （2）分析Universal Pictures和Paramount Pictures两家影视公司电影发行的走势

# In[65]:

# 抽取相关数据列，以release_year作为索引
companydf = moviesdf[['Universal Pictures', 'Paramount Pictures']]
companydf.index = moviesdf['year']

# 对Universal和Paramount公司的发行数量按年分组求和
companydf = companydf.groupby('year').sum()
companydf.tail()

# In[66]:

# 数据可视化
plt.figure(figsize=(12, 8))
plt.plot(companydf, label=companydf.columns)
plt.xticks(range(1910, 2018, 5))
plt.legend(companydf)
plt.title('Universal Pictures和Paramount Pictures公司的电影发行量时间走势', fontsize=15)
plt.xlabel('年份', fontsize=15)
plt.ylabel('数量（部）', fontsize=15)
plt.grid(True)
plt.show()

# 分析结论：从图中观察到，随着时间的推移，Universal Pictures和Paramount Pictures公司的电影发行量呈现出增长趋势，尤其是在1995年后增长迅速，其中Universal Pictures公司比Paramount Pictures公司发行的电影数量更多。

# ## 问题四：改编电影和原创电影的对比情况如何？

# In[67]:

# keywords列数据格式化
moviesdf['keywords'] = moviesdf['keywords'].apply(json.loads)
# 调用自定义函数decode处理keywords列数据
moviesdf['keywords'] = moviesdf['keywords'].apply(decode)
moviesdf['keywords'].tail()

# In[68]:

# 提取关键字
a = 'based on novel'
moviesdf['if_original'] = moviesdf['keywords'].str.contains(a).apply(lambda x: 'no original' if x else 'original')
moviesdf['if_original'].value_counts()

# In[69]:

original_profit = moviesdf[['if_original', 'budget', 'revenue', 'profit']]
original_profit = original_profit.groupby(by='if_original').mean()
original_profit

# In[70]:

# 数据可视化
plt.figure(figsize=(12, 8))
original_profit.plot(kind='bar')
plt.title('改编电影与原创电影在预算、收入和利润的比较')
plt.xlabel('改编电影和原创电影')
plt.ylabel('金额（美元）')
plt.show()

# 分析结论：从图上可以看出，改编电影的预算略高于原创电影，但改编电影的票房收入和利润远远高于原创电影，
# 这可能是改编电影拥有一定的影迷基础。

# ## 问题五：电影时长与电影票房及评分的关系

# In[71]:

# 电影时长与电影票房的关系
moviesdf.plot(kind='scatter', x='runtime', y='revenue', figsize=(8, 6))
plt.title('电影时长与电影票房的关系', fontsize=15)
plt.xlabel('电影时长（分钟）', fontsize=15)
plt.ylabel('电影票房（美元）', fontsize=15)
plt.grid(True)
plt.show()

# In[72]:

# 电影时长与评分的关系
moviesdf.plot(kind='scatter', x='runtime', y='vote_average', figsize=(8, 6))
plt.title('电影时长与电影平均评分的关系', fontsize=15)
plt.xlabel('电影时长（分钟）', fontsize=15)
plt.ylabel('电影平均评分', fontsize=15)
plt.grid(True)
plt.show()

# 分析结论：从图上可以看出，电影要想获得较高的票房及良好的口碑，电影的时长应保持在90~150分钟内。

# ## 问题六：分析电影关键字

# In[73]:

# 利用电影关键字制作词云图
# 建立keywords_list列表
keywords_list = []
for i in moviesdf['keywords']:
    keywords_list.append(i)
    keywords_list = list(keywords_list)
    keywords_list

# 把字符串列表连接成一个长字符串
lis = ''.join(keywords_list)
# 使用空格替换中间多余的字符串'\'s'
lis.replace('\'s', '')

# In[74]:

# 生成词云
wc = WordCloud(background_color="black",  # 背景颜色
               max_words=2000,  # 词云显示的最大词数
               max_font_size=100,  # 字体最大值
               random_state=12,  # 设置一个随机种子，用于随机着色
               )

# 根据字符串生成词云
wc.generate(lis)

plt.figure(figsize=(16, 8))
# 以下代码显示图片
plt.imshow(wc)
plt.axis("off")
plt.show()

# 分析结论：通过对电影关键字的分析，电影中经常被提及的词语是女性（woman）、独立（independent）,其次是谋杀（murder）、爱情（love）、警察（police）、暴力（violence），可见观众对女性和独立方面题材的电影最感兴趣，其次是是犯罪类和爱情类电影。
