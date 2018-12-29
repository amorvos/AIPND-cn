# encoding = utf8

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

movie_data = pd.read_csv("./tmdb-movies.csv")

movie_data = movie_data.dropna()

simple_data = movie_data[['id', 'popularity', 'budget', 'runtime', 'vote_average']]

# print(movie_data.loc[list(range(20)) + list([47, 48])])
# print('===========')
# print(movie_data.loc[50:60, ['popularity']])
# print('===========')
# print(movie_data[movie_data['popularity'] > 5])
# print('===========')
# print(movie_data[(movie_data['popularity'] > 5) & (movie_data['release_year'] > 1996)])
# print(movie_data.groupby('release_year').agg({"revenue": np.average}))
# array = np.array(movie_data)
# print(movie_data.groupby('director').agg({"popularity": np.average}).sort_values('popularity', ascending=False))


pop_data = movie_data[['original_title', 'popularity']].sort_values('popularity', ascending=False).head(20)

bin_edges = np.arange(0, pop_data['popularity'].max() + 1 / 4, 1 / 4)
plt.hist(data=pop_data, x='original_title', bins = bin_edges)

# y_means = pd.to_datetime(df['indate']) - pd.to_datetime(df['dob'])

y_means = movie_data['revenue'] - movie_data['budget']

#评分和票房高的电影的导演排行
top_votes =movie_data[movie_credit['vote_average'] >=8].sort_values(by = 'vote_average', ascending = False)
top_votes[top_votes.isnull()]
#删除空值
top_votes = top_votes[[ 'director', 'vote_average' , 'revenue']].dropna()
top_revenue = top_votes.sort_values(by = 'revenue', ascending = False)
top_revenue = top_revenue[[ 'revenue','director']]
top_revenue1 = top_revenue.groupby('director')['revenue'].mean().sort_values(ascending = True)

# 图表可视化
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
ax1 =top_revenue1.tail(10).plot.barh(width=0.8,color = '#228B22')
plt.xticks(fontsize=13 ,rotation = 0)
plt.yticks(fontsize=13)
plt.xlabel('票房',fontsize = 13)
plt.ylabel('导演', fontsize = 13)
plt.grid(True)
plt.title('高票房高评分导演排行榜', fontsize = 15)