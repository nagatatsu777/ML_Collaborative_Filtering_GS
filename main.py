import pandas as pd
import numpy as np

data = pd.read_csv('u.data',sep = '\t')
dataf = pd.DataFrame(data=data)
#construct user-item rating matrix
max_user_id = max(dataf['userid'])
max_movie_id = max(dataf['movieid'])
user_movie_matrix = []
for user_id in range(0,max_user_id+1):
    tem_list = [0]*(max_movie_id+1)
    user_movie_matrix.append(tem_list)
for i in range(0,len(dataf)):
    user_movie_matrix[data['userid'][i]][data['movieid'][i]] = data['ratings'][i]

#construct user-user similarity matrix
#identify gray sheep based on this similarity matrix(probably using outlier detection)
#Using the above matrix and a variable which is based on gray sheep
#identification(Ex.This attribute will further increase the similarity when users are in the
#same type(gray sheep and gray sheep or non-gray sheep and non-gray sheep and decrease
#Use the model created by knn to give an output