# This python file is used to generate some of the plots used in the documentation of the thesis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim.downloader
import ast
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure
from nltk.tokenize import word_tokenize
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import random


# training_set = gensim.downloader.load('word2vec-google-news-300')

# df = pd.DataFrame(columns=["address","html","hash_code","embedding_index","mean_vector"])
# df=pd.read_csv("modified_data.csv")
# df["mean_vector"]=df["mean_vector"].apply(lambda x: ast.literal_eval(x))
# df["embedding_index"]=df["embedding_index"].apply(lambda x: ast.literal_eval(x))

# def reform_data(indices):
#     data=np.array([training_set[x] for x in indices])
#     return data

# trip_advisor= reform_data(df["embedding_index"][2225])
# imdb= reform_data(df["embedding_index"][1607])
# nasa= reform_data(df["embedding_index"][2229])
# print(df["address"][2229])

# query1=word_tokenize("rocket launching space".lower())
# query2=word_tokenize("movies film cinema".lower())
# def convert_to_vecs(tokens):
#     idxs=[]
#     for word in tokens:
#         if training_set.__contains__(word):
#             idxs.append(training_set[word])
#     vecs=np.array([vec for vec in idxs])
#     return vecs

# query1_vecs = convert_to_vecs(query1)
# query2_vecs = convert_to_vecs(query2)

# PCA_query1=PCA(3).fit_transform(query1_vecs)
# PCA_query2=PCA(3).fit_transform(query2_vecs)


# PCA_trip=PCA().fit_transform(trip_advisor)
# PCA_imdb=PCA().fit_transform(imdb)
# PCA_nasa=PCA().fit_transform(nasa)

# def mean_vector(input):  
#     meanVec=np.mean(input,axis=0)
#     return meanVec

# PCA_query1_mean= mean_vector(PCA_query1)
# PCA_query2_mean= mean_vector(PCA_query2)


# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(111, projection='3d')


# scatter_trip = ax.scatter(PCA_nasa[:, 0], PCA_nasa[:, 1], PCA_nasa[:, 2], label='Nasa', alpha=0.5)
# scatter_goodreads = ax.scatter(PCA_imdb[:, 0], PCA_imdb[:, 1], PCA_imdb[:, 2], label='dissimilar document', alpha=0.5)
# scatter_query1=ax.scatter(PCA_query1[:, 0], PCA_query1[:, 1], PCA_query1[:, 2], label='Query Nasa',alpha=1, marker="X",s=200)
# scatter_query2=ax.scatter(PCA_query2[:, 0], PCA_query2[:, 1], PCA_query2[:, 2], label='Query IMDB',alpha=1, marker="s",s=200)

# scatter_query1_mean=ax.scatter(PCA_query1_mean[0], PCA_query1_mean[1], PCA_query1_mean[2], label='Query1Mean',alpha=1, marker="X",s=200)
# scatter_query2_mean=ax.scatter(PCA_query2_mean[0], PCA_query2_mean[1], PCA_query2_mean[2], label='Query2Mean',alpha=1, c='yellow', marker="s",s=200)



# # Add legend
# ax.legend()

# # Make the plot interactive
# ax.mouse_init()

# plt.show()



# #Generate synthetic data with two clusters and increased separation
# n_samples = 300
# n_features = 3  
# n_clusters = 2
# random_state = 0
# cluster_std = [1, 1.0] 
# data, labels = make_blobs(n_samples=n_samples,
#                            n_features=n_features,
#                           centers=n_clusters,
#                             cluster_std=cluster_std,
#                             random_state=random_state)

# n_samples = 5
# n_features = 3  
# n_clusters = 1
# random_state = 0
# cluster_std = [2]
# data1, labels1 = make_blobs(n_samples=n_samples, n_features=n_features,
#                           centers=n_clusters, cluster_std=cluster_std, random_state=random_state)

# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# query=data1[labels1==0]

# # Plot points for each cluster
# cluster1_points = data[labels == 0]
# cluster2_points = data[labels == 1]

# mean_vector_cluster1 = np.mean(cluster1_points, axis=0)
# mean_vector_cluster2 = np.mean(cluster2_points, axis=0)

# mean_vector_query = np.mean(query, axis=0)


# ax.scatter(cluster1_points[:, 0], cluster1_points[:, 1], cluster1_points[:, 2], c='blue', label='Cluster 1',alpha=0.1)
# ax.scatter(cluster2_points[:, 0], cluster2_points[:, 1], cluster2_points[:, 2], c='red', label='Cluster 2',alpha=0.1)

# ax.scatter(query[:, 0], query[:, 1], query[:, 2], c='green', label='Query',marker="x", s=200)

# ax.scatter(mean_vector_cluster1[0], mean_vector_cluster1[1], mean_vector_cluster1[2], label='MeanC1',marker="X", s=200)
# ax.scatter(mean_vector_cluster2[0], mean_vector_cluster2[1], mean_vector_cluster2[2], label='MeanC2',marker="X", s=200)
# ax.scatter(mean_vector_query[0], mean_vector_query[1], mean_vector_query[2], label='MeanQuery',marker="X", s=200)

# from scipy.spatial.distance import cosine

# print(cosine(mean_vector_query,mean_vector_cluster1))
# print(cosine(mean_vector_query,mean_vector_cluster2))





# # Set labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# ax.mouse_init()
# # Show the plot
# plt.show()



cluster01 = np.random.normal(0, 1, (50, 3))  # Generating 50 data points for cluster 1
cluster2 = np.random.normal(4, 1, (50, 3))  # Generating 50 data points for cluster 2
cluster1=[point for point in cluster01]
for i in range(30):
    rand_y=random.randint(0,10)*(-1)**(random.randint(1,2))
    rand_x=random.uniform(2,4)*(-1)
    rand_z=random.uniform(2,6)*(-1)
    cluster1.append(np.array([rand_x,rand_y,rand_z]))
# for i in range(5):
#     rand_y=random.uniform(0,0.5)*(-1)**(random.randint(1,2))
#     rand_x=random.uniform(0,0.5)*(-1)
#     rand_z=random.uniform(0,0.5)*(-1)
#     cluster1.append(np.array([rand_x,rand_y,rand_z]))
cluster1=np.array(cluster1)

mean_cluster1 = np.mean(cluster1, axis=0)
mean_cluster2 = np.mean(cluster2, axis=0)

query=[2,1.5,1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for cluster 1
ax.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2], c='b', label='Cluster 1', marker='o',alpha=0.2)

# Scatter plot for cluster 2
ax.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2], c='r', label='Cluster 2', marker='s',alpha=0.2)

# Mark centroids
ax.scatter(mean_cluster1[0], mean_cluster1[1], mean_cluster1[2], c='black', marker='X', s=200, label='Centroid 1')
ax.scatter(mean_cluster2[0], mean_cluster2[1], mean_cluster2[2], c='r', marker='X', s=200, label='Centroid 2')
ax.scatter(query[0], query[1], query[2], marker='X', s=200, label='Query')


# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a legend
ax.legend()
plt.savefig("latex/figures/lemmetization1.pdf")
plt.show()
cluster1=[point for point in cluster1]
for i in range(5):
    rand_y=random.uniform(0,2.5)
    rand_x=random.uniform(0,2)
    rand_z=random.uniform(0,2)
    cluster1.append(np.array([rand_x,rand_y,rand_z]))
cluster1=np.array(cluster1)

mean_cluster1 = np.mean(cluster1, axis=0)
mean_cluster2 = np.mean(cluster2, axis=0)

query=[2,1.5,1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for cluster 1
ax.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2], c='b', label='Cluster 1', marker='o',alpha=0.2)

# Scatter plot for cluster 2
ax.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2], c='r', label='Cluster 2', marker='s',alpha=0.2)

# Mark centroids
ax.scatter(mean_cluster1[0], mean_cluster1[1], mean_cluster1[2], c='black', marker='X', s=200, label='Centroid 1')
ax.scatter(mean_cluster2[0], mean_cluster2[1], mean_cluster2[2], c='r', marker='X', s=200, label='Centroid 2')
# ax.scatter(query[0], query[1], query[2], marker='X', s=200, label='Query')


# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a legend
ax.legend()

plt.show()