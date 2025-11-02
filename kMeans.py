import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
data=pd.read_csv(r'C:\Users\jeyas\Desktop\lab\ml\code\iris.csv')
x=data.iloc[:,:2]
x=pd.get_dummies(x)
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,8))
plt.plot(range(1,11),wcss,marker='o')
plt.show() 
k_means=KMeans(n_clusters=3,n_init=10,random_state=0)
k_means.fit(x)
labels=k_means.labels_
centroids=k_means.cluster_centers_

for i,centroid in enumerate(centroids):
  print(f"centroid{i+1} {centroid}")
colours=['blue','yellow','green']
cluster_colours=[colours[label] for label in labels]
plt.scatter(x.iloc[:,0],x.iloc[:,1],c=cluster_colours)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=200,linewidths=3,color='red',label='centroid') 
for i,color in enumerate(colours):
  plt.scatter([],[],c=color)
plt.legend()
plt.show()
