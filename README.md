# EX 10 :  Implementation-of-K-Means-Clustering-for-Customer-Segmentation
# Date : 28/10/24
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the Program.

Step 2: Import dataset and print head,info of the dataset.

Step 3: check for null values.

Step 4: Import kmeans and fit it to the dataset.

Step 5:Plot the graph using elbow method.

Step 6: Print the predicted array.

Step 7: Plot the customer segments.

Step 8: End the program.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: VISHAL M.A
RegisterNumber: 212222230177

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('Exp_9_Mall_Customers.csv')

data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss =[] #Within-cluster Sum pof Square.
#It is the sum of Squared distance between each point & the centroid in the cluster

for i in range(1,11):
    kmeans=KMeans(n_clusters = i, init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km = KMeans(n_clusters =5)
km.fit(data.iloc[:,3:])

KMeans(n_clusters=5)

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="black",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="orange",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="red",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
### Elbow Method:
![graph out](https://github.com/user-attachments/assets/e2152372-e655-4805-81a9-67ac80481b58)


### Y-Predict:
![y predict](https://github.com/user-attachments/assets/b16d4088-ece6-42b8-842d-f4bb64329022)


### Customer Segments:
![cst out](https://github.com/user-attachments/assets/68d2e1c7-064c-49a7-947e-32b6fce58668)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
