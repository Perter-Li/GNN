import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Return the Euclidean distance of every node to the all k cetroids
# Thus, clalist is a 2-dimension array of shape num_nodes * k 
def calcDis(nodes, centroids, k):
    clalist=[]
    for node in nodes:
        diff = np.tile(node, (k, 1)) - centroids  # calculate the diffenrence between the node and the centroids
        squaredDiff = diff ** 2     
        squaredDist = np.sum(squaredDiff, axis=1)   #sum along the col axis, to get the value to every centroids
        distance = squaredDist ** 0.5  #开根号
        clalist.append(distance) 
    clalist = np.array(clalist)  
    return clalist    # shape: [num_nodes, k]

# Update the centroids
def classify(nodes, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(nodes, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)    # Get the min index of node to the all centroids 
    newCentroids = pd.DataFrame(nodes).groupby(minDistIndices).mean() # Update the centroids
    newCentroids = newCentroids.values
 
    # Variable "changed" used for judging weather the centroids haved changed 
    changed = newCentroids - centroids
 
    return changed, newCentroids

def rearrange_to_groups(clalist, centroids, group_size):
    groups = [[] for _ in range(len(centroids))]# Get the index of node arranged for the groups
    groups_len = [0] * len(centroids) # Record the number of nodes in every group
    
    for (i,distance) in enumerate(clalist):
        while True:
            minDistIndices = np.argmin(distance) 
            if groups_len[minDistIndices] < group_size:
                groups_len[minDistIndices] += 1
                groups[minDistIndices].append(i)
                break
            else:
                # The node should be arranged to closest centroid which have overload
                # Thus we need to look for another centroid
                distance[minDistIndices] = np.finfo(np.float64).max
                
    return groups
    

# k-means
def kmeans(nodes, k, group_size):
    # select the centroids randomly
    centroids = random.sample(nodes.tolist(), k)
    
    # Update the centroids utils unchanged
    changed, newCentroids = classify(nodes, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(nodes, newCentroids, k)
 
    centroids = sorted(newCentroids.tolist())   
 
    # Get the node closest to the centroids
    clalist = calcDis(nodes, centroids, k) 
    
    groups = rearrange_to_groups(clalist, centroids, group_size)
        
    return centroids, groups


if __name__=='__main__': 
    dataset = np.random.rand(100,2)
    centroids, cluster = kmeans(dataset, 5 , group_size = 20)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
    for i in range(len(dataset)):
      plt.scatter(dataset[i][0],dataset[i][1], marker = 'o',color = 'green', s = 40 ,label = '原始点')
                                                    #  记号形状       颜色      点的大小      设置标签
      for j in range(len(centroids)):
        plt.scatter(centroids[j][0],centroids[j][1],marker='x',color='red',s=50,label='质心')
        plt.savefig('k-means.jpg')
        # plt.show()
