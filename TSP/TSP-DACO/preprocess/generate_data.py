import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from functools import reduce
import sys
import math

from clustering.kmeans import kmeans

class Cities():
    def __init__(self,num_cities):
        self.num_cities = num_cities
        self.pos =self.__generate(self.num_cities)
        self._distance = self.caldistance()
        
    def caldistance(self):
        pos = self.position()
        nrow, ncol = self.num_cities,self.num_cities
        distance = np.zeros((nrow, ncol),dtype=np.float64)
        for i in range(nrow):
            for j in range(ncol):
                temp_distance=np.power(pos[i,0]-pos[j,0],2)+np.power(pos[i,1]-pos[j,1],2)
                temp_distance =np.power(temp_distance,0.5)
                distance[i,j] = float(temp_distance)
        
        return distance
    
    
    def __generate(self,num_node):
        np.random.seed(0)
        return np.random.rand(num_node,2)
    
    # calculate the distance between the cities
    def distance(self):
        return self._distance
                
    
    def position(self):
        return self.pos
    
    def group_position(self,group):
        group_pos = np.zeros((len(group),2), dtype = np.float64)
        for i, node_num in enumerate(group):
            group_pos[i] = self.pos[node_num]
        return group_pos
            
                
    
    def plot_nodes(self):
        # self.pos :[num_node,2]
        plt.xlabel('x-coordinate'),plt.ylabel('y-coordinate')
        plt.title('TSP-Nodes')
        pos = self.position()
        plt.scatter(self.pos[:,0],self.pos[:,1])
        plt.savefig('node.jpg')
        
    
    '''
    Split the origin data, there are three method to achieve the goal, e.i random, cluster, order.
    random method is to split the data indexes randomly
    order method is to split the data indexes according to the order of them
    cluster method splits the data indexes by cluster depending on the distance of them, is a unsupervised method
    by the way, the output data group will be num_cities // group_size    
    '''    
    def split_data(self, group_size=1, method = "random"):
        self.group_size = group_size
        num_nodes =  self.num_cities
        num_groups = num_nodes // group_size
        if num_groups == 0:
           print(f'The group size is greater than the number of nodes, please set a appropriate size!')
           sys.exit(-1)
           
        
        if method == "random":
            indexes = np.arange(self.num_cities)
            np.random.shuffle(indexes)
            groups = []
            i = 0
            while i < num_groups:
                groups.append(indexes[i*group_size:(i+1)*group_size])
                i=i+1
            
            # there exists residual nodes
            if num_nodes % group_size:
                groups.append(indexes[i*group_size:])
            
            self.groups = groups
            return groups    
        elif method == "order":
            indexes = np.arange(self.num_cities)
            groups = []
            i = 0
            while i < num_groups:
                groups.append(indexes[i*group_size:(i+1)*group_size])
                i=i+1
            
            # there exists residual nodes
            if num_nodes % group_size:
                groups.append(indexes[i*group_size:])
            
            self.groups = groups
            return groups    
        else:
            centroids, groups = kmeans(self.pos, math.ceil(num_nodes / group_size), group_size = group_size)
            self.groups = groups
            return groups
           
        
        
if __name__ == '__main__':
    cities = Cities(250)
    groups = cities.split_data(100, method = "cluster")
    for i, group in enumerate(groups):
        print(f'Group {i} size {len(group)}\nNodes {group} \n {"*" * 50}')       
