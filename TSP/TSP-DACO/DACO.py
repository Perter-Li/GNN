import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from functools import reduce
import sys

import torch
from utils import load_model

class Cities():
    def __init__(self,num_cities):
        self.num_cities = num_cities
        self.pos =self.__generate(self.num_cities)
    
    
    def __generate(self,num_node):
        np.random.seed(0)
        return np.random.rand(num_node,2)
    
    # calculate the distance between the cities
    def distance(self):
        pos = self.position()
        nrow, ncol = self.num_cities,self.num_cities
        distance = np.zeros((nrow, ncol),dtype=np.float64)
        for i in range(nrow):
            for j in range(ncol):
                temp_distance=np.power(pos[i,0]-pos[j,0],2)+np.power(pos[i,1]-pos[j,1],2)
                temp_distance =np.power(temp_distance,0.5)
                distance[i,j] = float(temp_distance)
        
        return distance
                
    
    def position(self):
        return self.pos
    
        
    
    def plot_nodes(self):
        # self.pos :[num_node,2]
        plt.xlabel('x-coordinate'),plt.ylabel('y-coordinate')
        plt.title('TSP-Nodes')
        pos = self.position()
        plt.scatter(self.pos[:,0],self.pos[:,1])
        plt.savefig('node.jpg')

class Ant(object):
    
    def __init__(self, id, alpha, beta, cities):
        self.ID=id
        self.ALPHA = alpha
        self.BETA = beta
        self.cities = cities
        self.__clean_data()
        
    def __clean_data(self):
        self.path = [] # 当前蚂蚁路径
        self.total_distance = 0.0 # 当前路径总距离
        self.move_count = 0 # 移动次数
        self.current_city = -1 # 当前停留的城市
        self.city_num = self.cities.num_cities
        self.open_table_city = [True for i in range(self.city_num)
        ]
        self.distance = self.cities.distance()
        
        city_index = random.randint(0,self.city_num-1) # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1
        
    def update(self, path = None):
        if path is None:
            print("The path is invalid!")
            return
        
        self.path = path
        self.total_distance = self.__cal_total_distance()
        self.move_count=len(path)
        self.current_city = path[-1]
        self.open_table_city = [False for i in range(self.city_num)]
        
        
    # 选择下一个城市    
    def __choice_next_city(self, pheromone):
        next_city = -1
        select_cities_prob = [0.0 for i in range(self.city_num)]
        total_prob = 0.0
        
        for i in range(self.city_num):
            if self.open_table_city[i]:
                try:
                    ## 计算概率：与信息素浓度成正比，与距离成反比
                    select_cities_prob[i] = np.power(pheromone[self.current_city,i], self.ALPHA) * np.power((1.0/self.distance[self.current_city,i]), self.BETA)
                    total_prob += select_cities_prob[i] # 分母项
                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i))
                    sys.exit(1)
                    
                    
        # 轮盘选择城市
        if total_prob > 0.0:
            temp_prob = random.uniform(0.0,total_prob)
            for i in range(self.city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_cities_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
        
        # 有可能找不到下一个城市，例如：所有城市均被访问到了，或者是其他情况
        if (next_city == -1):
            next_city = random.randint(0, self.city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, self.city_num - 1)
                
        return next_city
    
    # 计算路径总距离
    def __cal_total_distance(self):
        temp_distance = 0.0
        
        for i in range(1, self.city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += self.distance[start][end]
        
        # 回路
        end = self.path[0]
        temp_distance += self.distance[start][end]
        
        return temp_distance
        # self.total_distance = temp_distance
    
    # 移动操作
    def __move(self, next_city):
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += self.distance[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1
        
    # 搜索路径
    def search_path(self, pheromone):
        self.__clean_data() # 这一步清楚操作很重要，否则将只能搜索一代就不再更新了
        # 搜索路径，直到遍历完所有城市为止
        while self.move_count < self.city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city(pheromone)
            self.__move(next_city)
        
        self.__cal_total_distance()
        
        

class DACO(object):
    def __init__(self, generations, model, cities, pheromone, ant_num, city_num, alpha, beta, rho, q):
       self.generations = generations
       self.model = model
       self.cities = cities
       self.pheromone = pheromone
       self.city_num = city_num
       self.ALPHA = alpha
       self.BETA = beta
       self.RHO = rho
       self.Q = q
       self.ants = [Ant(ID,self.ALPHA,self.BETA,self.cities) for ID in range(ant_num)]
       self.best_ant = Ant(-1,self.ALPHA,self.BETA,self.cities)
       self.best_ant.total_distance = 1 << 31
    
    def search_path(self):
        model_path = self.get_model_solution()
        self.best_ant.update(model_path)
        self.model_ant =copy.deepcopy(self.best_ant)
        
        for generation in range(self.generations):
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path(self.pheromone)
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            print(f'Generation:{generation+1}, cost:{self.best_ant.total_distance}')
            self.__update_pheromone()
        
        return self.best_ant, self.best_ant.total_distance, self.best_ant.path  
        
        
    
    def get_model_solution(self):
        tour = []
        oracle = self.make_oracle()
        while(len(tour) < self.city_num):
            p = oracle(tour)
            # Greedy
            i = np.argmax(p)
            tour.append(i)
            
        return tour
            
    def make_oracle(self,temperature=1.0):
        num_nodes = self.city_num
    
        xyt = torch.tensor(self.cities.position()).float()[None]  # Add batch dimension
        
        with torch.no_grad():  # Inference only
            embeddings, _ = self.model.embedder(self.model._init_embed(xyt))

            # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
            fixed = self.model._precompute(embeddings)
        
        def oracle(tour):
            with torch.no_grad():  # Inference only
                # Input tour with 0 based indices
                # Output vector with probabilities for locations not in tour
                tour = torch.tensor(tour).long()
                if len(tour) == 0:
                    step_context = self.model.W_placeholder
                else:
                    step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

                # Compute query = context node embedding, add batch and step dimensions (both 1)
                query = fixed.context_node_projected + self.model.project_step_context(step_context[None, None, :])

                # Create the mask and convert to bool depending on PyTorch version
                mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
                mask[tour] = 1
                mask = mask[None, None, :]  # Add batch and step dimension

                log_p, _ = self.model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
                p = torch.softmax(log_p / temperature, -1)[0, 0]
                assert (p[tour] == 0).all()
                assert (p.sum() - 1).abs() < 1e-5
                #assert np.allclose(p.sum().item(), 1)
            return p.numpy()
        
        return oracle
    
     # 更新信息素信息
    def __update_pheromone(self):
        
        # 获取每只蚂蚁在路径上留下的信息素
        temp_pheromone = np.zeros((self.city_num,self.city_num),dtype = np.float64)
        for ant in self.ants:
            for i in range(1, self.city_num):
                start, end = ant.path[i-1],ant.path[i]
                
                # 策略1
                temp_pheromone[start,end] += self.Q/ant.total_distance
                temp_pheromone[end,start] = temp_pheromone[start][end]
        
        # 更新所有城市之间的信息素，就信息素衰减加上新迭代信息素
        for i in range(self.city_num):
            for j in range(self.city_num):
                self.pheromone[i,j] = self.pheromone[i,j] * self.RHO + temp_pheromone[i,j]
                
    
    def plot_path(self):
        pos = self.cities.position()
        best_path = self.best_ant.path
        model_path = self.model_ant.path
        
        fig,((ax1),(ax2))=plt.subplots(2,1,figsize=(15,30),sharex=True,sharey=False) 
        for i in range(1+len(path)):
            if i == len(path):
                break
            ax1.plot([pos[model_path[i],0],pos[model_path[i-1],0]], [pos[model_path[i],1],pos[model_path[i-1],1]], color='y')
            ax1.scatter(pos[path[i]-1,0], pos[path[i]-1,1], color='b')
        # plt.set_title('{} nodes, total length {:.2f}'.format(len(self.tour), self.total_distance))
        ax1.set_title('model length {:.2f}'.format(self.model_ant.total_distance))
        for i in range(1+len(path)):
            if i == len(path):
                break
            ax2.plot([pos[best_path[i],0],pos[best_path[i-1],0]], [pos[best_path[i],1],pos[best_path[i-1],1]], color='r')
            ax2.scatter(pos[path[i]-1,0], pos[path[i]-1,1], color='b')
            
        ax2.set_title('daco length {:.2f}'.format(self.best_ant.total_distance))
        plt.savefig('path-100.jpg')
        plt.show()            
    
                
    
        
        
       
           
if __name__ == '__main__':
   # ACO Configuration
   (alpha, beta, rho, q) = (1.0,2.0,0.5,100.0) 
   (city_num, ant_num) = (100,50)
   generations = 200
   cities = Cities(city_num)
   pheromone = np.full((city_num,city_num),1.0)
   
   # Deep Model Configuration
   model, _ = load_model('pretrained/tsp_100/epoch-99.pt')
   model.eval()  # Put in evaluation mode to not track gradients
   daco = DACO(generations,model,cities,pheromone,ant_num,city_num,alpha,beta,rho,q)
   
   best_ant, cost, path = daco.search_path()
   print(f'cost;{cost}\t paht:{path}')
   daco.plot_path()
          
        