import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from functools import reduce
import sys

import torch
from utils import load_model
from preprocess import Cities

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
        
        

class EDACO(object):
    def __init__(self, generations, model, cities, pheromone, ant_num, groups, city_num, alpha, beta, rho, q):
       self.generations = generations
       self.model = model
       self.cities = cities
       self.pheromone = pheromone
       self.groups = groups
       self.city_num = city_num
       self.ALPHA = alpha
       self.BETA = beta
       self.RHO = rho
       self.Q = q
       self.ants = [Ant(ID,self.ALPHA,self.BETA,self.cities) for ID in range(ant_num)]
       self.best_ant = Ant(-1,self.ALPHA,self.BETA,self.cities)
       self.best_ant.total_distance = 1 << 31
    
    def search_path(self):
        model_path = self.combine_group_model_path()
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
        
        
    
    def get_model_solution(self, group):
        tour = []
        oracle = self.make_oracle(group)
        while(len(tour) < len(group)):
            p = oracle(tour)
            # Greedy
            i = np.argmax(p)
            tour.append(i)
            
        return tour
    
    def combine_group_model_path(self):
        groups_tour=[]
        for group in self.groups:
            tour_indexes =self.get_model_solution(group)
            groups_tour.append([group[i] for i in tour_indexes])
        self.groups_tour = groups_tour
        # # 简单策略， 将每一个组的最后一个值，与紧随其后的一个组的第一个值进行组合    
        # tour = []
        # for group in groups_tour:
        #     for node_num in group:
        #         tour.append(node_num)
        
        # 贪婪策略， 查找两个组中最近的距离
        def get_min_distance_between_groups(group_i, group_j):
            distance = self.cities.distance()
            minDist = np.finfo(np.float64).max
            node_1, node_2 = -1,-1
            for first in group_i:
                for second in group_j:
                    if minDist > distance[first, second]:
                        node_1, node_2 = first, second
                        minDist = distance[first,second]
            if node_1 < 0 or node_2 < 0:
                print(f'Could find a vilid minimum distance between group {group_i} and group {group_j}')
                exit(-1)
            return node_1,node_2
                    
                    
        tour = []
        
        for i in range(len(groups_tour)):
            if i == 0:
                group_first = groups_tour[0]
            else:
                group_second = groups_tour[i]
                (node_1, node_2) = get_min_distance_between_groups(group_first, group_second)
                node_1_index = group_first.index(node_1)
                node_2_index = group_second.index(node_2)
                # 按照重连接节点，将两个组划分为四个部分
                group_first_front=copy.deepcopy(group_first[:node_1_index]) # [0:node_1_index-1]
                group_first_tail= copy.deepcopy(group_first[node_1_index+1:]) # [node_1_index+1:]
                group_second_front=copy.deepcopy(group_second[:node_2_index]) # [0:node_2_index-1]
                group_second_tail= copy.deepcopy(group_second[node_2_index+1:]) # [node_2_index+1:]
                
                # print(f'group_first_front:{group_first_front}')
                # print(f'group_second_tail:{group_second_tail}')
                # print(f'group_second_front.reverse:{group_second_front[::-1]}')
                # print(f'group_first_tail:{group_first_tail}')
                
                assert len(set(group_first_front)) == len(group_first_front), 'group_first_font 存在重复元素'
                assert len(set(group_first_tail)) == len(group_first_tail), 'group_first_tail 存在重复元素'
                assert len(set(group_second_front)) == len(group_second_front), 'group_second_front 存在重复元素'
                assert len(set(group_second_tail)) == len(group_second_tail), 'group_second_tail 存在重复元素'
                # 四种方案中选择最优的路径
                tour_1 = group_first_front + list([node_1, node_2]) + group_second_tail + group_second_front[::-1]+ group_first_tail
                tour_2 = group_first_front + list([node_1, node_2]) + group_second_front[::-1] + group_second_tail + group_first_tail
                tour_3 = group_first_front + group_second_front[::-1] + group_second_tail[::-1] + list([node_2,node_1]) + group_first_tail
                tour_4 = group_first_front + group_second_tail + group_second_front + list([node_2, node_1]) + group_first_tail
                
                assert len(set(tour_1)) == len(tour_1), 'tour_1 存在重复元素'
                assert len(set(tour_2)) == len(tour_2), 'tour_2 存在重复元素'
                assert len(set(tour_3)) == len(tour_3), 'tour_3 存在重复元素'
                assert len(set(tour_4)) == len(tour_4), 'tour_4 存在重复元素'
                def get_tour_distance(tour):
                    temp_distance = 0.0
                    distance = self.cities.distance()
                    for i in range(1, len(tour)):
                        start, end = tour[i], tour[i-1]
                        temp_distance += distance[start][end]
                    
                    # 回路
                    end = tour[0]
                    temp_distance += distance[start][end]
                    
                    return temp_distance
                tour_list = [tour_1,tour_2, tour_3, tour_4]
                tour_distance=[get_tour_distance(path) for path in tour_list]
                best_tour = tour_list[tour_distance.index(min(tour_distance))]
                group_first = copy.deepcopy(best_tour)
                tour = copy.deepcopy(best_tour)
                # temp = []
                # for i1 in range(len(group_first)):
                #     temp.append(group_first[i1])
                #     if group_first[i1] == node_1:
                #         node_2_index = group_second.index(node_2)
                #         for i2 in range(len(group_second)):
                #             temp.append(group_second[(i2+node_2_index)%len(group_second)])
                # group_first = copy.deepcopy(temp)
                # tour = copy.deepcopy(temp)
                          
        print(f'Combined tour is\t {tour}')             
        return tour
    
        
      
              
    def make_oracle(self,group,temperature=1.0):
        num_nodes = len(group)
    
        xyt = torch.tensor(self.cities.group_position(group)).float()[None]  # Add batch dimension
        
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
        for i in range(1+len(model_path)):
            if i == len(model_path):
                break
            ax1.plot([pos[model_path[i],0],pos[model_path[i-1],0]], [pos[model_path[i],1],pos[model_path[i-1],1]], color='y')
            ax1.scatter(pos[model_path[i]-1,0], pos[model_path[i]-1,1], color='b')
        # plt.set_title('{} nodes, total length {:.2f}'.format(len(self.tour), self.total_distance))
        ax1.set_title('model length {:.2f}'.format(self.model_ant.total_distance))
        for i in range(1+len(best_path)):
            if i == len(best_path):
                break
            ax2.plot([pos[best_path[i],0],pos[best_path[i-1],0]], [pos[best_path[i],1],pos[best_path[i-1],1]], color='r')
            ax2.scatter(pos[best_path[i]-1,0], pos[best_path[i]-1,1], color='b')
            
        ax2.set_title('daco length {:.2f}'.format(self.best_ant.total_distance))
        plt.savefig('path-100.jpg')
        plt.show()
        
    def plot_groups(self,colors=None):
        assert len(self.groups_tour)==len(colors), 'Dismatch of the len of groups_tours and colors'
        pos = self.cities.position()
        plt.figure(figsize=(15,15))
        for group, color in zip(self.groups_tour ,colors):
            for i in range(1, len(group)):
               plt.plot([pos[group[i],0],pos[group[i-1],0]], [pos[group[i],1],pos[group[i-1],1]], color=color)
               plt.scatter(pos[group[i]-1,0], pos[group[i]-1,1], color='b')
            plt.plot([pos[group[i],0],pos[group[0],0]], [pos[group[i],1],pos[group[0],1]], color=color)
        plt.savefig('path-cluster-results.jpg')
                     
    
                
    
        
        
       
           
if __name__ == '__main__':
   # ACO Configuration
   (alpha, beta, rho, q) = (1.0,2.0,0.5,100.0) 
   (city_num, ant_num) = (100,50)
   generations = 200
   cities = Cities(city_num)
   groups = cities.split_data(20, 'cluster')
   pheromone = np.full((city_num,city_num),1.0)
   
   # Deep Model Configuration
   model, _ = load_model('pretrained/tsp_20/epoch-99.pt')
   model.eval()  # Put in evaluation mode to not track gradients
   daco = EDACO(generations,model,cities,pheromone,ant_num,groups,city_num,alpha,beta,rho,q)
   
   best_ant, cost, path = daco.search_path()
   print(f'cost;{cost}\t paht:{path}')
   daco.plot_path()
   colors =['#c72e29','#098154','#fb832d', 'red', 'blue']#三种不同颜色
   daco.plot_groups(colors)
          
        