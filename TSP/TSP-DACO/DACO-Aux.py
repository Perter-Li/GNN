import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from functools import reduce
import sys
import seaborn as sns

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
        # self.total_distance += self.distance[self.current_city][next_city]
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
        self.total_distance=self.__cal_total_distance()
        
        

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
       
       self.distance = self.cities.distance()
    
    def cost(self, route):
        temp_distance = 0.0
        
        for i in range(1, self.city_num):
            start, end = route[i], route[i-1]
            temp_distance += self.distance[start][end]
        
        # 回路
        end = route[0]
        temp_distance += self.distance[start][end]
        
        return temp_distance
    
    def search_path(self):
        # model_path = self.get_model_solution()
        # self.best_ant.update(model_path)
        self.model_ant =copy.deepcopy(self.best_ant)
      
        ref_model_path = self.get_model_solution(list(range(self.city_num)))
        self.model_ant.update(ref_model_path)
        self.fitness_value_lst =[]
        num_nodes = 20
        for generation in range(self.generations):
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path(self.pheromone)
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            if generation%10==0:
                num_nodes += 4
                num_inner = 6
                print(f'Start model adjust...')
                print(f'Partial Nodes: {num_nodes}')
                for i in range(num_inner):
                    best_path = self.best_ant.path
                    # 利用图神经模型进行局部调优
                    model_adjust_route = self.update_partial_route(num_nodes, best_path)
                    assert len(set(model_adjust_route)) == len(model_adjust_route), 'model_adjust_route 存在重复元素'
                    assert len(set(self.best_ant.path)) == len(self.best_ant.path), 'self.best_ant.path 存在重复元素'
                    model_cost = self.cost(model_adjust_route)
                    best_ant_cost = self.cost(self.best_ant.path)
                    if model_cost < best_ant_cost:
                        self.best_ant.update(model_adjust_route)
                        print(f'Valid model update.')
        
            self.fitness_value_lst.append(self.best_ant.total_distance)
            print(f'Generation:{generation+1}, cost:{self.best_ant.total_distance}')
            self.__update_pheromone()
        
        return self.best_ant, self.best_ant.total_distance, self.best_ant.path  
    
    # 获取best_ant中路径的一部分
    def update_partial_route(self, num_nodes, path):
        best_path = copy.deepcopy(path)
        route_len = len(best_path)
        # 随机设定一个起点si，选择从si+1开始的共num_nodes个节点，倘若到达数组末尾，则从索引0开始
        si = random.randint(0, route_len - 1)
        partial_route =[best_path[(si+i+1)%route_len] for i in range(num_nodes)]
        ei = (si + num_nodes +1)%route_len
        
        model_solution = self.get_model_solution(partial_route)
        
        # 将model产生的解进行拼接， 拼接的方式不是任意的，需要考虑最小距离
        new_partial_route = self.get_min_distance_node_pair(model_solution, best_path[si], best_path[ei])
        new_route = copy.deepcopy(best_path)
        for i in range(num_nodes):
            new_route[(i+si+1)%route_len] = new_partial_route[i] 
        
        return new_route
    
    def get_min_distance_node_pair(self, route, v1, v2):
        # 获取route中到v1,v2中距离最小的边
        distance = self.cities.distance()
        edges = np.zeros((len(route),2), dtype = np.int32)
        for i, node_num in enumerate(route):
            edges[i][0] =  route[i]
            edges[i][1] = route[(i+1)%len(route)]
        
        minDist = np.finfo(np.float64).max
        start, end = -1, -1    
        for edge in edges:
            n1, n2 = edge
            dis1=distance[v1][n1] + distance[v2][n2]-distance[n1][n2]
            dis2= distance[v1][n2] + distance[v2][n1] - distance[n1][n2]
            if min(dis1 ,dis2) < minDist:
                minDist = min(dis1, dis2)
                if dis1 < dis2:
                    start = n1
                    end = n2
                else:
                    start = n2
                    end = n1
                    
        si = route.index(start)
        ei = route.index(end)
        
        if route[(si+1)%len(route)] == end:
            new_route = list([route[si]])+ list(reversed(route[:si])) + list(reversed(route[ei:]))
        else:
            new_route = list(route[si::]) + list(route[:ei]) + list([route[ei]])
        
        return copy.deepcopy(new_route) 
                
            
            
        
    def get_model_solution(self, partial_route):
        # tour只能记录partial_route中的节点的索引序号
        tour = []
        oracle = self.make_oracle(partial_route)
        while(len(tour) < len(partial_route)):
            p = oracle(tour)
            # Greedy
            i = np.argmax(p)
            tour.append(i)
        # 在这里对应成正确的节点
        new_route = [partial_route[node_num] for node_num in tour]    
        return new_route
    
    
            
    def make_oracle(self, partial_route,temperature=1.0):
        
        def get_partial_postion(partial_route):
            route_pos = np.zeros((len(partial_route),2), dtype = np.float64)
            for i, node_num in enumerate(partial_route):
                route_pos[i] = self.cities.pos[node_num]
            return route_pos
            
        partial_pos= get_partial_postion(partial_route)
        
        num_nodes = len(partial_route)
        xyt = torch.tensor(partial_pos).float()[None]  # Add batch dimension
        
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
            if i == len( model_path):
                break
            ax1.plot([pos[model_path[i],0],pos[model_path[i-1],0]], [pos[model_path[i],1],pos[model_path[i-1],1]], color='y')
            ax1.scatter(pos[ model_path[i]-1,0], pos[ model_path[i]-1,1], color='b')
        # plt.set_title('{} nodes, total length {:.2f}'.format(len(self.tour), self.total_distance))
        ax1.set_title('model length {:.2f}'.format(self.model_ant.total_distance))
        for i in range(1+len(best_path)):
            if i == len(best_path):
                break
            ax2.plot([pos[best_path[i],0],pos[best_path[i-1],0]], [pos[best_path[i],1],pos[best_path[i-1],1]], color='r')
            ax2.scatter(pos[best_path[i]-1,0], pos[best_path[i]-1,1], color='b')
            
        ax2.set_title('daco length {:.2f}'.format(self.best_ant.total_distance))
        plt.savefig(f'(daco)path-{self.city_num}.jpg')
        plt.show()     
        
    def plot_loss(self):
         #绘图    
        sns.set_style('whitegrid')
        # plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
        # plt.rcParams['axes.unicode_minus'] = False
        plt.figure(2)
        plt.plot(range(1,1+len(self.fitness_value_lst)), self.fitness_value_lst)
        plt.title('Process')
        plt.ylabel('Optimal')
        plt.xlabel('Iteration({}->{})'.format(0, len(self.fitness_value_lst)))
        plt.savefig(f'(daco-loss)path-{self.city_num}.jpg')
        # plt.show()       
    
                
    
        
        
       
           
if __name__ == '__main__':
   # ACO Configuration
   (alpha, beta, rho, q) = (1.0,2.0,0.5,100.0) 
   (city_num, ant_num) = (200,50)
   generations = 200
   cities = Cities(city_num)
   pheromone = np.full((city_num,city_num),1.0)
   
   # Deep Model Configuration
   model, _ = load_model('pretrained/tsp_50/epoch-99.pt')
   model.eval()  # Put in evaluation mode to not track gradients
   daco = DACO(generations,model,cities,pheromone,ant_num,city_num,alpha,beta,rho,q)
   
   best_ant, cost, path = daco.search_path()
   print(f'cost;{cost}\t paht:{path}')
   daco.plot_path()
   daco.plot_loss()
          
        