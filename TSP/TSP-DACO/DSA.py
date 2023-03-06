
import numpy as np
import copy
import math
import random
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from utils import load_model
from preprocess import Cities

class DSA:
    def __init__(self,  model, cities, num_city, ts,te, num_loop=100,a=0.98):
        self.model = model 
        self.cities = cities
        self.num_city = num_city
        self.ts = ts
        self.te = te
        self.num_loop = num_loop
        self.a = a
        self.distance = cities.distance()
        self.cur_path = np.random.choice(list(range(self.num_city)), size=self.num_city, replace=False)
        self.best_path = copy.deepcopy(self.cur_path)
    
    def cost(self,path):
        temp_distance = 0.0
        for i in range(1, self.num_city):
            start, end = path[i], path[i-1]
            temp_distance += self.distance[start][end]
        
        # 回路
        end = path[0]
        temp_distance += self.distance[start][end]
        return temp_distance
    
    def search_path(self):
        
        model_path = self.get_model_solution()
        self.cur_path = model_path
        self.best_path = model_path
        self.model_path = model_path
        # self.model_path = copy.deepcopy(self.cur_path)
        cur_path = self.cur_path
        cur_cost = self.cost(cur_path)
        best_path = self.best_path
        best_cost = self.cost(best_path)
        
        self.fitness_value_lst=[]
        generation = 1
        t = self.ts
        while True:
            if t <= self.te:
                break
            #令温度为初始温度
            for rt2 in range(self.num_loop):
                new_path = self.update_path(cur_path)
                new_cost = self.cost(new_path)
                delt = new_cost - cur_cost
                if delt <= 0:
                    cur_path = new_path
                    cur_cost = new_cost
                    if best_cost > new_cost:
                        best_path = new_path
                        best_cost = new_cost
                elif delt > 0:
                    p = math.exp(-delt / t)
                    ranp = np.random.uniform(0, 1)
                    # Accept suboptimal result with probility p
                    if ranp < p:
                        cur_path = new_path
                        cur_cost = new_cost
            self.fitness_value_lst.append(best_cost)
            print(f'Generation: {generation}, cost:{best_cost}')
            t=t*a
            generation += 1
        
        self.best_path = best_path
        return self.best_path, self.cost(self.best_path)  
    
    # def update_path(self,old_path, num_iter):
    #     # 这里有多种策略可以选择
    #     #如果是偶数次，二变换法
    #     '''
    #     注意：数组直接复制是复制地址
    #     例如， current = route
    #     想要得到一个新的有同样内容的数组，应该用： current = copy.copy(route) 
    #     '''
    #     current = copy.copy(old_path)  
    #     n = self.num_city
    #     if num_iter % 2 == 0:
           
    #         u = random.randint(0, n-1)
    #         v = random.randint(0, n-1)
    #         temp = current[u]
    #         current[u] = current[v]
    #         current[v] = temp
    #     #如果是奇数次，三变换法 
    #     else:
    #         temp2 = random.sample(range(0, n), 3)
    #         temp2.sort()
    #         u = temp2[0]
    #         v = temp2[1]
    #         w = temp2[2]
    #         w1 = w + 1
    #         temp3 = [0 for col in range(v - u + 1)]
    #         j =0
    #         for i in range(u, v + 1):
    #             temp3[j] = current[i]
    #             j += 1
            
    #         for i2 in range(v + 1, w + 1):
    #             current[i2 - (v-u+1)] = current[i2]
    #         w = w - (v-u+1)
    #         j = 0
    #         for i3 in range(w+1, w1):
    #             current[i3] = temp3[j]
    #             j += 1
        
    #     return current
    
    def update_path(self,old_path, prob=[0.3,0.65,0.05]):
        # 这里有多种策略可以选择
        '''
        注意：数组直接复制是复制地址
        例如， current = route
        想要得到一个新的有同样内容的数组，应该用： current = copy.copy(route) 
        '''
        current = list(old_path) 
        n = self.num_city
        
        rand = np.random.uniform(0, 1)
        # vertex insert
        if rand < prob[0]:
            temp = random.sample(range(0, n), 2)
            temp.sort()
           
            u = temp[0]
            v = temp[1]
            before_u = copy.deepcopy(current[:u]) # [:u-1]
            between = copy.deepcopy(current[u+1:v]) # [u+1, v-1]
            after_v = copy.deepcopy(current[v+1:]) # [v+1:]
            
            current = before_u + [current[v]] + between + [current[u]] + after_v
        elif rand < prob[0]+ prob[1]:
            # block reverse
            temp = random.sample(range(0, n), 2)
            temp.sort()
            u = temp[0]
            v = temp[1]
            before_u = copy.deepcopy(current[:u]) # [:u-1]
            between = copy.deepcopy(current[u+1:v]) # [u+1, v-1]
            after_v = copy.deepcopy(current[v+1:]) # [v+1:]
            
            current = before_u+[current[u]]+ between[::-1]+[current[v]]+after_v
        # block insert
        else:
            temp = random.sample(range(0,n),3)
            temp.sort()
            u = temp[0]
            v = temp[1]
            w = temp[2]
            before_u = copy.deepcopy(current[:u]) # [:u-1]
            between_u_v = copy.deepcopy(current[u+1:v]) # [u+1, v-1]
            between_v_w = copy.deepcopy(current[v+1:w]) # [v+1,w-1]
            after_w = copy.deepcopy(current[w+1:]) # [w+1,:]
            
            current = before_u + [current[v]] + between_v_w + [current[w]] + [current[u]] + between_u_v + after_w 
        
        return np.array(current)
    
    def get_model_solution(self):
        tour = []
        oracle = self.make_oracle()
        while(len(tour) < self.num_city):
            p = oracle(tour)
            # Greedy
            i = np.argmax(p)
            tour.append(i)
            
        return tour
            
    def make_oracle(self,temperature=1.0):
        num_nodes = self.num_city
    
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
    
    def plot_path(self):
        pos = self.cities.position()
        best_path = self.best_path
        model_path = self.model_path
        fig,((ax1),(ax2))=plt.subplots(2,1,figsize=(15,30),sharex=True,sharey=False) 
        for i in range(1+len(model_path)):
            if i == len(model_path):
                break
            ax1.plot([pos[model_path[i],0],pos[model_path[i-1],0]], [pos[model_path[i],1],pos[model_path[i-1],1]], color='y')
            ax1.scatter(pos[model_path[i]-1,0], pos[model_path[i]-1,1], color='b')
        # plt.set_title('{} nodes, total length {:.2f}'.format(len(self.tour), self.total_distance))
        ax1.set_title('model length {:.2f}'.format(self.cost(model_path)))
        for i in range(1+len(best_path)):
            if i == len(best_path):
                break
            ax2.plot([pos[best_path[i],0],pos[best_path[i-1],0]], [pos[best_path[i],1],pos[best_path[i-1],1]], color='r')
            ax2.scatter(pos[best_path[i]-1,0], pos[best_path[i]-1,1], color='b')
            
        ax2.set_title('dsa length {:.2f}'.format(self.cost(best_path)))
        plt.savefig('(dsa)path-50.jpg')  
        
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
        plt.savefig('(dsa-loss)path-50.jpg')
        # plt.show()
        
    

if __name__=='__main__':
    num_city = 50
    ts, te = 4000, 1e-8
    num_loop = 600
    a = 0.99
    cities = Cities(num_city)
    
     # Deep Model Configuration
    model, _ = load_model('pretrained/tsp_50/epoch-99.pt')
    model.eval()  # Put in evaluation mode to not track gradients
    
    sa = DSA(model,cities,num_city,ts,te,num_loop,a)
    cost, path = sa.search_path()
    print(f'cost;{cost}\t path:{path}')
    sa.plot_path()    
    sa.plot_loss()     