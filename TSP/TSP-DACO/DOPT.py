
import copy
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import torch
from utils import load_model
from preprocess import Cities

class DOPT2:
    def __init__(self,model, max_count, cities, num_city):
        self.model = model
        self.max_count = max_count
        self.cities = cities
        self.distance = cities.distance()
        self.num_city = num_city
        self.best_path  = np.random.choice(list(range(self.num_city)), size=self.num_city, replace=False)
        # self.= copy.deepcopy(self.cur_path)
    
    
    
    def search_path(self):
        self.fitness_value_lst=[]
        model_path = self.get_model_solution()
        self.best_path = model_path
        self.model_path = model_path
        best_cost = self.cost(self.best_path)
        for count in range(self.max_count):
            cur_path = self.best_path
            cur_cost = self.cost(self.best_path)
            # Swap all possible nodes in current path
            for i in range(self.num_city-1):
                for j in range(i+1, self.num_city):
                    new_path = self.opt2Swap(cur_path,i, j)
                    new_cost = self.cost(new_path)
                    if new_cost < cur_cost:
                        self.best_path = new_path
                        # self.cur_path = new_path
                        best_cost = new_cost
            print(f'Count:{count+1}, cost: {best_cost}')
            self.fitness_value_lst.append(best_cost)
            
        return self.best_path, self.cost(self.best_path)
    
    def opt2Swap(self, path, v1, v2):
    
        before_v1 =copy.deepcopy(path[:v1+1])
        between = copy.deepcopy(path[v1+1:v2+1])
        after_v2 = copy.deepcopy(path[v2+1:])
        
        current = list(before_v1) + list(between[::-1]) + list(after_v2)
        return current                    
                    
    def cost(self,path):
        temp_distance = 0.0
        for i in range(1, self.num_city):
            start, end = path[i], path[i-1]
            temp_distance += self.distance[start][end]
        
        # 回路
        end = path[0]
        temp_distance += self.distance[start][end]
        return temp_distance
    
   
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
            
        ax2.set_title('dopt2 length {:.2f}'.format(self.cost(best_path)))
        plt.savefig('(dopt2)path-50.jpg')  
        
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
        plt.savefig('(dopt2-loss)path-50.jpg')
        # plt.show()
    

        
if __name__ == '__main__':
    max_count = 300
    num_city = 50
    cities = Cities(num_city)
    
    # Deep Model Configuration
    model, _ = load_model('pretrained/tsp_50/epoch-99.pt')
    model.eval()  # Put in evaluation mode to not track gradients
    
    opt2 = DOPT2(model,max_count, cities, num_city)
    cost, path= opt2.search_path()
    print(f'cost;{cost}\t path:{path}')
    opt2.plot_path()
    opt2.plot_loss()
        