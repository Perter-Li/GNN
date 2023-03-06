import numpy as np
import copy
from matplotlib import pyplot as plt
import random
import seaborn as sns

import torch
from utils import load_model
from preprocess import Cities


class Particle:
    def __init__(self, path, cities, num_city):
        self.distance = cities.distance()
        self.num_city = num_city
        self._path = path
        self._best_path = path
    
    def path(self):
        return self._path
    
    def best_path(self):
        return self._best_path
    
    def update_path(self, new_path):
        self._path = new_path
        if self.__cost(new_path) <= self.__cost(self._best_path):
            self._best_path = new_path
    
    
    def __cost(self, path):
        temp_distance = 0.0
        for i in range(1, self.num_city):
            start, end = path[i], path[i-1]
            temp_distance += self.distance[start][end]
        
        # 回路
        end = path[0]
        temp_distance += self.distance[start][end]
        return temp_distance
        
    def cost(self):
        return self.__cost(self._path)


class DPSO:
    def __init__(self, generations, model, cities, num_city, num_particles, alpha, beta):
        self.generations = generations
        self.model = model
        self.cities = cities
        self.num_city = num_city
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.particles, self.particles_fitness = self.init_particles()
        self.gbest = self.particles[self.particles_fitness.argmin()]
        self.gbest_fitness = self.particles_fitness.min()
        
        
    def init_particles(self):
            particles =[]
            particles_fitness =np.zeros((self.num_particles+1, 1))
            # 初始化种群各个粒子的位置，作为个体的历史最优pbest
            init_path = np.zeros((self.num_particles+1, self.num_city), dtype=np.int64)
            model_path = self.get_model_solution()
            
            for i in range(self.num_particles+1):
                if i == 0:
                    init_path[i]=model_path
                else:
                    init_path[i] = np.random.choice(list(range(self.num_city)), size=self.num_city, replace=False)
                particle = Particle(list(init_path[i]), self.cities, self.num_city)
                particles.append(particle)
                particles_fitness[i] = particle.cost()
            
            return particles, particles_fitness
                
    # 定义位置更新函数。
    def do_ss(self, x_i, ss):
        """
        执行交换操作
        :param x_i:
        :param ss: 由交换子组成的交换序列(swap sequence)
        :return:
        """
        for i, j, r in ss:
            rand = np.random.random()
            if rand <= r:
                x_i[i], x_i[j] = x_i[j], x_i[i]
        return x_i 
    
    #定义速度更新函数
    def get_ss(self, x_best, x_i, r):
        """
        计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
        :param x_best: pbest or gbest
        :param x_i: 粒子当前的解
        :param r: 随机因子
        :return:
        """
        # x_i_backup= copy.deepcopy(x_i)
        
        # velocity_ss = []
        # for i in range(len(x_i_backup)):
        #     if x_i_backup[i] != x_best[i]:
        #         j = np.where(x_i_backup == x_best[i])[0][0]
        #         so = (i, j, r)  # 得到交换子
        #         velocity_ss.append(so)
        #         x_i_backup[i],x_i_backup[j] = x_i_backup[j], x_i_backup[i]  # 执行交换操作

        # return velocity_ss   
        velocity_ss = []
        for i in range(len(x_i)):
            if x_i[i] != x_best[i]:
                j = np.where(x_i == x_best[i])[0][0]
                so = (i, j, r)  # 得到交换子
                velocity_ss.append(so)
                x_i[i], x_i[j] = x_i[j], x_i[i]  # 执行交换操作

        return velocity_ss
    
    def search_path(self):
        # model_path = self.get_model_solution()
        # self.gbest.update_path(model_path)
        self.model_best = copy.deepcopy(self.gbest)
        self.gbest_fitness = self.gbest.cost()
        self.fitness_value_lst=[]
        for generation in range(self.generations):
            for i,particle in enumerate(self.particles):
                pbest_path = copy.deepcopy(particle.best_path())
                curr_path = copy.deepcopy(particle.path())
                gbest_path = self.gbest.path()
                # 计算交换序列，即 v = r1(pbest-xi) + r2(gbest-xi)
                ss1 = self.get_ss(pbest_path, curr_path, self.alpha)
                ss2 = self.get_ss(gbest_path, curr_path, self.beta)
                ss = ss1 + ss2
                new_path = self.do_ss(curr_path, ss)

                particle.update_path(new_path)
                fitness_new = particle.cost()
                fitness_old = self.particles_fitness[i]
                if fitness_new < fitness_old:
                    self.particles_fitness[i] = fitness_new

                # best_fitness_old = self.particles_best_fitness[i]
                # if fitness_new < best_fitness_old:
                #     self.particles_best_fitness[i]= fitness_new
                
            gbest_fitness_new = self.particles_fitness.min()
            gbest_new = self.particles[self.particles_fitness.argmin()]
            if gbest_fitness_new < self.gbest_fitness:
                self.gbest_fitness = gbest_fitness_new
                self.gbest = gbest_new 
            print(f'Generation: {generation} \t cost: {self.gbest_fitness}')
            self.fitness_value_lst.append(self.gbest_fitness)
            
        return self.gbest, self.gbest_fitness, self.gbest.path()
    
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
        best_path = self.gbest.path()
        model_path = self.model_best.path()
        
        fig,((ax1),(ax2))=plt.subplots(2,1,figsize=(15,30),sharex=True,sharey=False) 
        for i in range(1+len(model_path)):
            if i == len(model_path):
                break
            ax1.plot([pos[model_path[i],0],pos[model_path[i-1],0]], [pos[model_path[i],1],pos[model_path[i-1],1]], color='y')
            ax1.scatter(pos[model_path[i]-1,0], pos[model_path[i]-1,1], color='b')
        # plt.set_title('{} nodes, total length {:.2f}'.format(len(self.tour), self.total_distance))
        ax1.set_title('model length {:.2f}'.format(self.model_best.cost()))
        for i in range(1+len(best_path)):
            if i == len(best_path):
                break
            ax2.plot([pos[best_path[i],0],pos[best_path[i-1],0]], [pos[best_path[i],1],pos[best_path[i-1],1]], color='r')
            ax2.scatter(pos[best_path[i]-1,0], pos[best_path[i]-1,1], color='b')
            
        ax2.set_title('dpso length {:.2f}'.format(self.gbest.cost()))
        plt.savefig('(dpso)path-50.jpg')  
        
    def plot_loss(self):
         #绘图    
        sns.set_style('whitegrid')
        # plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
        # plt.rcParams['axes.unicode_minus'] = False
        plt.figure(2)
        plt.plot(range(1,1+len(self.fitness_value_lst)), self.fitness_value_lst)
        plt.title('Process')
        plt.ylabel('Optimal')
        plt.xlabel('Iteration({}->{})'.format(0, self.generations))
        plt.savefig('(dpso-loss)path-50.jpg')
        # plt.show()
        
                
if __name__ == '__main__':
    (num_city, num_particles) = (50,700)
    generations = 500
    # alpha,beta=0.8,0.8
    cities = Cities(num_city)
    
    # Deep Model Configuration
    model, _ = load_model('pretrained/tsp_50/epoch-99.pt')
    model.eval()  # Put in evaluation mode to not track gradients
    alpha, beta = 0.95, 0.95
    dpso = DPSO(generations,model,cities, num_city, num_particles, alpha, beta)
    _,fitness,_ =dpso.search_path()
    dpso.plot_path()
    dpso.plot_loss()
    # min_fitness=10000
    # suitable_a, suitable_b = 0,0
    # for alpha in np.arange(0,1,0.05):
    #     for beta in np.arange(0,1,0.05):
    #         dpso = DPSO(generations,model,cities, num_city, num_particles, alpha, beta)
    #         _,fitness,_ =dpso.search_path()
    #         if fitness < min_fitness:
    #             min_fitness = fitness
    #             suitable_a, suitable_b = alpha,beta
    #         print(f'curr_alpha: {alpha}, curr_beta: {beta}, curr_fitness: {fitness}')
    # print('*'*50)
    # print(f'alpha: {suitable_a}, beta: {suitable_b}, fitness: {min_fitness}')
    