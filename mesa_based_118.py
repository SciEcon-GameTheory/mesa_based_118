import matplotlib.pyplot as plt
import numpy as np
import random
import math

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class PoolSelectionAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.states = self.model.initial_state
        self.lbd = 1 / 600;     self.propagation = 0.005
        self.R = 1000;          self.rou = 2
        self.p = 0.01;          self.block_size = [100, 100]
        self.w = [30, 20];      self.population = self.model.num_agent * self.model.initial_state
        self.T = 600

    def sum(self):
        sum_value = 0
        for i in range(0, self.model.width):
            sum_value += self.w[i] * self.states[i]
        return sum_value

    def generate_probability(self, current_pool, new_pool):
        j = new_pool[0]; o = current_pool[0]
        # payoff for the potential pool
        yj_p1 = (self.R + self.rou * self.block_size[j]) / (self.model.num_agent * self.states[j])
        yj_p2 = (self.w[j] * self.states[j]) / self.sum()
        yj_p3 = math.exp(-self.block_size[j] * (self.propagation) / self.T)
        yj_p4 = self.p * self.w[j]
        yj = yj_p1 * yj_p2 * yj_p3 - yj_p4
        # payoff for the origin pool
        yo_p1 = (self.R + self.rou * self.block_size[o]) / (self.model.num_agent * self.states[o])
        yo_p2 = (self.w[o] * self.states[o]) / self.sum()
        yo_p3 = math.exp((-self.block_size[o] * (self.propagation)) / self.T)
        yo_p4 = self.p * self.w[o]
        yo = yo_p1 * yo_p2 * yo_p3 - yo_p4
        probability = self.states[j] * max((yj - yo), 0)

        return probability

    def update_state(self):
        states = []
        for cell in self.model.grid.coord_iter():
            cell_content, x, y = cell
            agent_count = len(cell_content)
            states.append(agent_count)
        states = np.array(states)
        states = states / self.model.num_agent
        self.states = states

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        current_pool = self.pos
        new_pool = self.random.choice(possible_steps)
        SwitchProbability = self.generate_probability(current_pool=current_pool, new_pool=new_pool)
        # print("SwitchProbability: ", SwitchProbability)

        if current_pool == new_pool:
            pass
        else:
            if random.random() < SwitchProbability:
                self.model.grid.move_agent(self, new_pool)
            else:
                pass
        self.update_state()

    def step(self):
        self.move()

class PoolSelectionModel(Model):

    def __init__(self, N, width, height, initial_state):
        super().__init__()
        self.running = True
        self.num_agent = N
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.initial_state = initial_state
        self.width = width; self.height = height

        for i in range(self.num_agent):
            a = PoolSelectionAgent(i, self)
            self.schedule.add(a)
            # Add the agents to a random grid cell based on the settings
            x = np.random.choice([i for i in range(width)], p=self.initial_state)
            y = height - 1
            self.grid.place_agent(a, (x, y))
        # self.datacollector = DataCollector()

    def step(self):
        self.schedule.step()

if __name__ == '__main__':
    model = PoolSelectionModel(N=5000, width=2, height=1,
                               initial_state=np.array([0.75, 0.25]))
    for i in range(200):
        for cell in model.grid.coord_iter():
            cell_content, x, y = cell
            agent_count = len(cell_content)
            print("Round id: ", i)
            print(agent_count)
        print('----------------------------------------')
        model.step()

    # agent_counts = np.zeros((model.grid.width, model.grid.height))
    # for cell in model.grid.coord_iter():
    #     cell_content, x, y = cell
    #     agent_count = len(cell_content)
    #     agent_counts[x][y] = agent_count
    # plt.imshow(agent_counts, interpolation='nearest')
    # plt.colorbar()
    # plt.show()