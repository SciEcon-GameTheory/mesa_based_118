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
        self.states = None
        self.lbd = 1 / 600;     self.propagation = 0.005
        self.R = 1000;          self.rou = 2
        self.p = 0.01;          self.block_size = [[100] * self.model.width for i in range(self.model.height)]
        self.w = [[random.randint(20,30)] * self.model.width for i in range(self.model.height)]
        self.T = 600


    def sum(self):
        sum_value = 0
        for i in range(0, self.model.width):
            for k in range(0, self.model.height):
                sum_value += (self.w[i][k] * self.states[i][k])
        return sum_value

    def generate_probability(self, current_pool, new_pool):
        xj,yj = new_pool; xo,yo = current_pool
        # payoff for the potential pool
        if self.states[xj][yj] == 0:
            yj_p1 = 0
            yj_p2 = 0
        else:
            yj_p1 = (self.R + self.rou * self.block_size[xj][yj]) / (self.model.num_agent * self.states[xj][yj])
            yj_p2 = (self.w[xj][yj] * self.states[xj][yj]) / self.sum()
        yj_p3 = math.exp(-self.block_size[xj][yj] * (self.propagation) / self.T)
        yj_p4 = self.p * self.w[xj][yj]
        yjr = yj_p1 * yj_p2 * yj_p3 - yj_p4
        # payoff for the origin pool
        if self.states[xo][yo] == 0:
            yo_p1 = 0
            yo_p2 = 0
        else:
            yo_p1 = (self.R + self.rou * self.block_size[xo][yo]) / (self.model.num_agent * self.states[xo][yo])
            yo_p2 = (self.w[xo][yo] * self.states[xo][yo]) / self.sum()
        yo_p3 = math.exp((-self.block_size[xo][yo] * (self.propagation)) / self.T)
        yo_p4 = self.p * self.w[xo][yo]
        yor = yo_p1 * yo_p2 * yo_p3 - yo_p4
        probability = self.states[xj][yj] * max((yjr - yor), 0)
        return probability

    def update_state(self):
        states = [[0] * self.model.width for i in range(self.model.height)]
        for cell in self.model.grid.coord_iter():
            cell_content, x, y = cell
            agent_count = len(cell_content)
            states[x][y] = agent_count / self.model.num_agent
        self.states = states

    def move(self):
        self.update_state()
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        current_pool = self.pos
        new_pool = self.random.choice(possible_steps)
        SwitchProbability = self.generate_probability(current_pool=current_pool, new_pool=new_pool)

        if current_pool == new_pool:
            pass
        else:
            if random.random() < SwitchProbability:
                self.model.grid.move_agent(self, new_pool)
            else:
                pass

    def step(self):
        self.move()


class PoolSelectionModel(Model):

    def __init__(self, N, width, height):
        super().__init__()
        self.running = True
        self.num_agent = N
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.width = width; self.height = height

        for i in range(self.num_agent):
            a = PoolSelectionAgent(i, self)
            self.schedule.add(a)
            # Add the agents to a random grid cell based on the settings
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()

if __name__ == '__main__':
    width = 10; height = 10
    model = PoolSelectionModel(N=200, width=width, height=height)
    for i in range(20):
        model.step()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    plt.imshow(agent_counts, interpolation='nearest')
    plt.colorbar()
    plt.show()