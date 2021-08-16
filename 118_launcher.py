import numpy as np
from mesa_based_118_test import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "pink",
                 "r": 0.5}
    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)


server = ModularServer(PoolSelectionModel,
                       [grid],
                       "118",
                       {"N":100, "width":10, "height":10})

server.port = 8521 # The default
server.launch()