import numpy as np
from mesa_based_118_withchart import *
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

grid = CanvasGrid(agent_portrayal, 2, 2, 500, 500)

chart = ChartModule([{"Label": "1", "Color": "Black"}, {"Label": "2", "Color": "Blue"},
                     {"Label": "3", "Color": "Red"}, {"Label": "4", "Color": "Pink"}],

                    data_collector_name='datacollector')

server = ModularServer(PoolSelectionModel,
                       [grid, chart],
                       "118",
                       {"N":20, "width":2, "height":2})

server.port = 8521 # The default
server.launch()