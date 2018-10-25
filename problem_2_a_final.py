# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:28:14 2018

@author: rajar
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Count time
start_time = time.time()

# Initialize all variables
n = int(50)
m = int(50)

# Initialize arrays for storing epidemic parameters, probabilities
# and number of runs for each probability
probabilities = np.arange(0.0, 1.001, 0.01)
epidemic_size = []
epidemic_length = []
runs = np.arange(1.0, 50.1, 1.0)

for pr in probabilities:

    total_infected_size = 0
    total_infected_length = 0

    for run in runs:

        graph = nx.grid_2d_graph(m, n, periodic=False, create_using=None)
        
        # Append all nodes to the list nodes
        nodes = []
        for i in graph.nodes():
            nodes.append(i)
        
        # Initialize variables
        infected = []
        inf1 = random.randint(0,n-1)
        inf2 = random.randint(0,m-1)
        infected.append((inf1, inf2))
        to_be_infected = [(n-1, m-1)]
        count = 0
        
        # while new_infected is not equal to infected
        # or no new nodes are infected
        while (to_be_infected != []):
            to_be_infected = []
            
            for i in infected:
                #print('Already infected node:',i)
                for j in graph.neighbors(i):
                    #print('Neighbors of infected node' ,i, 'is node' ,j)
                    if (j not in infected):
                        prob = random.uniform(0,1)
                        if (prob <= pr):
                            to_be_infected.append(j)
                            #print('New nodes added')
                        infected = list(set(infected) | set(to_be_infected))
                        
            if (to_be_infected is not None):
                count += 1
        
        total_infected_size += len(infected)
        total_infected_length += count

    # Find number of infected nodes and update array
    epidemic_size.append(total_infected_size/len(runs))
    # Find number of steps and update array
    epidemic_length.append(total_infected_length/len(runs))

    


# Modifications for plotting
epidemic_size_normalized = []

for i in epidemic_size:
    epidemic_size_normalized.append(i/(m*n))
    
for j in range(len(epidemic_size_normalized)):
    if (epidemic_size_normalized[j] >= 0.02):
        p_crit_1 = probabilities[j]
        break
    
for j in range(len(epidemic_size_normalized)):
    if (epidemic_size_normalized[j] >= 0.99):
        p_crit_2 = probabilities[j]
        break
    
for j in range(len(epidemic_length)):
    if (epidemic_length[j] >= 2.00):
        p_crit_3 = probabilities[j]
        break

#Plotting <s> vs. p
plt.scatter(probabilities,epidemic_size_normalized, linestyle='-', color='b', linewidth=1)
plt.xlabel('Probabilities p')
plt.ylabel('Average Epidemic size <s>')
#plt.ylim((0.9,3.1))
yy_1 = np.linspace(0.0, 1.0, num=100)
xx_1 = []
for i in yy_1:
    xx_1.append(p_crit_1)
plt.plot(xx_1,yy_1,linestyle='-', color='k', linewidth=1)
plt.savefig('grid_s_versus_p_p-100_runs-50', dpi = 300)
plt.show()   

#Plotting <l> vs. p
plt.scatter(probabilities,epidemic_length, linestyle='-', color='b', linewidth=1)
plt.xlabel('Probabilities p')
plt.ylabel('Average Epidemic length <l>')
#plt.ylim((0.9,3.1))
yy_2 = np.linspace(0.0, 91.0, num=100)
xx_2 = []
for i in yy_2:
    xx_2.append(p_crit_3)
plt.plot(xx_2,yy_2,linestyle='-', color='k', linewidth=1)
xx_3 = np.linspace(0.0, 1.0, num=100)
yy_3 = []
for i in xx_3:
    yy_3.append(math.log(n*m))
plt.plot(xx_3,yy_3,linestyle='-.', color='k', linewidth=1)
plt.savefig('grid_l_versus_p_p-100_runs-50', dpi = 300)
plt.show() 

# Record elapsed time
elapsed_time = time.time() - start_time

print((elapsed_time/3600), 'hours' )