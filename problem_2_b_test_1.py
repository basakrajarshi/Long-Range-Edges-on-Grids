# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:33 2018

@author: rajar
"""
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D

# Count time
start_time = time.time()

# Initialize all variables
n = int(50)
m = int(50)

array_long_range_probablitiy = []
array_transmission_probability = []
array_epi_size = []
array_epi_len = []

long_range_prob = np.arange(0.0, 1.001, 0.1)

for q in long_range_prob:

    # Initialize arrays for storing epidemic parameters, probabilities
    # and number of runs for each probability
    probabilities = np.arange(0.0, 1.001, 0.1)
    epidemic_size = []
    epidemic_length = []
    runs = np.arange(1.0, 20.01, 1.0)

    #graph = nx.grid_2d_graph(m, n, periodic=False, create_using=None)
    #nx.draw(graph)

    for pr in probabilities:

        total_infected_size = 0
        total_infected_length = 0

        for run in runs:

            graph = nx.grid_2d_graph(m, n, periodic=False, create_using=None)
            #graph = nx.planted_partition_graph(q, k, p_in, p_out, seed=42
            #                                   , directed=False)
            #nx.draw(graph)

            # Adding a random long range edge
            count = 0
            while (count != 1):
                # generate four random numbers between [0,m-1] and [0,n-1]
                rg_x1 = random.randint(0,n-1)
                rg_y1 = random.randint(0,m-1)
                rg_x2 = random.randint(0,n-1)
                rg_y2 = random.randint(0,m-1)
                # Check if a self-loop is being added
                if (rg_x1 != rg_x2 and rg_y1 != rg_y2):
                    # Check if an edge exists between these two numbers
                    if (graph.has_edge((rg_x1, rg_y1),
                                       (rg_x2, rg_y2)) == False):
                        count = 1
                        prob = random.uniform(0,1)
                        if (prob <= q):
                            graph.add_edge((rg_x1, rg_y1),
                                           (rg_x2, rg_y2))

            # Append all nodes to the list nodes
            nodes = []
            for i in graph.nodes():
                nodes.append(i)
            
            #print(random.randint(0,n))
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
                # Generate a random number between 0 and 1 
                #prob = random.uniform(0,1)
                # Check if the generated number is less than or equal to p
                #if (prob <= p_in and prob <= p_out):
                    # Add all of the current node in 'infected' list's neighbours 
                    # to 'infected' that are not in 'infected'
                    #print(infected)
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
                            #print('The nodes to be infected are',to_be_infected)
                            #print('The     new infected nodes are' ,infected)
                    #to_be_infected = []
                if (to_be_infected is not None):
                    count += 1
            
            total_infected_size += len(infected)
            total_infected_length += count

        # Find number of infected nodes and update array
        #epidemic_size.append(total_infected_size/len(runs))
        # Find number of steps and update array
        #epidemic_length.append(total_infected_length/len(runs))

        array_long_range_probablitiy.append(q) # Array for the x-axis
        array_transmission_probability.append(pr) # Array for the y-axis
        array_epi_size.append(total_infected_size/len(runs)) # Array for the z(1) axis
        array_epi_len.append(total_infected_length/len(runs)) # Array for the z(2) axis



# Modifications for plotting
array_epi_size_norm = []

for i in array_epi_size:
     array_epi_size_norm.append(i/(n*m))


# Generating the 3d Plots
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax.plot(array_epsilon, array_probability, array_epi_size, 
#        label='Epidemic Size')
ax.scatter(array_long_range_probablitiy, array_transmission_probability, array_epi_size_norm)
ax.set_xlabel('Long range edge Prob. q')
ax.set_ylabel('Transmission Prob. p')
ax.set_zlabel('Average Epidemic Size <s>')
ax.legend()
plt.savefig('lre_s_vs_p_vs_q_p-10_q-10_runs-20', dpi = 300)
plt.show()


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax.plot(array_epsilon, array_probability, array_epi_len, 
#        label='Epidemic Length')
ax.scatter(array_long_range_probablitiy, array_transmission_probability, array_epi_len)
ax.set_xlabel('Long range edge Prob. q')
ax.set_ylabel('Transmission Prob. p')
ax.set_zlabel('Average Epidemic Length <l>')
ax.legend()
plt.savefig('lre_l_vs_p_vs_q_p-10_q-10_runs-20', dpi = 300)
plt.show()


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax.plot(array_epsilon, array_probability, array_epi_size, 
#        label='Epidemic Size')
ax.scatter(array_transmission_probability, array_long_range_probablitiy, array_epi_size_norm)
ax.set_xlabel('Transmission Prob. p')
ax.set_ylabel('Long range edge Prob. q')
ax.set_zlabel('Average Epidemic Size <s>')
ax.legend()
plt.savefig('lre_s_vs_q_vs_p_p-10_q-10_runs-20', dpi = 300)
plt.show()


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax.plot(array_epsilon, array_probability, array_epi_len, 
#        label='Epidemic Length')
ax.scatter(array_transmission_probability, array_long_range_probablitiy, array_epi_len)
ax.set_xlabel('Transmission Prob. p')
ax.set_ylabel('Long range edge Prob. q')
ax.set_zlabel('Average Epidemic Length <l>')
ax.legend()
plt.savefig('lre_l_vs_q_vs_p_p-10_q-10_runs-20', dpi = 300)
plt.show()


# Record elapsed time
elapsed_time = time.time() - start_time

print((elapsed_time/3600), 'hours' )