# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:52:27 2018

@author: rajar
"""

import networkx as nx
import numpy as np
import random

n1 = int(4)
m1 = int(4)

#graph = nx.grid_2d_graph(m1, n1, periodic=False, create_using=None)
#nx.draw(graph)


# Define long range edge probability
long_range_prob = np.arange(1.0, 1.001, 1.0)

for q in long_range_prob:
    # Genenerate a graph
    graph = nx.grid_2d_graph(n1, m1, periodic=False, create_using=None)
    count = 0
    while (count != 1):
        # generate four random numbers between [0,m-1] and [0,n-1]
        rg_x1 = random.randint(0,n1-1)
        rg_y1 = random.randint(0,m1-1)
        rg_x2 = random.randint(0,n1-1)
        rg_y2 = random.randint(0,m1-1)
        # Check if a self-loop is being added
        if (rg_x1 != rg_x2 and rg_y1 != rg_y2):
            # Check if an edge exists between these two numbers
            if (graph.has_edge((rg_x1, rg_y1),(rg_x2, rg_y2)) == False):
                count = 1
                prob = random.uniform(0,1)
                if (prob <= q):
                    graph.add_edge((rg_x1, rg_y1),(rg_x2, rg_y2))
                    print('Edge', '(' , rg_x1 , ',' , rg_y1, ')' ,
                          '(' , rg_x2 , ',' , rg_y2, ')', 'added')
                    nx.draw(graph)

#nodes = []
#for i in graph.nodes():
#    print(i)
#    nodes.append(i)
    
edges = []
for j in graph.edges():
    #print(j)
    edges.append(j)
print(len(edges))    
#print(graph.has_edge((9,5),(9,6)))