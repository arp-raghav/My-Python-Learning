import csv
import matplotlib.pyplot as plt
from random import sample
import networkx as nx
import pandas as pd
import numpy as np
#Task 1 ( Loading the data)
data_frame = pd.read_csv("C:\\Users\\arpit\\Downloads\\wiki-Vote.txt\\abc.csv")
graph = nx.convert_matrix.from_pandas_edgelist(data_frame, source = 'From Node Id',target = 'To
Node Id')
#Printing the information of Graph
print(nx.info(graph))
#Graph 1
nx.draw(graph, with_labels = True)
fig = plt.figure(figsize = (14, 9))
#Graph 2
nx.draw(graph,
 node_size = 1,
 node_color = 'C1')
fig.set_facecolor('c')
fig = plt.figure(figsize = (14, 9))
colors = np.linspace(0, 1, len(graph.nodes))
#Graph 3
nx.draw(graph, with_labels = True,
 node_size = 30,
 node_color = colors,
 edge_color = 'g')
fig.set_facecolor('k')
colors = np.linspace(0, 1, len(graph.nodes))
k = 1.0
layout = nx.spring_layout(graph, k = k)
fig = plt.figure(figsize = (14, 9))
#Graph 4
nx.draw(graph,with_labels = True,
 node_size = 30,
 node_color = colors,
 pos = layout,
 edge_color = 'c')
fig.set_facecolor('k')
colors = np.linspace(0, 1, len(graph.nodes))
k = 1.0
layout = nx.kamada_kawai_layout(graph)
fig = plt.figure(figsize=(14, 9))
#Graph 5
nx.draw(graph,with_labels = True,
 node_size = 30,
 node_color = colors,
 pos = layout,
 edge_color = 'g')
fig.set_facecolor('k')
colors = np.linspace(0, 1, len(graph.nodes))
k = 1.0
layout = nx.spiral_layout(graph)
fig = plt.figure(figsize = (14, 9))
#Graph 6
nx.draw(graph,with_labels = True,
 node_size = 30,
 node_color = colors,
 pos = layout,
 edge_color = 'g')
fig.set_facecolor('k')
#Plotting the Chart showing In Degree of each user on X axis and Probability on Y axis
m=3
degrees = [val for (node, val) in graph.degree()]
#Printing the In degree of each user
print(sorted(degrees))
degrees = np.array(degrees, dtype= float)
degree_prob = ((degrees/(graph.number_of_nodes()-1)))
#Printing the Probabity of user could have In Degree
print(degree_prob)
#Plotting the chart
plt.figure(figsize=(15, 15))
plt.loglog(degrees[m:], degree_prob[m:],'go-')
plt.xlabel('Degree')
plt.ylabel('Probability P(k)')
plt.show() 
