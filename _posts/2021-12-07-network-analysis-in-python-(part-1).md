---
title: Network Analysis in Python (Part 1)
date: 2021-12-07 11:22:08 +0100
categories: [Machine Learning, Deep Learning, DataCamp]
tags: [DataCamp]
math: true
mermaid: true
---

Network Analysis in Python (Part 1)
======================================







 This is the memo of the 26th course of ‘Data Scientist with Python’ track.


**You can find the original course
 [HERE](https://www.datacamp.com/courses/network-analysis-in-python-part-1)**
 .





---



# **1. Introduction to networks**
--------------------------------




## **1.1 Introduction to networks**



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture6-6.png?w=1024)

####
**Basics of NetworkX API, using Twitter network**



 To get you up and running with the NetworkX API, we will run through some basic functions that let you query a Twitter network that has been pre-loaded for you and is available in the IPython Shell as
 `T`
 . The Twitter network comes from
 [KONECT](http://konect.uni-koblenz.de/)
 , and shows a snapshot of a subset of Twitter users. It is an anonymized Twitter network with metadata.




 You’re now going to use the NetworkX API to explore some basic properties of the network, and are encouraged to experiment with the data in the IPython Shell.




 Wait for the IPython shell to indicate that the graph that has been preloaded under the variable name
 `T`
 (representing a
 **T**
 witter network), and then answer the following question:




 What is the size of the graph
 `T`
 , the type of
 `T.nodes()`
 , and the data structure of the third element of the last edge listed in
 `T.edges(data=True)`
 ?





```

T
<networkx.classes.digraph.DiGraph at 0x7f0c553c89b0>


T.nodes()
NodeView((1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
...
23365, 23366, 23367, 23368, 23369, 23370))


T.nodes(data=True)
NodeDataView({1: {'category': 'I', 'occupation': 'politician'}, 3: {'category': 'D', 'occupation': 'celebrity'}, 4: {'category': 'I', 'occupation': 'politician'}, 5: {'category': 'I', 'occupation': 'scientist'},

...



T.edges(data=True)
OutEdgeDataView([(1, 3, {'date': datetime.date(2012, 11, 16)}), (1, 4, {'date': datetime.date(2013, 6, 7)}), (1, 5, {'date': datetime.date(2009, 7, 27)}),
...
(23324, 23335, {'date': datetime.date(2012, 2, 1)}), (23324, 23336, {'date': datetime.date(2010, 9, 20)})])


len(T)
23369

type(T.nodes())
networkx.classes.reportviews.NodeView

list(T.edges(data=True))[3]
(1, 6, {'date': datetime.date(2014, 12, 18)})

```


####
**Basic drawing of a network using NetworkX**




```python

# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph to screen
nx.draw(T_sub)
plt.show()


T_sub
# <networkx.classes.digraph.DiGraph at 0x7f0c4988bd68>

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture7-4.png?w=1024)

####
**Queries on a graph**



 Now that you know some basic properties of the graph and have practiced using NetworkX’s drawing facilities to visualize components of it, it’s time to explore how you can query it for nodes and edges. Specifically, you’re going to look for “nodes of interest” and “edges of interest”.





```python

# Use a list comprehension to get the nodes of interest: noi
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']

# Use a list comprehension to get the edges of interest: eoi
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]

```




```

noi[:3]
[5, 9, 13]

eoi[:3]
[(1, 5), (1, 9), (1, 13)]

```




---


## **1.2 Types of graphs**


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture8-3.png?w=822)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture9-2.png?w=824)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture10-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture11-3.png?w=784)


####
**Checking the un/directed status of a graph**



 Which type of graph do you think the Twitter network data you have been working with corresponds to?





```

type(T)
networkx.classes.digraph.DiGraph

```


####
**Specifying a weight on edges**



 Weights can be added to edges in a graph, typically indicating the “strength” of an edge. In NetworkX, the weight is indicated by the
 `'weight'`
 key in the metadata dictionary.




 Before attempting the exercise, use the IPython Shell to access the dictionary metadata of
 `T`
 and explore it, for instance by running the commands
 `T.edges[1, 10]`
 and then
 `T.edges[10, 1]`
 . Note how there’s only one field, and now you’re going to add another field, called
 `'weight'`
 .





```

T.edges[1, 10]
# {'date': datetime.date(2012, 9, 8)}

T.edges[10, 1]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    T.edges[10, 1]
  File "<stdin>", line 927, in __getitem__
    return self._adjdict[u][v]
KeyError: 1

```




```python

# Set the weight of the edge
T.edges[1, 10]['weight'] = 2

# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data=True):

    # Check if node 293 is involved
    if 293 in [u, v]:

        # Set the weight to 1.1
        T.edges[u, v]['weight'] = 1.1

T.edges[1, 10]
# {'date': datetime.date(2012, 9, 8), 'weight': 2}

```


####
**Checking whether there are self-loops in the graph**



 NetworkX also allows edges that begin and end on the same node; while this would be non-intuitive for a social network graph, it is useful to model data such as trip networks, in which individuals begin at one location and end in another.




 In this exercise as well as later ones, you’ll find the
 `assert`
 statement useful. An
 `assert`
 -ions checks whether the statement placed after it evaluates to True, otherwise it will throw an
 `AssertionError`
 .




 To begin, use the
 `.number_of_selfloops()`
 method on
 `T`
 in the IPython Shell to get the number of edges that begin and end on the same node. A number of self-loops have been synthetically added to the graph. Your job in this exercise is to write a function that returns these edges.





```

T.number_of_selfloops()
42

```




```python

# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []

    # Iterate over all the edges of G
    for u, v in G.edges():

    # Check if node u and node v are the same
        if u == v:

            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)

    return nodes_in_selfloops

# Check whether number of self loops equals the number of nodes in self loops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))

```




---


## **1.3 Network visualization**



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture12-3.png?w=749)

![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture13-3.png?w=852)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture14-3.png?w=861)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture15-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture16-3.png?w=968)


####
**Visualizing using Matrix plots**



 It is time to try your first “fancy” graph visualization method: a matrix plot. To do this,
 `nxviz`
 provides a
 `MatrixPlot`
 object.




`nxviz`
 is a package for visualizing graphs in a rational fashion. Under the hood, the
 `MatrixPlot`
 utilizes
 `nx.to_numpy_matrix(G)`
 , which returns the matrix form of the graph. Here, each node is one column and one row, and an edge between the two nodes is indicated by the value 1. In doing so, however, only the
 `weight`
 metadata is preserved; all other metadata is lost, as you’ll verify using an
 `assert`
 statement.




 A corresponding
 `nx.from_numpy_matrix(A)`
 allows one to quickly create a graph from a NumPy matrix. The default graph type is
 `Graph()`
 ; if you want to make it a
 `DiGraph()`
 , that has to be specified using the
 `create_using`
 keyword argument, e.g. (
 `nx.from_numpy_matrix(A, create_using=nx.DiGraph)`
 ).





```

import matplotlib.pyplot as plt
import networkx as nx
# Import nxviz
import nxviz as nv

# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)

# Draw m to the screen
m.draw()

# Display the plot
plt.show()

# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)

# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()

```




```

m
<nxviz.plots.MatrixPlot at 0x7fdc9e02cf28>

A
matrix([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        ...,
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.]])

T_conv
<networkx.classes.digraph.DiGraph at 0x7fdc9cdfc438>

T_conv.nodes(data=True)
NodeDataView({0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {},
...

T_conv.edges(data=True)
OutEdgeDataView([(4, 73, {'weight': 1.0}), (5, 125, {'weight': 1.0}), (6, 7, {'weight': 1.0}),
...


T.nodes(data=True)
NodeDataView({27: {'category': 'D', 'occupation': 'scientist'}, 35: {'category': 'P', 'occupation': 'scientist'},
...

T.edges(data=True)
OutEdgeDataView([(151, 5071, {'date': datetime.date(2011, 2, 21)}), (180, 12678, {'date': datetime.date(2013, 6, 7)}),
...

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture17-1.png?w=1024)

####
**Visualizing using Circos plots**



 Circos plots are a rational, non-cluttered way of visualizing graph data, in which nodes are ordered around the circumference in some fashion, and the edges are drawn within the circle that results, giving a beautiful as well as informative visualization about the structure of the network.




 In this exercise, you’ll continue getting practice with the nxviz API, this time with the
 `CircosPlot`
 object.





```python

# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot

# Create the CircosPlot object: c
c = CircosPlot(T)

# Draw c to the screen
c.draw()

# Display the plot
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture18-1.png?w=1024)

####
**Visualizing using Arc plots**



 Following on what you’ve learned about the nxviz API, now try making an ArcPlot of the network. Two keyword arguments that you will try here are
 `node_order='keyX'`
 and
 `node_color='keyX'`
 , in which you specify a key in the node metadata dictionary to color and order the nodes by.





```python

# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import ArcPlot

# Create the un-customized ArcPlot object: a
a = ArcPlot(T)

# Draw a to the screen
a.draw()

# Display the plot
plt.show()

# Create the customized ArcPlot object: a2
a2 = ArcPlot(T, node_order='category', node_color='category')

# Draw a2 to the screen
a2.draw()

# Display the plot
plt.show()


```


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture19-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture20-2.png?w=1024)



 Notice the node coloring in the customized ArcPlot compared to the uncustomized version. In the customized ArcPlot, the nodes in each of the categories –
 `'I'`
 ,
 `'D'`
 , and
 `'P'`
 – have their own color.





---



# **2. Important nodes**
-----------------------


## **2.1 Degree centrality**


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture21-1.png?w=622)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture22-1.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture24-1.png?w=959)


####
**Compute number of neighbors for each node**



 How do you evaluate whether a node is an important one or not? There are a few ways to do so, and here, you’re going to look at one metric: the number of neighbors that a node has.




 Your job in this exercise is to write a function that returns all nodes that have
 `m`
 neighbors.





```

T.neighbors(19)
<dict_keyiterator at 0x7f301617f458>

list(T.neighbors(19))
[5,
 8,
 12035,

...

 37,
 5485,
 48]

```




```python

# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()

    # Iterate over all nodes in G
    for n in G.nodes():

        # Check if the number of neighbors of n matches m
        if len(list(G.neighbors(n))) == m:

            # Add the node n to the set
            nodes.add(n)

    # Return the nodes with m neighbors
    return nodes


# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)

{22533, 1803, 11276, 11279, 6161, 4261, 10149, 3880, 16681, 5420, 14898, 64, 14539, 6862, 20430, 9689, 475, 1374, 6112, 9186, 17762, 14956, 2927, 11764, 4725}

```



 Great work! The number of neighbors a node has is one way to identify important nodes. It looks like 25 nodes in graph
 `T`
 have 6 neighbors.



####
**Compute degree distribution**



 The number of neighbors that a node has is called its “degree”, and it’s possible to compute the degree distribution across the entire graph. In this exercise, your job is to compute the degree distribution across
 `T`
 .





```python

# Compute the degree of every node: degrees
degrees = [len(list(T.neighbors(n))) for n in T.nodes()]

# Print the degrees
print(degrees)

[47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 10, 27, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 60, 0, 11, 4, 0, 12, 0, 0, 56, 53, 0, 30, 0, 0, 0, 0, 12, 0, 0
...

```


####
**Degree centrality distribution**



 The degree of a node is the number of neighbors that it has. The degree centrality is the number of neighbors divided by all possible neighbors that it could have. Depending on whether self-loops are allowed, the set of possible neighbors a node could have could also include the node itself.





```python

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)

# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.xlabel('centrality')
plt.ylabel('count')
plt.show()

# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist(degrees)
plt.xlabel('neighbors')
plt.ylabel('count')
plt.show()

# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.xlabel('degrees')
plt.ylabel('centrality')
plt.show()

```


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture1-5.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture2-6.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture3-7.png?w=1024)




---


## **2.2 Graph algorithms**



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture4-7.png?w=1019)

![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture5-7.png?w=993)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture6-7.png?w=813)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture7-5.png?w=798)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture8-4.png?w=794)


####
**Shortest Path I**



 You can leverage what you know about finding neighbors to try finding paths in a network. One algorithm for path-finding between two nodes is the “breadth-first search” (BFS) algorithm. In a BFS algorithm, you start from a particular node and iteratively search through its neighbors and neighbors’ neighbors until you find the destination node.




 Pathfinding algorithms are important because they provide another way of assessing node importance; you’ll see this in a later exercise.




 In this set of 3 exercises, you’re going to build up slowly to get to the final BFS algorithm. The problem has been broken into 3 parts that, if you complete in succession, will get you to a first pass implementation of the BFS algorithm.





```python

# Define path_exists()
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()

    # Initialize the queue of nodes to visit with the first node: queue
    queue = [node1]

    # Iterate over the nodes in the queue
    for node in queue:

        # Get neighbors of the node
        neighbors = G.neighbors(node)

        # Check to see if the destination node is in the set of neighbors
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break


```


####
**Shortest Path II**



 Now that you’ve got the code for checking whether the destination node is present in neighbors, next up, you’re going to extend the same function to write the code for the condition where the destination node is
 **not**
 present in the neighbors.





```

def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]

    for node in queue:
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True

        else:
            # Add current node to visited nodes
            visited_nodes.add(node)

            # Add neighbors of current node that have not yet been visited
            queue.extend([n for n in neighbors if n not in visited_nodes])

```


####
**Shortest Path III**



 This is the final exercise of this trio! You’re now going to complete the problem by writing the code that returns
 `False`
 if there’s no path between two nodes.





```

def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]

    for node in queue:
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])

        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))

            # Place the appropriate return statement
            return False

```



 You’ve just written an implementation of the BFS algorithm!





---


## **2.3 Betweenness centrality**


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture9-3.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture10-5.png?w=878)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture11-4.png?w=1024)


####
**NetworkX betweenness centrality on a social network**



 Betweenness centrality is a node importance metric that uses information about the shortest paths in a network. It is defined as the fraction of all possible shortest paths between any pair of nodes that pass through the node.





```python

# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)

# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)

# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))
plt.xlabel('betweenness_centrality')
plt.ylabel('degree_centrality')

# Display the plot
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture12-4.png?w=1024)


 Now that you know how to compute different metrics for node importance, you’re going to take a deep dive into the Twitter network.



####
**Deep dive – Twitter network**



 You’re going to now take a deep dive into a Twitter network, which will help reinforce what you’ve learned earlier.




 First, you’re going to find the nodes that can broadcast messages very efficiently to lots of people one degree of separation away.





```python

# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)

    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))

    nodes = set()

    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():

        # Check if the current value has the maximum degree centrality
        if v == max_dc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes

# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)
print(top_dc)
# {11824}

# Write the assertion statement
for node in top_dc:
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())

```



 It looks like node 11824 has the highest degree centrality.



####
**Deep dive – Twitter network part II**



 Next, you’re going to do an analogous deep dive on betweenness centrality!





```python

# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):

    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(G)

    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))

    nodes = set()

    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():

        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes

# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc = find_node_with_highest_bet_cent(T)
print(top_bc)
# {1}

# Write an assertion statement that checks that the node(s) is/are correctly identified.
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())

```



 You have correctly identified that node 1 has the highest betweenness centrality!





---



# **3. Structures**
------------------


## **3.1 Cliques & communities**



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture13-4.png?w=922)

####
**Identifying triangle relationships**



 Now that you’ve learned about cliques, it’s time to try leveraging what you know to find structures in a network. Triangles are what you’ll go for first. We may be interested in triangles because they’re the simplest complex clique. Let’s write a few functions; these exercises will bring you through the fundamental logic behind network algorithms.




 In the Twitter network, each node has an
 `'occupation'`
 label associated with it, in which the Twitter user’s work occupation is divided into
 `celebrity`
 ,
 `politician`
 and
 `scientist`
 . One potential application of triangle-finding algorithms is to find out whether users that have similar occupations are more likely to be in a clique with one another.





```python

# check if node 3 is connect with node 4
T.has_edge(3, 4)
False

T.has_edge(1, 3)
True

# create a iterator contains all combinations of 2 neighbors of node 1.
combinations(T.neighbors(1), 2)
<itertools.combinations at 0x7f8450bebe58>

[x for x in combinations(T.neighbors(1), 2)]
[(3, 4),
 (3, 5),
 (3, 6),
...

```




```

from itertools import combinations

# Define is_in_triangle()
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not.

    Returns a boolean.
    """
    in_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check if an edge exists between n1 and n2
        if G.has_edge(n1, n2):
            in_triangle = True
            break
    return in_triangle

```


####
**Finding nodes involved in triangles**



 NetworkX provides an API for counting the number of triangles that every node is involved in:
 `nx.triangles(G)`
 . It returns a dictionary of nodes as the keys and number of triangles as the values.


 Your job in this exercise is to modify the function defined earlier to extract all of the nodes involved in a triangle relationship with a given node.





```

from itertools import combinations

# Write a function that identifies all nodes in a triangle relationship with a given node.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1, n2):

            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)

            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)

    return triangle_nodes

# Write the assertion statement
assert len(nodes_in_triangle(T, 1)) == 35

```



 Your function correctly identified that node 1 is in a triangle relationship with 35 other nodes.



####
**Finding open triangles**



 Let us now move on to finding open triangles! Recall that they form the basis of friend recommendation systems; if “A” knows “B” and “A” knows “C”, then it’s probable that “B” also knows “C”.





```

from itertools import combinations

# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):

            in_open_triangle = True

            break

    return in_open_triangle

# Compute the number of open triangles in T
num_open_triangles = 0

# Iterate over all the nodes in T
for n in T.nodes():

    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):

        # Increment num_open_triangles
        num_open_triangles += 1

print(num_open_triangles)

# 22

```



 It looks like 22 nodes in graph
 `T`
 are in open triangles!





---


## **3.2 Maximal cliques**


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture15-4.png?w=1024)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture16-4.png?w=1024)


####
**Finding all maximal cliques of size “n”**



 Now that you’ve explored triangles (and open triangles), let’s move on to the concept of maximal cliques. Maximal cliques are cliques that cannot be extended by adding an adjacent edge, and are a useful property of the graph when finding communities. NetworkX provides a function that allows you to identify the nodes involved in each maximal clique in a graph:
 `nx.find_cliques(G)`
 .





```python

# Define maximal_cliques()
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs

# Check that there are 33 maximal cliques of size 3 in the graph T
assert len(maximal_cliques(T, 3)) == 33

```




```

nx.find_cliques(T)
<generator object find_cliques at 0x7f3fb875f938>

maximal_cliques(T, 3)
[[1, 13, 19],
 [1, 16, 48],
 [1, 19, 8],
...

```




---


## **3.3 Subgraphs**


####
**Subgraphs I**



 There may be times when you just want to analyze a subset of nodes in a network. To do so, you can copy them out into another graph object using
 `G.subgraph(nodes)`
 , which returns a new
 `graph`
 object (of the same type as the original graph) that is comprised of the iterable of
 `nodes`
 that was passed in.





```

nodes_of_interest = [29, 38, 42]

# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []

    # Iterate over the nodes of interest
    for n in nodes_of_interest:

        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)

        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):

            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)

    return G.subgraph(nodes_to_draw)

# Extract the subgraph with the nodes of interest: T_draw
T_draw = get_nodes_and_nbrs(T, nodes_of_interest)

# Draw the subgraph to the screen
nx.draw(T_draw)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture18-2.png?w=626)


 The subgraph consisting of the nodes of interest and their neighbors has 7 nodes.



####
**Subgraphs II**



 Using a list comprehension, extract nodes that have the metadata
 `'occupation'`
 as
 `'celebrity'`
 alongside their neighbors:





```python

# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']

# Create the set of nodes: nodeset
nodeset = set(nodes)

# Iterate over nodes
for n in nodes:

    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)

    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs)

# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture19-2.png?w=617)


 You’re now ready to bring together all of the concepts you’ve learned and apply them to a case study!





---



# **4. Bringing it all together**
--------------------------------


## **4.1 Case study**



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture20-3.png?w=905)

![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture21-2.png?w=737)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture22-2.png?w=622)


####
**Characterizing the network (I)**



 To start out, let’s do some basic characterization of the network, by looking at the number of nodes and number of edges in a network.





```

G.nodes(data=True)
NodeDataView({'u41': {'bipartite': 'users', 'grouping': 0}, 'u69': {'bipartite': 'users', 'grouping': 0},
...
, 'u14964': {'bipartite': 'users', 'grouping': 1}})


G.edges(data=True)
EdgeDataView([('u41', 'u2022', {}), ('u41', 'u69', {}), ('u41', 'u5082', {}), ('u41', 'u298', {}),
...
 ('u9866', 'u10603', {}), ('u9866', 'u10340', {}), ('u9997', 'u10500', {}), ('u10340', 'u10603', {})])

```




```

len(G.nodes())
56519

len(G.edges())
72900

```


####
**Characterizing the network (II)**



 Let’s continue recalling what you’ve learned before about node importances, by plotting the degree distribution of a network. This is the distribution of node degrees computed across all nodes in a network.





```python

# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Plot the degree distribution of the GitHub collaboration network
plt.hist(list(nx.degree_centrality(G).values()))
plt.xlabel('degree_centrality')
plt.ylabel('count')
plt.show()


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture23-1.png?w=1024)

####
**Characterizing the network (III)**



 The last exercise was on degree centrality; this time round, let’s recall betweenness centrality!





```python

# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Plot the degree distribution of the GitHub collaboration network
plt.hist(list(nx.betweenness_centrality(G).values()))
plt.xlabel('betweenness_centrality')
plt.ylabel('count')
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture24-2.png?w=1024)



---


## **4.2 Case study part II: Visualization**



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture26-1.png?w=1024)

####
**MatrixPlot**



 Let’s now practice making some visualizations. The first one will be the MatrixPlot. In a MatrixPlot, the matrix is the representation of the edges.





```python

# Import necessary modules
from nxviz import MatrixPlot
import matplotlib.pyplot as plt

# Calculate the largest connected component subgraph: largest_ccs
largest_ccs = sorted(nx.connected_component_subgraphs(G), key=lambda x: len(x))[-1]

# Create the customized MatrixPlot object: h
h = MatrixPlot(graph=largest_ccs, node_grouping='grouping')

# Draw the MatrixPlot to the screen
h.draw()
plt.show()

```




```

nx.connected_component_subgraphs(G)
<generator object connected_component_subgraphs at 0x7f758c7e4eb8>


list(nx.connected_component_subgraphs(G))
[<networkx.classes.graph.Graph at 0x7f758c81bfd0>,
 <networkx.classes.graph.Graph at 0x7f758c81bef0>]

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture27.png?w=1024)


 Recall that in a MatrixPlot, nodes are the rows and columns of the matrix, and cells are filled in according to whether an edge exists between the pairs of nodes.



####
**ArcPlot**




```python

# Import necessary modules
from nxviz.plots import ArcPlot
import matplotlib.pyplot as plt

# Iterate over all the nodes in G, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)

# Create the ArcPlot object: a
a = ArcPlot(graph=G, node_order='degree')

# Draw the ArcPlot to the screen
a.draw()
plt.show()

```




```

G.node['u41']
{'bipartite': 'users', 'grouping': 0}

# get the value of degree
nx.degree(G, 'u41')
5

# set the value 'degree' to 5
G.node['u41']['degree'] = nx.degree(G, 'u41')

# show the value
G.node['u41']
{'bipartite': 'users', 'degree': 5, 'grouping': 0}

G.node['u41']['degree']
5


```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture28.png?w=1024)

####
**CircosPlot**




```python

# Import necessary modules
from nxviz import CircosPlot
import matplotlib.pyplot as plt

# Iterate over all the nodes, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)

# Create the CircosPlot object: c
c = CircosPlot(graph=G, node_order='degree', node_grouping='grouping', node_color='grouping')

# Draw the CircosPlot object to the screen
c.draw()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture29.png?w=1024)


 This CircosPlot provides a compact alternative to the ArcPlot. It is easy to see in this plot that most users belong to one group.





---


## **4.3 Case study part III: Cliques**


![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture30.png?w=680)
![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture31.png?w=807)


####
**Finding cliques (I)**



 You’re now going to practice finding cliques in
 `G`
 . Recall that cliques are “groups of nodes that are fully connected to one another”, while a maximal clique is a clique that cannot be extended by adding another node in the graph.





```python

# Calculate the maximal cliques in G: cliques
cliques = nx.find_cliques(G)

# Count and print the number of maximal cliques in G
print(len(list(cliques)))
# 19


list(cliques)
[ ['u4761', 'u2643', 'u4329', 'u1254', 'u2737', 'u2289'],
 ['u2022', 'u9866', 'u435', 'u10340', 'u7623', 'u322', 'u8135', 'u10603'],
...
 ['u655', 'u2906'],
 ['u655', 'u914']]

```


####
**Finding cliques (II)**



 Let’s continue by finding a particular maximal clique, and then plotting that clique.





```python

# Import necessary modules
import networkx as nx
from nxviz import CircosPlot
import matplotlib.pyplot as plt

# Find the author(s) that are part of the largest maximal clique: largest_clique
largest_clique = sorted(nx.find_cliques(G), key=lambda x:len(x))[-1]

# Create the subgraph of the largest_clique: G_lc
G_lc = G.subgraph(largest_clique)

# Create the CircosPlot object: c
c = CircosPlot(G_lc)

# Draw the CircosPlot to the screen
c.draw()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture32.png?w=1024)


 The subgraph consisting of the largest maximal clique has 14 users.





---


## **4.4 Case study part IV: Final tasks**


####
**Finding important collaborators**



 You’ll now look at important nodes once more. Here, you’ll make use of the
 `degree_centrality()`
 and
 `betweenness_centrality()`
 functions in NetworkX to compute each of the respective centrality scores, and then use that information to find the “important nodes”. In other words, your job in this exercise is to find the user(s) that have collaborated with the most number of users.





```python

# Compute the degree centralities of G: deg_cent
deg_cent = nx.degree_centrality(G)

# Compute the maximum degree centrality: max_dc
max_dc = max(deg_cent.values())

# Find the user(s) that have collaborated the most: prolific_collaborators
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc]

# Print the most prolific collaborator(s)
print(prolific_collaborators)
# ['u741']

```



 It looks like
 `'u741'`
 is the most prolific collaborator.



####
**Characterizing editing communities**



 You’re now going to combine what you’ve learned about the BFS algorithm and concept of maximal cliques to visualize the network with an ArcPlot.





```python

# Import necessary modules
from nxviz import ArcPlot
import matplotlib.pyplot as plt

# Identify the largest maximal clique: largest_max_clique
largest_max_clique = set(sorted(nx.find_cliques(G), key=lambda x: len(x))[-1])

# Create a subgraph from the largest_max_clique: G_lmc
G_lmc = G.subgraph(largest_max_clique).copy()

# Go out 1 degree of separation
for node in list(G_lmc.nodes()):
    G_lmc.add_nodes_from(G.neighbors(node))
    G_lmc.add_edges_from(zip([node]*len(list(G.neighbors(node))), G.neighbors(node)))

# Record each node's degree centrality score
for n in G_lmc.nodes():
    G_lmc.node[n]['degree centrality'] = nx.degree_centrality(G_lmc)[n]

# Create the ArcPlot object: a
a = ArcPlot(graph=G_lmc, node_order='degree centrality')

# Draw the ArcPlot to the screen
a.draw()
plt.show()

```



![Desktop View]({{ site.baseurl }}/assets/datacamp/network-analysis-in-python-(part-1)/capture34.png?w=1024)

####
**Recommending co-editors who have yet to edit together**



 Finally, you’re going to leverage the concept of open triangles to recommend users on GitHub to collaborate!





```python

# Import necessary modules
from itertools import combinations
from collections import defaultdict

# Initialize the defaultdict: recommended
recommended = defaultdict(int)

# Iterate over all the nodes in G
for n, d in G.nodes(data=True):

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check whether n1 and n2 do not have an edge
        if not G.has_edge(n1, n2):

            # Increment recommended
            recommended[(n1, n2)] += 1

# Identify the top 10 pairs of users
all_counts = sorted(recommended.values())
top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
print(top10_pairs)


```




```

recommended
defaultdict(int,
            {('u10090', 'u2737'): 1,
             ('u10090', 'u3243'): 1,
             ('u10090', 'u3658'): 1,
             ('u10090', 'u4329'): 1,
...

```



 You’ve identified pairs of users who should collaborate together, and in doing so, built your very own recommendation system!




 The End.


 Thank you for reading.



