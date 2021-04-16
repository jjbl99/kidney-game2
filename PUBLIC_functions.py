import importlib
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from math import * 
import networkx as nx


def adj_to_coord(graph, NDDs = np.array([])): 
    """
    Function that converts an adjacency matrix graph to a list of coordinates
    NDDs is the list of pairs that are eligible to receive a graft. 
    An edge is created for each such pair, and added at the end of G_coord. 
    """
    n = graph.shape[0]
    (row,col) = np.where(graph==1)
    G_coord = np.concatenate([row[:,np.newaxis],col[:,np.newaxis]], axis = 1)
    
    k = NDDs.shape[0]
    if k > 0: 
        G_coord = np.concatenate((G_coord, np.concatenate((np.arange(n,n+k)[:,np.newaxis], NDDs[:,np.newaxis]),axis=1)), axis = 0)
    return G_coord 


def check_match(match,G_coord,n,e,k):
    """
    function that returns the chains and cycles in a given match

    chains is a matrix such that chains(i,j) = 0 is edge j is in chain i 
    cycles is a matrix such that cycles(i,j) = 0 is edge j is in chain i 
  
    match is a logical array such that match(i) = 1 if edge i is selected
    G_coord is the coordinate representation of the graph G
    n is the number of pairs in G
    e is the number of edges in G
    k is the number of NDDs
    """
    e = G_coord.shape[0] - k
    N = e+k
    
    ## Exploration of chains
    c_nbr = np.sum(match[e:]) # number of chains in the match
    init_match= match.copy()
    if np.sum(match) != 0: 
        idEdges = np.where(match == 1)[0]
        nG_coord = G_coord[idEdges,:] # coordinates of chosen edges
        
        point_to = np.zeros(n+k, dtype=int)
        point_to[nG_coord[:,0]] = nG_coord[:,1]
        
        in_edge = np.zeros(n+k,dtype=int)
        in_edge[nG_coord[:,0]] = idEdges

    if c_nbr != 0:
        chains = np.zeros((c_nbr,N),dtype=int)
        count = 0
        for i in range(n,n+k): # for each NDD
            if point_to[i] == 0: # if there is no edge, skip i
                continue
            else:
                pos = i
                match[in_edge[pos]] = 0
                while point_to[pos] != 0: # match(point_to(pos,1),1)
                    previous = pos
                    chains[count,in_edge[pos]] = 1
                    pos = point_to[pos]
                    match[in_edge[pos]] = 0
                    point_to[previous] = 0
                count += 1
    else:
        chains = np.array([])
        
    ## Exploration of cycles
    if np.sum(match) > 0:
        cycles = np.zeros((np.sum(match),N),dtype=int)
        if cycles.size != 0:
            count2 = 0
            for i in range(0,n): # for each node
                path = np.zeros(n+k, dtype = int)
                if point_to[i] == 0:# if there is no edge, skip i
                    continue
                else:
                    path[i] = 1
                    pos = i
                    while 1:
                        pos = point_to[pos]
                        match[in_edge[pos]] = 0
                        if path[pos] == 1:
                            break
                        else:
                            path[pos] = 1
                    point_to[path == 1] = 0 # we note that we already visited that edge
                    cycles[count2,in_edge[path == 1]] = 1
                    count2 += 1
        cycles = cycles[:count2,:]
    else: 
        cycles = np.array([])
    return cycles, chains
