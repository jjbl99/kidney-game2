import os
import git
from git import Repo
import numpy as np
import pandas as pd
import csv
import importlib
import PUBLIC_functions
importlib.reload(PUBLIC_functions)
from PUBLIC_functions import *


def init(REPO, P):
    global repo 
    global PATH
    repo = REPO
    PATH = P


# +
def git_add(new_files_PATH):
    """git adds the files in the list new_files_PATH"""
    repo.git.add(new_files_PATH)

def git_push(COMMIT_MESSAGE):
    """function that adds, commits, and pushes any new change"""
    try: 
        repo.git.add(update = True) # automatically adds any file that appeared in the repo
        repo.index.commit(COMMIT_MESSAGE) 
        origin = repo.remote(name='origin') # choose branch
        origin.push()
    except: 
        print('Some error occured while pushing the code')

def git_pull():
    o = repo.remote(name='origin')
    o.pull()


# -

def new_game(p_PATH): 
    """changes the status.txt located at p_PATH to start-game"""
    # repo.git.add(update = True)
    
    file = open(os.path.join(p_PATH,"status.txt"),"w",encoding='utf-8')
    file.truncate(0)
    file.write("new-game")
    file.close()
    
    np.savetxt(os.path.join(p_PATH,'match.csv'), np.array([]), fmt='%i', delimiter=',')
    git_add(os.path.join(p_PATH, 'match.csv'))
    
    git_push('new-game')


def get_market(p_PATH):
    """returns an object market that was extracted from the files located at p_PATH
    """
    par = pd.read_csv(os.path.join(p_PATH,'environment.csv'),delimiter=';',header=None).to_numpy()

    [t],[n],[K],[L],[M] = par[:-1].astype(int)
    
    
    if (par[-1].dtype == float) and (np.isnan(par[-1,0])):
        NDDs = np.array([])
        
    else:
        if type(par[-1,0]) == str:
            NDDs = np.array(par[-1,0].strip().split(','), dtype = int)
        else:
            NDDs = np.array(par[-1])
        
        if NDDs.shape == ():
            NDDs = NDDs[np.newaxis]
    
    # graph = pd.read_csv(os.path.join(p_PATH,'graph.csv'),delimiter=';').to_numpy()
    graph = np.genfromtxt(os.path.join(p_PATH,'graph.csv'),delimiter=',').astype(int)
    
    data = pd.read_csv(os.path.join(p_PATH,'data.csv'),delimiter=';')
    
    return market(t,n,K,L,M,NDDs,graph,data)


# +
# def status(p_PATH): 
#     file = open(player_PATH+'/status.txt',"r") # TRY AND READ DIRECTLY
#     instruction = file.readline()
#     if instruction == "you cannot play": 
#         break
# -

def submit(p_PATH, t, match = np.array([])):
    
    # push match.csv
    np.savetxt(os.path.join(p_PATH,'match.csv'), match, fmt='%i', delimiter=',')
    
    # update status
    file = open(os.path.join(p_PATH,"status.txt"),"w")
    file.truncate(0)
    file.write("player played")
    file.close()
    
    git_push('match added')


def end_game():
    """changes the status.txt located at p_PATH to final-submission"""
    file = open(os.path.join(p_PATH,"status.txt"),"w")
    file.truncate(0)
    file.write("final-submission")
    file.close()
    git_push('final-submission')


class market():
    def __init__(self,t,n,K,L,M,NDDs,graph,data,with_plots = False):
        self.t = t # period
        self.n = n # number of pairs
        
        # self.pairs = self.new_pairs(n0) # PDPs in the market
        self.graph = graph
        self.NDDs = NDDs
        self.k = self.NDDs.shape[0]
        self.K,self.L,self.M = K,L,M
        # self.n_tot = n0
        
        self.data = data
        
        # vector such that pair2index[index of a pair in current pool] = index of that pair in self.data
        self.pair2index = np.arange(0,self.n) # NEED TO IMPORT???????
        
        if with_plots: 
            self.show_graph(self.graph, self.NDDs)
    
    def player_update(self,matches):
        # start a match run 
        G_coord = adj_to_coord(self.graph,self.NDDs)
        e = G_coord.shape[0] - self.k
        
        # G_coord = np.concatenate((G_coord, np.concatenate((np.arange(self.n,self.n+self.k)[:,np.newaxis], self.NDDs[:,np.newaxis]),axis=1)), axis = 0) # we add the NDDs here
        #print('WE HAVE SIZE: ', G_coord.shape, self.k, e)
        cycles, chains = check_match(matches.copy(),G_coord,self.n,e,self.k)
        # print(self.n,self.k,self.NDDs.shape)
        #print('POST SIZE: ', matches.shape)
        print("OK, your result (in matches) is: ", matches.sum())
        
        if cycles.size > 0:
            cy_id = self.pair2index[G_coord[np.sum(cycles,axis=0).astype(bool)][:,1]]
            self.data.loc[cy_id, 'cycle'] = 1
        if chains.size > 0:
            ch_id = self.pair2index[G_coord[np.sum(chains,axis=0).astype(bool)][:,1]]
            self.data.loc[ch_id, 'cycle'] = 0
            # print('this bugs: ', NDD_list.size, pair2index, k)
            self.data.loc[self.pair2index[self.NDDs[matches[e:].astype(bool)]],'started chain'] = 1
            
        if matches.sum() > 0:
            m_id = self.pair2index[G_coord[matches.astype(bool)][:,1]] # gives the indices of pairs who were just matched
            self.data.loc[m_id,'exit date'] = self.t
            self.data.loc[m_id,'last match run'] = self.t
            self.data.loc[m_id,'matched'] = 1
            
            # only keep unmatched pairs
            to_keep = ~np.isin(np.arange(0,self.n), G_coord[matches.astype(bool)][:,1])
            self.pairs = self.pairs[to_keep,:]
            self.graph = self.graph[to_keep,:][:,to_keep]
            self.NDDs = self.NDDs[~(matches[e:].astype(bool))]
            self.NDDs = self.pair2index[self.NDDs] # coordinate in data frame 
            self.pair2index = self.pair2index[to_keep] # update pairs2index
            self.NDDs = np.where(np.isin(self.pair2index,self.NDDs))[0]
            self.k -= np.sum(matches[e:])
            self.n -= np.sum(matches)
                
    def show_graph(self, adjacency_matrix, NDD_list):
        """
        displays a plot of the directed graph whose adjacency matrix is given as input
        code adapted from https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file
        """
        gr = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)
        color_map = np.array(['blue']*(adjacency_matrix.shape[0]))
        if NDD_list.shape[0]!=0:
            color_map[NDD_list] = 'red'
        color_map = color_map.tolist()
        nx.draw_spring(gr, node_color=color_map, with_labels=True, node_size=500) # , labels=mylabels, with_labels=True)
        plt.show()
        
    def show_match(self,G_coord,match):
        gr2 = nx.from_edgelist(G_coord[match.astype(bool)], create_using=nx.DiGraph)
        nx.draw_spring(gr2, with_labels=True, node_size=500)
        plt.show()


