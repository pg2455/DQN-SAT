#! /usr/bin/env python2

from networkx.generators import random_graphs

class GenerateRandomGraphs(object):
    """
    Generate random graphs using various available methods

    Parameters:
    ______________
        None
    """

    def __init__(self):
        pass

    def get_barabasi_graph(self,n,m):
        """
        Generate a graph based on Barabasi-Albert model

        Parameters:
        ______________
            n: int
                number of nodes
            m: int
                number of edges to attach from a new node to existing nodes
        """
        return random_graphs.barabasi_albert_graph(n,m)

    def get_erdos_renyi_graph(self, n, p):
        """
        Generate a graph based on Erdos-Renyi model

        Parameters:
        ______________
            n: int
                number of nodes
            p: float
                probability of an edge between two nodes
        """
        return random_graphs.erdos_renyi_graph(n,p)
