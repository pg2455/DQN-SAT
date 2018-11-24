#! /usr/bin/env python2
import sys
sys.path.append('../')
from cnf_to_graph import CNFTOGRAPH

if __name__ == "__main__":
    to_graph = CNFTOGRAPH()
    g = to_graph.get_cvig('/Users/pgupta/Workspace/SATConversion/explore/graph_sat/data/barabasi_n3_m1.dimacs')
    print 'Nodes: ', g.nodes
    print 'Edges: ', g.edges
