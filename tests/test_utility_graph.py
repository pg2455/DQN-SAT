#! /usr/bin/env python2

import sys
sys.path.append('../')
from cnf_compiler import CNFCompiler
from utility_graph import CVIG
from pdb import set_trace as bp
from templates import *

def test_cnf_compiler():
    dimacs_filepath = "/Users/pgupta/Workspace/SATConversion/explore/graph_sat/data/barabasi_n4_m1.dimacs"
    clauses, n_variables, n_clauses = cnf_compiler.get_cnf_from_dimacs(dimacs_filepath)
    print "Success: CNF read from the file - n_variables: {}, n_clauses: {}".format(n_variables, n_clauses)
    return clauses, n_variables, n_clauses

def test_graph_initialization(n_variables, n_clauses, clauses):
    graph = CVIG(n_variables, n_clauses, clauses)
    print "Success: CVIG formed from CNF. ",
    print " Number of nodes in the graph: {}".format(len(graph.nodes))
    return graph

def test_node_count_reset(graph):
    print "Node count before deleting graph : {}".format(Node.count)
    Node.reset()
    del graph
    print "Node count after deleting graph :{} ".format(Node.count)

def test_2nd_graph_nodes():
    graph = CVIG(n_variables, n_clauses, clauses)
    print " Number of nodes in the graph: {}".format(len(graph.nodes))

def test_features():
    dimacs_filepath = "/Users/pgupta/Workspace/SATConversion/explore/graph_sat/data/toy_cnf.dimacs"
    clauses, n_variables, n_clauses = cnf_compiler.get_cnf_from_dimacs(dimacs_filepath)
    graph = CVIG(n_variables, n_clauses, clauses)
    print "node features: {}".format(graph.get_node_feature_matrix())
    print "adjacency matrix: {}".format(graph.get_adjacency_matrix())
    print "edge features: {}".format(graph.get_edge_feature_matrix())

    print "assigning 1,-2"
    graph.update_node_assignment([1,-2])
    print "assigned_decision_variable indices: {}".format(graph.get_assigned_decision_variable_indices())
    print "node features: {}".format(graph.get_node_feature_matrix())
    bp()

if __name__ == "__main__":
    cnf_compiler = CNFCompiler()

    clauses, n_variables, n_clauses  = test_cnf_compiler()

    graph1 = test_graph_initialization(n_variables, n_clauses, clauses)

    test_node_count_reset(graph1)

    graph2 = test_graph_initialization(n_variables, n_clauses, clauses)

    test_features()
