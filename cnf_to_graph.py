#! /usr/bin/env python2
import networkx
from cnf_compiler import CNFCompiler

class CNFTOGRAPH(object):
    """
    Convert a given CNF to a graph structure

    Parameters:
    ______________
        None
    """

    def __init__(self):
        """
        instantiate the objects
        """
        self.cnf_compiler = CNFCompiler()

    def get_cvig(self, cnf_dimacs_file):
        """
        Convert a given CNF to the graph of the form of CVIG(Clause Variable Incidence Graph)
        Its an undirected bipartite graphical representation.

        Parameters:
        ______________
            cnf_dimacs_file: str
                File path of the DIMACS file containing CNF

        Returns:
        ______________
            graph: networkx.classes.graph.Graph
                Corresponding graph with vertices as cx (for the node as a clause), vx(for the node as a variable),
                and edge as (cx,vx, properties) with properties as a dictionary.
                properties can be extended as per the use case.
        """
        cnf = self.cnf_compiler.get_cnf_from_dimacs(cnf_dimacs_file)
        text = open(cnf_dimacs_file).readline().split()
        n_variables, n_clauses = map(int, text[2:])

        graph = networkx.Graph()
        graph.add_nodes_from(map(lambda x: ('v' + str(x+1), {'included_in_state': False}), range(n_variables)))
        graph.add_nodes_from(map(lambda x: ('c' + str(x+1), {'included_in_state': False}), range(n_clauses)))

        for c,clause in enumerate(cnf):
            graph.add_edges_from(map(lambda x: ('c'+str(c+1),'v'+str(abs(x)), {'literal_type':sign(x), 'n_variables':len(clause)}), clause))

        return graph


def sign(x):
    return int(x/abs(x))
