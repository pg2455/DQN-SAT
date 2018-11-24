#! /usr/bin/env python2
from networkx.classes.function import non_edges
from cnf_compiler import CNFCompiler

class HamiltonianCNF(object):
    """
    Generates SAT of the decision problem: does there exist a hamiltonian cycle in a graph?
    Solution is from here: https://www.csie.ntu.edu.tw/~lyuu/complexity/2011/20111018.pdf
    Last set of clauses representing non-edges is modified in this code. Look at the function defintion.
    Variable representation:

             Path position
                 |
                 v
                     Nodes ->     1  2  3
                                  _  _  _
                                1|1  2  3
                                2|4  5  6
                                3|7  8  9

    Parameters:
    ______________
        None
    """


    def __init__(self):
        pass

    def get_cnf(self, graph, cycle = False):
        """
        Get SAT of HAMILTONIAN decision problem.

        Parameters:
        ______________
            graph: networkx.classes.graph.Graph
                graph object from networkx

            cycle: bool
                True if SAT of hamiltonian cycle is needed

        Returns:
        ______________
            clauses: list(list)
                each list is a disjunction of variables in it
        """
        n_nodes = len(graph.nodes)
        non_existent_edges = non_edges(graph)

        clauses = []
        clauses.extend(self._get_node_must_appear_in_the_path_clauses(n_nodes))
        clauses.extend(self._get_node_cannot_appear_twice_in_the_path_clauses(n_nodes))
        clauses.extend(self._get_each_position_on_the_path_not_null_clauses(n_nodes))
        clauses.extend(self._get_no_two_nodes_occupy_same_position_clauses(n_nodes))
        clauses.extend(self._get_nonadjacent_nodes_cannot_be_adjacent_clauses(non_existent_edges, n_nodes, cycle))

        return clauses

    def _get_node_must_appear_in_the_path_clauses(self, n_nodes):
        """
        Get clauses to ascertain that each node appears in the final path. Refer to the slides.

        Parameters:
        ______________
            n_nodes: int
                number of nodes in the graph

        Returns:
        ______________
            clauses: list(list); a total of n_nodes list. one corresponding to each node.
                each list is a disjunctions of variables in it.

        """
        n_positions = n_nodes
        clauses = []
        for j in range(n_nodes):
            clauses.append([self._get_variable_number(i+1, j+1, n_nodes) for i in range(n_positions)])
        return clauses

    def _get_node_cannot_appear_twice_in_the_path_clauses(self, n_nodes):
        """
        Get clauses to ascertain that each node cannot appear twice in a path. Refer to the slides.

        Parameters:
        ______________
            n_nodes: int
                number of nodes in the graph

        Returns:
        ______________
            clauses: list(list); a total of n_ndoes + n_nodes*(n_nodes - 1)/2 list. one corresponding to each node.
                each list is a disjunctions of variables in it.

        """
        n_positions = n_nodes
        clauses = []
        for j in range(n_nodes):
            for i in range(n_positions):
                for k in range(i):
                    clauses.append([-1*self._get_variable_number(i+1, j+1,n_nodes), -1*self._get_variable_number(k+1, j+1,n_nodes)])
        return clauses

    def _get_each_position_on_the_path_not_null_clauses(self, n_nodes):
        """
        Get clauses to ascertain that each position has a node. Refer to the slides.

        Parameters:
        ______________
            n_nodes: int
                number of nodes in the graph

        Returns:
        ______________
            clauses: list(list); a total of n_nodes list. one corresponding to each position.
                each list is a disjunctions of variables in it.

        """
        n_positions = n_nodes
        clauses = []
        for i in range(n_positions):
            clauses.append([self._get_variable_number(i+1, j+1, n_nodes) for j in range(n_nodes)])
        return clauses

    def _get_no_two_nodes_occupy_same_position_clauses(self, n_nodes):
        """
        Get clauses to ascertain that each positions cannot have two nodes simultaneously. Refer to the slides.

        Parameters:
        ______________
            n_nodes: int
                number of nodes in the graph

        Returns:
        ______________
            clauses: list(list); a total of n_ndoes + n_nodes*(n_nodes - 1)/2 list. one corresponding to each position.
                each list is a disjunctions of variables in it.

        """
        n_positions = n_nodes
        clauses = []
        for i in range(n_positions):
            for j in range(n_nodes):
                for k in range(j):
                    clauses.append([-1*self._get_variable_number(i+1, j+1,n_nodes), -1*self._get_variable_number(i+1, k+1,n_nodes)])
        return clauses

    def _get_nonadjacent_nodes_cannot_be_adjacent_clauses(self, non_existent_edges, n_positions, cycle = False):
        """
        Get clauses to ascertain that adjacent positions in the path are not occupied by non-adjacent nodes in the graph. Refer to the slides.
        Example: (1, 2) is a non-edge in the grap then the set of clauses should be
         -1, -5; -4, -8; -7 -2 to exclude the possibility of 2 coming after 1
         -2, -4; -5, -7; -8 -1 to exclude the possibility of 1 coming after 2 (this part is not included in the slides)

        Parameters:
        ______________
            non_existent_edges: generator object
                each element is a node pair that doesn't have an edge in a graph

            n_positions: int
                number of positions in the path. equal to number of nodes.

            cycle: bool
                True: if the SAT of hamiltonian cycle needs to be generated; False: If SAT of hamiltonian path needs to be generated

        Returns:
        ______________
            clauses: list(list); a total of non_edges * (n_nodes -1) claues.
                each list is a disjunctions of variables in it.
        """
        clauses = []
        for (i,j) in non_existent_edges:
            for k in range(n_positions-1):
                clauses.append([-1*self._get_variable_number(k+1,i+1, n_positions), -1*self._get_variable_number(k+2,j+1, n_positions)])
                clauses.append([-1*self._get_variable_number(k+1,j+1, n_positions), -1*self._get_variable_number(k+2,i+1, n_positions)])

            if cycle:
                clauses.append([-1*self._get_variable_number(n_positions,i+1, n_positions), -1*self._get_variable_number(1,j+1, n_positions)])
                clauses.append([-1*self._get_variable_number(n_positions,j+1, n_positions), -1*self._get_variable_number(1,i+1, n_positions)])
        return clauses


    def _get_variable_number(self,i,j, n_nodes):
        """
        Get the variable number to be put in cnf. Variables in the slide are indexed by i,j.
        These variables need to be converted to a single integer for representation in CNF.

        Parameters:
        ______________
            i: int
                row index; represents the index in hamiltonian path
            j: int
                column index; represents a node
        Returns:
        ______________
            variable_number: int
                variable number as indexed in the final SAT CNF
        """
        return (i-1)*n_nodes + j


if __name__ == "__main__":
    # test
    from networkx.generators import random_graphs
    graph = random_graphs.barabasi_albert_graph(3,1)

    ham_sat = HamiltonianCNF()
    clauses = ham_sat.get_cnf(graph)
    print len(clauses)
    print clauses
