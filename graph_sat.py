#! /usr/bin/env python2
import networkx, pycosat
from hamiltonian_cnf import HamiltonianCNF
from cnf_compiler import CNFCompiler

class GraphSAT(object):
    """
    Get SAT formula for various decision problems on graphs

    Parameters:
    ______________
        None
    """
    def __init__(self):
        """
        instantiate objects to convert graph decision problems to SAT
        """
        self.hamiltonian_cnf = HamiltonianCNF()
        self.cnf_compiler = CNFCompiler()

    def get_hamiltonian_sat(self, graph, file_path = None, solve = False):
        """
        Get a SAT formula for deicision problems on graphs.
        Decision problem: Is there a Hamiltonian Cycle in the given graph?

        Parameters:
        ______________
            graph: networkx.classes.graph.Graph
                graph object from networkx

            file_path: str
                saves the cnf string in DIMACS format; returns clauses if None
        """
        clauses = self.hamiltonian_cnf.get_cnf(graph)
        n_nodes = len(graph.nodes)
        n_variables = n_nodes * n_nodes
        comments = []
        if solve:
            print "solving .."
            comments += [self.get_solver_comment(clauses)]

        if file_path:
            return self.cnf_compiler.get_dimacs_string_from_cnf(clauses, n_variables, save_file = file_path, comments = comments)

        return clauses, comments

    def get_solver_comment(self, clauses):
        """
        get the comment suggesting whether the given cnf is satisfiable.

        Parameters:
        _______________
            clauses: list(list)
                each list is a disjunction of variables

        Returns:
        _______________
            comment: str

        """
        solution = pycosat.solve(clauses)
        if type(solution) == list:
            solution = True

        return "satisfiable {}".format(solution)
