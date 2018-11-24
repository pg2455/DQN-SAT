# /usr/bin/env python2
import torch, random
from collections import defaultdict
from templates import *

class CVIG(DataWrapper):
    """
    This is the representation of CNF in CVIG(Clause Variable Incidence Graph) format.
    Its an undirected bipartite graphical representation.

    Paramters:
    ______________
        n_variables: int
            number of decision variables

        n_clauses: int
            number of clauses in SAT formula

        clauses: list(list)
            clauses of the SAT formula, where each clause is a list of variables

    """
    def __init__(self, n_variables, n_clauses, clauses):
        Node.reset()
        self.n_variables = n_variables
        if n_clauses != len(clauses):
            warnings.warn("n_clauses does not match number of clauses. Using n_clauses as length of the list")
        self.n_clauses = len(clauses)
        clause_sizes = map(len,clauses)
        self.min_clause_size = min(clause_sizes)
        self.max_clause_size = max(clause_sizes)
        self.FEATURE_LIST = FEATURE_LIST
        self.NODE_FEATURE_DIMENSION = NODE_FEATURE_DIMENSION
        self.NODE_BIAS = NODE_BIAS
        self.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION

        super(CVIG, self).__init__(clauses)

    def update_state(self, new_assigned_variables, learnt_clauses = []):
        """
        After the action is performed, the new state is observed in terms of newly assigned variables and a set of learnt clauses (if any).

        Parameters:
        ______________
            new_assigned_variables: list(int)
                current assignment

            learnt_clauses: list(list(int))
                new clauses as a result of conflict

        """
        # list of assigned variable indices
        self.update_node_assignment(new_assigned_variables)
        self.update_features_from_clause(learnt_clauses, finalize_matrices=True)

    def get_current_state(self):
        return self.get_node_feature_matrix(), \
         self.get_adjacency_matrix(), \
         self.get_edge_feature_matrix(), \
         self.get_current_embedding(),\
         self.get_unassigned_decision_variable_indices(), \
         self.get_assigned_decision_variable_indices()

    def is_satisfied(self):
        return len(self.unassigned_decision_variables) == 0

    def set_current_embedding(self, current_embedding):
        """
        Sets the current embedding of each variable. If current_embedding needs to be initialized using
        some other logic, it can be done outside of this class.

        Parameters:
        ______________
            current_embedding: torch.Tensor(size = embedding_dimension x n_nodes)
                embedding matrix where each column is an embedding of the variable represented by the column

        Returns: None
        """
        self.current_embedding = current_embedding

class UtilityGraph(object):
    """
    This class outlines a typical graph class. The functions defined here will be
    required in DQN.
    """
    def __init__(self, n_variables, n_clauses, clauses, graph_type='cvig'):
        if graph_type == 'cvig':
            self.graph = CVIG(n_variables, n_clauses, clauses)

    def set_current_embedding(self, current_embedding):
        self.graph.set_current_embedding(current_embedding)

    def get_current_embedding(self):
        return self.graph.get_current_embedding()

    def get_current_state(self):
        return self.graph.get_current_state()

    def update_state(self, new_assigned_variables, learnt_clauses):
        self.graph.update_state(new_assigned_variables, learnt_clauses)

    def is_satisfied(self):
        return self.graph.is_satisfied()
