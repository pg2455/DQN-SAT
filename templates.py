#! usr/bin/env python2
import torch, warnings
from collections import defaultdict
from proxy_configuration import *

class Features(object):
    """
    Holds algorithm data ready for feature matrices. Updating the feature matrices within this class enables
    singple point of control on feature matrices

    Data:
    ______________
        edge_data:
        adjacent_nodes:
        node_features:
        edge_feature_matrix:
        adjacency_matrix:

    List of Topograghical features:
    __________________________________
        Node Features: (normalized by)
            n_neighbors: atom node (n_clauses)--> number of clauses an atom is present in; clause node (n_variables) --> number of atoms it has
            n_pos_neighbors: atom node (n_clauses) --> n_neighbors; clause node (len clause)--> number of positive literals
            n_neg_neighbors: atom node --> 0; clause node (len clause)--> number of negative literals
            avg_n_variables: atom node (n_variables)--> average number of other atoms in the clause; clause node --> n_neighbors
            variance_n_variables: atom node --> variance in number of other atoms (if only one clause: -1); clause node --> -1; -1 ==> not enough data to compute variance
            n_min_size_clauses: atom node (n_clauses) --> number of min size clauses the atom is present in; clause node --> 1 if its a min size clause, 0 otherwise
            n_max_size_clauses: atom node (n_clauses) --> number of max size clauses the atom is present in; clause node --> 1 if its a max size clause, 0 otherwise
            n_literals: 1(if the atom is present in one form only) or 2(if both pos and neg are present)
            n_pos_occurence: atom node (self.n_neighbors) --> number of times it is present as positive literal; clause node --> 1
            n_neg_occurence: atom node (self.n_neighbors) --> numnber of times it is present as negative literal; clause node --> 0

        Edge Features:
            literal_type: 1 if the atom is present in positive literal form else -1
            n_other_variables: fraction of other variables (len of clause - 1) present in the clause (fraction of n_variables)

    List of Dynamical Solver based features:
    ___________________________________________
        Node Features:
            current_solution_state:-1,0,1; 0 ==> unassigned

        Edge Features:

    """
    def __init__(self):
        self.node_features = torch.zeros(0,0)
        self.current_embedding = torch.zeros(0,0)
        self.edge_data = []
        self.adjacent_nodes = [[],[]] # store tuples
        self.feature_index ={
            key:i for i,key in enumerate(FEATURE_LIST)
        }


    def set_node_feature(self, index, key, value):
        if key in self.feature_index:
            self.node_features[getattr(self, key)][index] = value

    # not used atm
    def increment_node_feature(self, index, key, value):
        if key in self.feature_index:
            self.node_features[getattr(self, key)][index] += value

    def get_node_feature(self, index, key):
        return self.node_features[getattr(self, key)][index]

    def __getattr__(self, key):
        return self.feature_index[key]

    def initialize_node_feature(self):
        # create a column for the node in node_feature matrix
        self.node_features = torch.cat([self.node_features, torch.zeros(NODE_FEATURE_DIMENSION, 1) ], dim = 1)
        if NODE_BIAS:
            self.node_features[-1][-1] = 1 # bias

        # create a column for that node in current_embedding
        self.current_embedding = torch.cat([self.current_embedding, torch.zeros(EMBEDDING_DIMENSION, 1) ], dim = 1)

    def add_edge(self, to, fro, literal, clause):
        self.adjacent_nodes[0].append(to)
        self.adjacent_nodes[1].append(fro)
        self.edge_data.append(self.get_edge_features(literal, clause))

    def get_edge_features(self, literal, clause):
        """
        Refer to update_features_from_clause for the feature description

        Parameters:
        ____________
            literal: int
                form of variable present in the class e.g. -1 ==> -x1, 5 ==> x5

            clause: list(int)
                list of literals in a clause

        Returns:
        ____________
            edge_features: list(float)
        """
        return [2 * (literal > 0) - 1, 1.0*(len(clause) - 1)/self.n_variables]


    def get_edge_feature_matrix(self):
        return self.edge_feature_matrix.to_dense()

    def get_node_feature_matrix(self):
        return self.node_features

    def get_adjacency_matrix(self):
        return self.adjacency_matrix.to_dense()

    def get_current_embedding(self):
        return self.current_embedding

    def finalize_feature_matrices(self):
        """
        Stores the adjacency and edge feature matrix in readily accessible format.
        It makes the access to updated matrices quick.
        """
        x,y = self.adjacent_nodes
        i = torch.LongTensor([x+y, y+x])
        self.adjacency_matrix = torch.sparse.FloatTensor(i,torch.FloatTensor([1 for _ in x+y]))
        v = torch.FloatTensor(self.edge_data + self.edge_data)
        self.edge_feature_matrix = torch.sparse.FloatTensor(i, v)
        for key in self.assigned_decision_variables.keys() + self.unassigned_decision_variables.keys():
            self.nodes[key].commit_features()

class DataWrapper(Features):
    """
    It holds matrices needed for the algorithm - node_feature_matrix, edge_feature_matrix, get_adjacency_matrix
    This class abstracts data processing; it helps in understanding data and the algorithm;

    Parameters:
    ____________
        clauses: list(list(int))
            each element is a list of literals present in a clause

    Data:
    ____________
        nodes:
        assigned_decision_variables:
        unassigned_decision_variables:
        Features Object
    """
    def __init__(self, clauses):
        for key in ['n_clauses', 'min_clause_size', 'max_clause_size', 'n_variables']:
            if key not in self.__dict__:
                raise AssertionError(" Global variable: {} not found in Node object".format(key))

        self.nodes = {}
        self.assigned_decision_variables = {} #variable name --> index
        self.unassigned_decision_variables = {}
        super(DataWrapper, self).__init__()

        for i in xrange(self.n_variables):
            self.nodes['v{}'.format(i+1)] = x = AtomNode(self)

        self.update_features_from_clause(clauses, finalize_matrices = True)

    def update_features_from_clause(self,clauses, finalize_matrices = True):
        """
        Updates the features of the node based on clauses. Intuitively, these features should encode graph topology.

        Parameters:
        ___________________
            clauses: list(list(int))
                list of clause which is list(int)
                each int is a literal as it appears in a clause e.g. [-1, 5] ==> -x1 V x5 is the clause


        """
        if clauses and type(clauses[0]) != list:
            clauses = list(clauses)

        for clause in  clauses:
            self.nodes['c{}'.format(Node.count+1)] = x = ClauseNode(clause, self)
            for literal in clause:
                atom  = self.nodes['v{}'.format(abs(literal))]
                atom.update_features(literal, clause)
                self.add_edge(atom.get_index(), x.get_index(), literal, clause)

        if clauses and finalize_matrices:
            self.finalize_feature_matrices()

    def update_node_assignment(self, new_assignment):
        """
        updates the solver state and corresponding dynamic features of the nodes

        Parameters:
        ___________________
            new_assignment: list(int)
                each element represents the truth value of the variable e.g. -1 ==> 1 is assigned False, 2 ==> 2 is assigned True

        """
        new = set([self.nodes['v{}'.format(abs(x))] for x in new_assignment])
        _ = [x.unassign() for x in self.get_assigned_decision_variables() - new]

        for assignment in new_assignment:
            self.nodes['v{}'.format(abs(assignment))].assign(assignment > 0)

    def unassign_decision_variable(self, atom, index):
        if 'v{}'.format(atom) in self.assigned_decision_variables:
            self.assigned_decision_variables.pop('v{}'.format(atom))
        self.unassigned_decision_variables['v{}'.format(atom)] = index

    def assign_decision_variable(self, atom, index):
        if 'v{}'.format(atom) in self.unassigned_decision_variables:
            self.unassigned_decision_variables.pop('v{}'.format(atom))
        self.assigned_decision_variables['v{}'.format(atom)] = index

    def get_unassigned_decision_variable_indices(self):
        return self.unassigned_decision_variables.values()

    def get_assigned_decision_variable_indices(self):
        return self.assigned_decision_variables.values()

    def get_assigned_decision_variables(self):
        return set([self.nodes[k] for k in self.assigned_decision_variables])


class Node(object):
    """
    Class to represent a node which can correspond to an atom or a clause.
    It holds features and essential information to enable embedding updates.

    Parameters:
    ____________
        count : static int
            keeps count of nodes assigned
    """
    count = 0
    def __init__(self, data_wrapper):
        """
        index: int
            henceforth, data corresponding to this node (node_features, edge_features) will be refered to by this number

        atom: uint
            representation of literal in its base form (no information about sign)

        assignment: int; 0,1,-1
            Keeps track of assignment. 0 ==> not assigned yet; 1 ==> True; -1 ==> False

        """
        self.index = Node.count
        self.atom = self.index + 1
        Node.count += 1
        self.data_wrapper = data_wrapper
        self.data_wrapper.initialize_node_feature()
        self.unassign()

    def _assign(self, bool_value):
        if self.assignment != (2*bool_value - 1):
            # 1 <-> -1; 0 -> 1,-1
            pass

        self.assignment = 2*bool_value - 1
        self.set_assignment()

    def _unassign(self):
        #time_delta = self.counter - self.last_assign_counter
        self.assignment = 0
        self.set_assignment()

    def set_assignment(self):
        self.data_wrapper.set_node_feature(self.index, 'current_solution_state', self.assignment)

    def get_index(self):
        return self.index

    def get_feature_vector(self):
        return self.data_wrapper.node_features[:,self.index]

    @staticmethod
    def reset():
        """
        When a new graph is initialized, Node.count needs to be zeroed out
        """
        Node.count = 0

class ClauseNode(Node):
    def __init__(self, clause, data_wrapper):
        super(ClauseNode, self).__init__(data_wrapper)
        self.update_features(clause)

    def assign(self, bool_value):
        self._assign(bool_value)

    def unassign(self):
        self._unassign()

    def update_features(self, clause):
        """
        refer to the docstring of DataWrapper.update_features_from_clause for the list of features being tracked
        """
        self.data_wrapper.set_node_feature(self.index, 'n_neighbors', 1.0*len(clause)/self.data_wrapper.n_variables)

        if len(clause) <= self.data_wrapper.min_clause_size:
            self.data_wrapper.set_node_feature(self.index, 'n_min_size_clauses', 1)

        if len(clause) >= self.data_wrapper.max_clause_size:
            self.data_wrapper.set_node_feature(self.index, 'n_max_size_clauses', 1)

        self.data_wrapper.set_node_feature(self.index, 'n_pos_neighbors', 1.0*sum([1 for x in clause if x > 0])/len(clause))
        self.data_wrapper.set_node_feature(self.index, 'n_neg_neighbors', 1.0*sum([1 for x in clause if x < 0])/len(clause))
        self.data_wrapper.set_node_feature(self.index, 'avg_n_variables', 1.0*len(clause)/self.data_wrapper.n_variables)
        self.data_wrapper.set_node_feature(self.index, 'variance_n_variables', -1)
        self.data_wrapper.set_node_feature(self.index, 'n_literals', 1)
        self.data_wrapper.set_node_feature(self.index, 'n_pos_occurence', 1)
        self.data_wrapper.set_node_feature(self.index, 'n_neg_occurence', 0)

class AtomNode(Node):
    def __init__(self, data_wrapper):
        super(AtomNode, self).__init__(data_wrapper)
        self.n_neighbors = 0
        self.n_min_size_clauses = 0
        self.n_max_size_clauses = 0
        self.n_pos_neighbors = 0
        self.n_pos_occurence = 0
        self.n_neg_occurence = 0
        self.avg = 0.0
        self.var = -1.0
        self.unassign()

    def unassign(self):
        self._unassign()
        self.data_wrapper.unassign_decision_variable(self.atom, self.index)

    def assign(self, bool_value):
        self._assign(bool_value)
        self.data_wrapper.assign_decision_variable(self.atom, self.index)

    def update_features(self, literal, clause):
        """
        refer to the docstring of DataWrapper.update_features_from_clause for the list of features being tracked
        """
        self.n_neighbors += 1

        if len(clause) <= self.data_wrapper.min_clause_size:
            self.n_min_size_clauses += 1
        if len(clause) >= self.data_wrapper.max_clause_size:
            self.n_max_size_clauses += 1

        # n_pos_neighbors for decision variables is n_neighbors
        # self.data_wrapper.increment_node_feature(self.index, 'n_pos_neighbors', 1)
        # self.data_wrapper.increment_node_feature(self.index, 'n_neg_neighbors', 0)
        old_avg = self.avg
        delta1 = (len(clause) - 1) -  self.avg
        self.avg += 1.0*((len(clause) - 1) -  self.avg) / (self.n_neighbors)
        if self.n_neighbors >= 2:
            delta2 = (len(clause) - 1) -  old_avg
            new_var = (self.var*(self.n_neighbors - 2) + delta1 *  delta2)/(self.n_neighbors - 1)
            self.data_wrapper.set_node_feature(self.index, 'var_n_variables', new_var)
        else:
            self.data_wrapper.set_node_feature(self.index, 'var_n_variables', -2)

        self.n_pos_occurence += literal > 0
        self.n_neg_occurence += literal < 0

        if self.data_wrapper.get_node_feature(self.index, 'n_pos_occurence') and self.data_wrapper.get_node_feature(self.index, 'n_neg_occurence'):
            self.data_wrapper.set_node_feature(self.index, 'n_literals', 2)
        else:
            self.data_wrapper.set_node_feature(self.index, 'n_literals', 1)

    def commit_features(self):
        """
        normalizes the features of the atom so that they very between 0 and 1.
        Features normalized:
        ________________________________
            n_neighbors:
            n_min_size_clauses/n_max_size_clauses:
            n_pos_neighbors:

        """
        self.data_wrapper.set_node_feature(self.index, 'n_neighbors', 1.0*self.n_neighbors/self.data_wrapper.n_clauses)
        self.data_wrapper.set_node_feature(self.index, 'n_pos_neighbors', 1.0*self.n_neighbors/self.data_wrapper.n_clauses)
        self.data_wrapper.set_node_feature(self.index, 'n_min_size_clauses', 1.0*self.n_min_size_clauses/self.data_wrapper.n_clauses)
        self.data_wrapper.set_node_feature(self.index, 'n_max_size_clauses', 1.0*self.n_max_size_clauses/self.data_wrapper.n_clauses)
        self.data_wrapper.set_node_feature(self.index, 'n_pos_occurence', 1.0*self.n_pos_occurence/self.n_neighbors)
        self.data_wrapper.set_node_feature(self.index, 'n_neg_occurence', 1.0*self.n_neg_occurence/self.n_neighbors)
        self.data_wrapper.set_node_feature(self.index,'avg_n_variables' ,self.avg/self.data_wrapper.n_variables)
