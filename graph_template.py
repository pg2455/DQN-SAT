
import sys, torch
if "/home/satgpupg/OvalSAT/" not in sys.path:  sys.path.append("/home/satgpupg/OvalSAT/");
from proxy_configuration import NODE_FEATURE_LIST, NODE_BIAS, NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, EMBEDDING_DIMENSION

class Node(object):
    rows = {f:index for index,f in enumerate(NODE_FEATURE_LIST)}
    def __init__(self, index, node_type, clause = []):
        assert (node_type == 'v' and clause == []) or (node_type =='c' and clause != []), \
        "Incorrect node initialization - type:{}, clause:{}".format(node_type, clause);
        self.type = node_type
        self.index = index
        self.value = 0 # 0: unassigned; 1/-1: T/F
        self.neighbors = []
        if node_type =='c':
            self.clause = clause

               
    def update_clause_based_features(self, clause, data_interface):
        # called for both clause and variables
        for f, index in self.rows.iteritems():
            if f[0] == 'c':
                getattr(self, 'update_'+f[2:])(clause, index, data_interface)
    
    def update_solver_based_features(self, value, data_interface):
        assert type(value) == dict, "Incorrect value type: {}".format(value)
        # these will always be called on variables; 
        for f, index in self.rows.iteritems():
            if f[0] == 's':
                getattr(self, 'update_'+f[2:])(value, index, data_interface)
    
    def update_assignment(self, value, index, data_interface):
        if 'assignment' not in value:
            return None

        data_interface.assign_node_feature(index, self.index,  value['assignment'])
        if self.type == 'v' and value['assignment'] != 0:
            # update clause node features
            for i in self.neighbors:
                # if the variable is present as postive literal, assigning True will satisfy the clause
                if (value['assignment'] > 0 and  i > 0) or (value['assignment'] < 0 and i < 0): 
                    data_interface.assign_node_feature(index, abs(i), 1)

    def update_decision_level(self, value, index, data_interface):
        if 'decision_level' not in value:
            return None
        
        if self.type == 'v':
            data_interface.assign_node_feature(index, self.index, value['decision_level'])
        
    def update_n_neighbors(self, clause, index, data_interface):
        if self.type == 'v' and clause.clause != []:
            literal_type = [x for x in clause.clause if abs(x) == self.index+1][0] > 0
            self.neighbors.append(clause.index *[-1, 1][literal_type] )
            #self.n_neighbors += 1
            data_interface.node_features[index, self.index] = 1.0*len(self.neighbors)/data_interface.total_clauses
        
        if self.type == 'c':
            data_interface.assign_node_feature(index, self.index, 1.0*len(self.clause)/data_interface.n_variables)
    
    def __str__(self):
        return "{}{}_{}".format(self.type, self.index, self.value)

class GraphData(object):
    def __init__(self, n_variables):
        self.n_variables = n_variables
        self.total_clauses = 0
        self.node_features = torch.zeros(0,0)
        self.current_embedding = torch.zeros(0,0)
        self.edge_data = []
        self.adjacent_nodes = [[], []]
        self.adjacency_list = {}
       
    def featurize_clause(self, clause, nodes):
        assert type(clause) == Node and clause.type == 'c', "Incorrect arguments. Expected Clause "
        clause.update_clause_based_features(clause, self)
        self.total_clauses += 1
        for i in clause.clause:
            nodes['v{}'.format(abs(i))].update_clause_based_features(clause, self)
            self.add_edge(nodes['v{}'.format(abs(i))].index, i, clause)
    
    def add_edge(self, variable_index, v, clause):
        self.adjacent_nodes[0].append(variable_index)
        self.adjacent_nodes[1].append(clause.index)
        
        f = [2*(v > 0) - 1] # literal type
        self.edge_data.append(f)
                    
    def init_node_feature(self):
        # create a column for the node in node_feature matrix
        self.node_features = torch.cat([self.node_features, torch.zeros(NODE_FEATURE_DIMENSION, 1) ], dim = 1)
        if NODE_BIAS:
            self.node_features[-1][-1] = 1 # bias

        # create a column for the node in current_embedding
        self.current_embedding = torch.cat([self.current_embedding, torch.zeros(1,EMBEDDING_DIMENSION) ], dim = 0)
 
    def clear_clause_assignments(self, index, n_variables):
        self.node_features[index, n_variables:] = 0
        
    def assign_node_feature(self, index1, index2, value):
        self.node_features[index1, index2] = value
    
    def finalize_feature_matrices(self):
        x,y = self.adjacent_nodes
        i = torch.LongTensor([x+y, y+x])
        self.adjacency_matrix = torch.sparse.FloatTensor(i,torch.FloatTensor([1 for _ in x+y]))
        v = torch.FloatTensor(self.edge_data + self.edge_data)
        self.edge_feature_matrix = torch.sparse.FloatTensor(i, v)
        
    def get_edge_feature_matrix(self):
        return self.edge_feature_matrix

    def get_node_feature_matrix(self):
        return self.node_features

    def get_adjacency_matrix(self):
        return self.adjacency_matrix

    def get_current_embedding(self):
        return self.current_embedding
    
class Graph(object):
    def __init__(self, n_variables, clauses):
        self.n_variables = n_variables
        self.n_clauses = len(clauses)
        self.n_learnt_clauses = 0
        self.n_learnt_clauses = 0
        self.nodes, self.unassigned_decision_variables, self.assigned_decision_variables = {}, {}, {}
        self.data = GraphData(n_variables)
        self.clause_index = 0
        
        for index in range(n_variables):
            self.nodes['v{}'.format(index+1)] = Node(index, 'v')
            self.data.init_node_feature()
            self.unassigned_decision_variables['v{}'.format(index+1)] = self.nodes['v{}'.format(index+1)] 
        self.index = index

        self.assigned_decision_variables = {}
        self.update_state([], clauses)
        
        
    def update_state(self, new_assigned_variables, clauses = []):
        if self.clause_index > self.n_clauses -1:
            self.n_learnt_clauses += len(clauses)
        self._update_features(clauses)
        
        self._update_node_assignment(new_assigned_variables)
        self._commit_features()
        

    def _update_features(self, clauses):
        assert sum([type(i) != list for i in clauses]) == 0, "GraphFeatures.update_features: Expected clauses to be of type [[],]"
        for clause in clauses:        
            self.index += 1
            self.clause_index += 1
            self.nodes['c{}'.format(self.clause_index)] = x = Node(self.index, 'c', clause)
            self.data.init_node_feature()
            self.data.featurize_clause(x, self.nodes)
            
    def _update_node_assignment(self, assignment):
        new = set([self.nodes['v{}'.format(abs(x))] for x in assignment])
        
        for x in set(self.assigned_decision_variables.values()) - new:
            _v = {'assignment':0}
            x.update_solver_based_features(_v, self.data )
            self.unassigned_decision_variables['v{}'.format(x.index+1)] = x
        
        self.data.clear_clause_assignments(Node.rows['s_assignment'], self.n_variables)
        self.assigned_decision_variables = {}
        for x in assignment:
            _v = {'assignment': [-1,1][x>0]}
            self.nodes['v{}'.format(abs(x))].update_solver_based_features(_v, self.data )
            self.assigned_decision_variables['v{}'.format(abs(x))] = self.nodes['v{}'.format(abs(x))]
            self.unassigned_decision_variables.pop('v{}'.format(abs(x)), 0)
   
    def get_current_state(self):
        return [self.data.get_node_feature_matrix().permute(1,0), \
            self.data.get_adjacency_matrix(), \
            self.data.get_edge_feature_matrix(), \
            self.data.get_current_embedding(),\
            self.get_unassigned_decision_variable_indices(), \
            self.get_assigned_decision_variable_indices()]
    
    def get_unassigned_decision_variable_indices(self):
        return [i.index for i in self.unassigned_decision_variables.values()]

    def get_assigned_decision_variable_indices(self):
        return [i.index for i in self.assigned_decision_variables.values()]

    
    def set_current_embedding(self, embeddings):
        self.data.current_embedding = embeddings
    
    def get_current_embedding(self):
        return self.data.current_embedding

    def _commit_features(self):
        self.data.finalize_feature_matrices()
    
    def is_satisfied(self):
        return self.get_unassigned_decision_variable_indices() == []
