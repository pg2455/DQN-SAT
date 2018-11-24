import torch, random
import torch.nn as nn  
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    This is the Q-function neural network that will return Q_value for choosing a
    variable (i.e a p-dimensional embedding) in a state (defined by a p-dimensional vector). It \
    returns a 2 X 1 vector corresponding to the value of the variable in False or True state.

    Refer to the paper: https://arxiv.org/abs/1704.01665. Only difference here is in the dimension of theta_5.
    It is 2p x 2 since the needed output is a 2-dimensional vector; q-value for choosing False or True.
    Parameters:
    ______________
        embedding_dimension: int
            number of dimensions desired for embedding of variables (p in the paper)

        embedding_update_iterations: int
            number of iterations needed to update the embedding before using the Q function (T in the paper)

        node_feature_dimension: int
            number of dimensions in the vector for node features (x_v in the paper)

        edge_feature_dimesion: int
            number of dimensions in the vector for edge features (w(u,v) in the paper)

    """

    def __init__(self, embedding_dimension = 100 , embedding_update_iterations = 4,
                    node_feature_dimension = 1, edge_feature_dimension = 1, global_state_dimension=200, model_path = None):
        """
        Instantiate the class with weights needed for message passing and neural network.
        """
        super(QNetwork, self).__init__()
        self.update_iterations = embedding_update_iterations
        self.embedding_dimension = embedding_dimension
        self.edge_feature_dimension = edge_feature_dimension
        self.node_feature_dimension = node_feature_dimension
        self.global_state_dimension = global_state_dimension
        

        self.theta1 = nn.Linear(node_feature_dimension, embedding_dimension,  bias = False)
        self.theta2 = nn.Linear(embedding_dimension, embedding_dimension, bias = False)
        self.theta3 = nn.Linear(embedding_dimension, embedding_dimension, bias = False)
        self.theta4 = nn.Linear(edge_feature_dimension, embedding_dimension, bias = False)

        self.theta5_1 = nn.Linear(embedding_dimension, 1, bias = False)
        self.theta5_2 = nn.Linear(embedding_dimension, 2, bias = False)

        self.theta6 = nn.Linear(global_state_dimension, embedding_dimension, bias = False)
        self.theta7 = nn.Linear(embedding_dimension, embedding_dimension, bias = False)
        
        self.theta8 = nn.Linear(global_state_dimension, embedding_dimension, bias = False)
        self.theta9 = nn.Linear(embedding_dimension, global_state_dimension, bias = False)

        if model_path:
            self.transfer_weights(model_path)
        else:
            self.initialize_weights()
        
        self.use_weight_norm()
        
        
        
    def use_weight_norm(self):
        self.theta1 = weight_norm(self.theta1, name="weight")
        self.theta2 = weight_norm(self.theta2, name="weight")
        self.theta3 = weight_norm(self.theta3, name="weight")
        self.theta4 = weight_norm(self.theta4, name="weight")
        self.theta5_1 = weight_norm(self.theta5_1, name="weight")
        self.theta5_2 = weight_norm(self.theta5_2, name="weight")
        self.theta6 = weight_norm(self.theta6, name="weight")
        self.theta7 = weight_norm(self.theta7, name="weight")
        self.theta8 = weight_norm(self.theta8, name="weight")
        self.theta9 = weight_norm(self.theta9, name="weight")
    

    def initialize_weights(self):
        """
        Initializes weights of the parameters.

        Parameters: None

        Returns: None
        """
        std= 0.05
        nn.init.normal_(self.theta1.weight, std = std)
        nn.init.normal_(self.theta2.weight, std = std)
        nn.init.normal_(self.theta3.weight, std = std)
        nn.init.normal_(self.theta4.weight, std = std)
        # nn.init.normal(self.theta5, std = std)
        nn.init.normal_(self.theta6.weight, std = std)
        nn.init.normal_(self.theta7.weight, std = std)
        nn.init.normal_(self.theta5_1.weight, std = std)
        nn.init.normal_(self.theta5_2.weight, std = std)

    
    def transfer_weights(self, MODEL_PATH):
        """
        Initialize weights from the model at MODEL_PATH

        Parameters:
        _______________

            MODEL_PATH: str
                filepath of the model; model is saved using torch.save(model)

        """
        _model  = torch.load(MODEL_PATH)
        for param, _param in zip(self.parameters(), _model.parameters()):
            param.data = _param.data

    def forward(self, node_feature_matrix, adjacency_matrix, edge_feature_matrix, current_embedding, unassigned_decision_variable_indices, assigned_variable_indices):
        """
        Defines the forward pass of the algorithm. There are two parts to the forward pass.
        1. Message passing to update the embeddings which is given by equation 3 in the paper
        2. Q-value approximation which is given by equation 4 in the paper

        n_nodes = number of nodes in the graph

        Parameters:
        ______________
            node_feature_matrix:torch.FloatTensor (size = [ n_nodes, node_feature_dimension])
                each column represents the features of a node in the graph

            adjacency_matrix: torch.SparseTensor (size = [n_nodes, n_nodes])
                graphical representation of nodes and edges in the form of adjacency matrix.

            edge_feature_matrix: torch.SparseTensor (size = [n_nodes, n_nodes, edge_feature_dimension])
                (i,j,r): r represents the vector of edge feature in for an edge between j-r. For each r, there is a symmetric matrix.

            current_embedding: torch.FloatTensor (size = [ n_nodes, embedding_dimension])
                current embedding of the nodes in the graph

            unassigned_decision_variable_indices: list
                its a mask to get the columns from Q-value. It represents variables for which Q-value is needed

            assigned_variable_indices: list
                it is needed to calculate the current state depending on the helper function

        Returns:
        _______________
            current_embedding:torch.FloatTensor (size = [embedding_dimension, n_nodes])
                new updated embeddings

            q_values:

        """

        # # check dimensions
        assert node_feature_matrix.size(0) == adjacency_matrix.size(0) == edge_feature_matrix.size(0) == edge_feature_matrix.size(1)
        assert current_embedding.size(1) == self.embedding_dimension
        assert edge_feature_matrix.size(2) == self.edge_feature_dimension
        assert node_feature_matrix.size(1) == self.node_feature_dimension

        n_nodes = adjacency_matrix.size(0)
        _adj = adjacency_matrix.to_dense()
        n_variables = len(assigned_variable_indices) + len(unassigned_decision_variable_indices)
        
        state_embedding = self.get_current_state(current_embedding, n_variables)
        
        for _ in range(self.update_iterations):
            # equation 3
            tmp = current_embedding.data
            term1 = self.theta1(node_feature_matrix) 
            term2 = self.theta2(_adj.mm(current_embedding))/torch.unsqueeze(_adj.sum(1), dim=1)
            
            term2[n_variables:,:]  += self.theta8(state_embedding)

            term3 = self.theta4(edge_feature_matrix.to_dense()[:,]).sum(1)
            term3 = self.theta3(F.relu(term3))
                        
            current_embedding = F.relu(term1 + term2 + term3)
            state_embedding = self.get_current_state(current_embedding, n_variables)
            
            delta = tmp - current_embedding.data

        b_ = F.relu(self.theta7(current_embedding[unassigned_decision_variable_indices,:]))
        a_ = F.relu(self.theta6(state_embedding))

        q_values = self.theta5_1(a_).expand(b_.size(0),2) + self.theta5_2(b_)
        
        del term1, term2, term3, a_,b_
        return current_embedding, q_values.permute(1,0), delta

    def get_current_state(self, current_embedding, n_variables, assigned_variable_indices = None, unassigned_decision_variable_indices = None):
        """
        This is a state calculation. Depending on which variables are assigned at the moment (this is also dependent on state representation).

        Parameters:
        _____________
            current_embedding: n_nodes, embedding_dimension
            assigned_variable_indices:
            unassigned_decision_variable_indices:

        Returns:
        ___________

            state : Tensor()
                State vector of the graph.

        """
        #state of the entire graph                                      
        clause_embeddings = current_embedding[n_variables:,]
        return F.relu(self.theta9(clause_embeddings).mean(dim=0))