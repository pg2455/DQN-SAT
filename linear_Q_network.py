# /usr/bin/env python2
import torch, random
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as bp
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
                    node_feature_dimension = 1, edge_feature_dimension = 1):
        """
        Instantiate the class with weights needed for message passing and neural network.
        """
        super(QNetwork, self).__init__()
        self.update_iterations = embedding_update_iterations
        self.embedding_dimension = embedding_dimension
        self.edge_feature_dimension = edge_feature_dimension
        self.node_feature_dimension = node_feature_dimension

        self.theta1 = nn.Parameter(torch.Tensor(embedding_dimension, node_feature_dimension))
        self.theta2 = nn.Parameter(torch.Tensor(embedding_dimension, embedding_dimension))
        self.theta3 = nn.Parameter(torch.Tensor(embedding_dimension, embedding_dimension))
        self.theta4 = nn.Parameter(torch.Tensor(embedding_dimension, edge_feature_dimension))

        self.theta5 = nn.Parameter(torch.Tensor(2*embedding_dimension, 2))
        self.theta6 = nn.Parameter(torch.Tensor(embedding_dimension, embedding_dimension))
        self.theta7 = nn.Parameter(torch.Tensor(embedding_dimension, embedding_dimension))

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes weights of the parameters.

        Parameters:
        ______________
            None

        Returns:
        ______________
            None
        """
        # try centering inputs
        # nn.init.normal(self.theta1, std = self.node_feature_dimension ** -0.5)
        # nn.init.normal(self.theta2, std = self.embedding_dimension ** -0.5)
        # nn.init.normal(self.theta3, std = self.embedding_dimension ** -0.5)
        # nn.init.normal(self.theta4, std = self.edge_feature_dimension ** -0.5)
        # nn.init.normal(self.theta5, std = (2*self.embedding_dimension) ** -0.5)
        # nn.init.normal(self.theta6, std = self.embedding_dimension ** -0.5)
        # nn.init.normal(self.theta7, std = self.embedding_dimension ** -0.5)

        # std = (self.embedding_dimension * (4*self.embedding_dimension + self.node_feature_dimension + self.edge_feature_dimension + 4)) ** -0.5
        std = 0.0095
        nn.init.normal(self.theta1, std = std)
        nn.init.normal(self.theta2, std = std)
        nn.init.normal(self.theta3, std = std)
        nn.init.normal(self.theta4, std = std)
        nn.init.normal(self.theta5, std = std)
        nn.init.normal(self.theta6, std = std)
        nn.init.normal(self.theta7, std = std)

    def forward(self, node_feature_matrix, adjacency_matrix, edge_feature_matrix, current_embedding, unassigned_decision_variable_indices, assigned_variable_indices):
        """
        Defines the forward pass of the algorithm. There are two parts to the forward pass.
        1. Message passing to update the embeddings which is given by equation 3 in the paper
        2. Q-value approximation which is given by equation 4 in the paper

        n_nodes = number of nodes in the graph

        Parameters:
        ______________
            node_feature_matrix: torch.autograd.Variable( torch.FloatTensor (size = [node_feature_dimension, n_nodes]))
                each column represents the features of a node in the graph

            adjacency_matrix: torch.autograd.Variable( torch.ByteTensor (size = [n_nodes, n_nodes]))
                graphical representation of nodes and edges in the form of adjacency matrix.

            edge_feature_matrix: torch.autograd.Variable( torch.FloatTensor (size = [n_nodes, n_nodes, edge_feature_dimension]))
                (i,j,r): r represents the vector of edge feature in for an edge between j-r. For each r, there is a symmetric matrix.

            current_embedding: torch.autograd.Variable( torch.FloatTensor (size = [embedding_dimension, n_nodes]))
                current embedding of the nodes in the graph

            unassigned_decision_variables: list
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
        assert node_feature_matrix.size(1) == adjacency_matrix.size(0) == edge_feature_matrix.size(0) == edge_feature_matrix.size(1)
        assert torch.eq(adjacency_matrix.t(), adjacency_matrix).all()
        assert current_embedding.size(0) == self.embedding_dimension
        # assert torch.eq(edge_feature_matrix[2].t(), edge_feature_matrix[2]).all()
        assert edge_feature_matrix.size(2) == self.edge_feature_dimension
        assert node_feature_matrix.size(0) == self.node_feature_dimension

        n_nodes = adjacency_matrix.size(0)

        for _ in range(self.update_iterations):
            # equation 3
            # bp()
            tmp = current_embedding.data
            term1 = self.theta1.mm(node_feature_matrix)
            # term2 = self.theta2.mm(F.tanh(current_embedding.mm(adjacency_matrix)))
            term2 = self.theta2.mm(current_embedding.mm(adjacency_matrix))

            term3 = self.theta4.expand(n_nodes, self.embedding_dimension, self.edge_feature_dimension)
            term3 = term3.bmm(edge_feature_matrix.transpose(-1,-2)).sum(2).permute(1,0) # - negative indexing for transpose
            term3 = self.theta3.mm(term3)

            # F.relu(F.tanh(term1 + term2 + term3))
            current_embedding = term1 + term2 + term3
            delta = tmp - current_embedding.data

        # bp()
        # current sstate
        state = self.get_current_state(current_embedding, assigned_variable_indices, unassigned_decision_variable_indices)

        # equation 4 view
        a_ = self.theta6.mm(state.expand(len(unassigned_decision_variable_indices), self.embedding_dimension).transpose(1,0)) # state is same for all the actions
        b_ = self.theta7.mm(current_embedding[:, unassigned_decision_variable_indices])
        q_values = self.theta5.transpose(1,0).mm(F.relu(torch.cat([a_,b_])))
        return current_embedding, q_values, delta

    def get_current_state(self, current_embedding, assigned_variable_indices = None, unassigned_decision_variable_indices = None):
        """
        This is a state calculation. Depending on which variables are assigned at the moment (this is also dependent on state representation).

        Parameters:
        _____________
            current_embedding:
            assigned_variable_indices:
            unassigned_decision_variable_indices:

        Returns:
        ___________

            state : Tensor()
                State vector of the graph.

        """
        #state of the entire graph
        return current_embedding.sum(-1)
        if assigned_variable_indices:
            return current_embedding[:, assigned_variable_indices].sum(-1) # sum of all assigned variable vectors

        return current_embedding[:,[random.randint(0, current_embedding.size(1))]]
