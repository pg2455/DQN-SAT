#! /usr/bin/env python2
import torch.nn.functional as F
import torch.optim as optim

EMBEDDING_DIMENSION = 32
EMBEDDING_UPDATE_ITERATIONS = 4
EDGE_FEATURE_DIMENSION = 1
GLOBAL_STATE_DIMENSION = 16

TOPOGRAPHICAL_FEATURE_LIST = ['n_neighbors', 'n_pos_neighbors', 'n_neg_neighbors', 'avg_n_variables',
            'var_n_variables', 'n_min_size_clauses', 'n_max_size_clauses',
                'n_literals', 'n_pos_occurence', 'n_neg_occurence']
SOLVER_BASED_FEATURE_LIST = ['current_solution_state']
FEATURE_LIST = TOPOGRAPHICAL_FEATURE_LIST + SOLVER_BASED_FEATURE_LIST
NODE_BIAS = 1 # 1 or 0 only
NODE_FEATURE_LIST = ['s_assignment', 'c_n_neighbors']
NODE_FEATURE_DIMENSION = len(NODE_FEATURE_LIST) + NODE_BIAS

#
TRAINING_MEMORY_CAPACITY = 1000
VALIDATION_MEMORY_CAPACITY = 100
TRAIN_VALIDATION_SPLIT_RATIO = 0.1
VALIDATION_BATCH_SIZE = 100
N_TRAINING_ITER = 4
TRAINING_BATCH_SIZE = 64 
PRE_OPTIMIZATION_ITERATIONS = 300 # training_batch_size *N_training_iter  ~ pre_optimization_iterations
MODEL_UPDATE_ITERATIONS = 1000
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 50000
NUM_EPISODES = 5000

#
P_NORM = 2
LOSS_FUNCTION = F.smooth_l1_loss#F.mse_loss #F.smooth_l1_loss
REGULARIZATION_PARAMETER = 0.0
OPTIMIZER_TECHNIQUE = optim.Adam#optim.RMSprop
LEARNING_RATE = 1e-4
OPTIMIZER = lambda parameters: OPTIMIZER_TECHNIQUE(parameters , LEARNING_RATE, weight_decay=0)

#
SOLUTION_REWARD = 0
CONFLICT_REWARD = 1
INDETERMINATE_REWARD = -3

def GET_REWARD(conflict, state):
    if state == "NOT_TERMINAL":
        return INDETERMINATE_REWARD
    elif state == "SOLUTION":
        # return SOLUTION_REWARD + conflict * CONFLICT_REWARD
        return SOLUTION_REWARD
    elif state == "CONFLICT":
        # return conflict * CONFLICT_REWARD
        return conflict - abs(INDETERMINATE_REWARD)

# PATHS; include / at the end of the path
import sys
if "linux" in sys.platform:
    GIT_REPO_PATH = "/home/satgpupg/OvalSAT/"
    DATA_DIR = "/home/satgpupg/data/erdos_p_quarter/unsat/datatrain5_7/e3/"
    MODEL_PATH = None
else:
    GIT_REPO_PATH = "/Users/pgupta/Workspace/SATConversion/explore/graph_sat/"
    DATA_DIR = "./data10/"
    MODEL_PATH = None

VARIABLE_SELECTION_FILE = GIT_REPO_PATH + "variable_selection.txt"
RESULTS_FILE = GIT_REPO_PATH + "results.txt"
EXECUTABLE = GIT_REPO_PATH + 'minisat2/core/minisat_release'

# TEST_DIRS = ["/home/satgpupg/data/benchmarking_data/erdos/unsat/data4/",
#              "/home/satgpupg/data/benchmarking_data/erdos/unsat/datatest6/",
#              "/home/satgpupg/data/benchmarking_data/erdos/unsat/datatest5/"]
TEST_DIRS = []

import datetime
TIME_ID = datetime.datetime.now().strftime("%d_%b_%H_%M")


def get_hyperparameters_dict():
    return dict(
        MODEL = dict(
            EMBEDDING_DIMENSION = EMBEDDING_DIMENSION,
            EMBEDDING_UPDATE_ITERATIONS = EMBEDDING_UPDATE_ITERATIONS,
            EDGE_FEATURE_DIMENSION = EDGE_FEATURE_DIMENSION,
            NODE_FEATURE_DIMENSION = NODE_FEATURE_DIMENSION,
            GLOBAL_STATE_DIMENSION = GLOBAL_STATE_DIMENSION
            ),
        TRAINING = dict(
            TRAINING_MEMORY_CAPACITY = TRAINING_MEMORY_CAPACITY,
            VALIDATION_MEMORY_CAPACITY = VALIDATION_MEMORY_CAPACITY,
            TRAIN_VALIDATION_SPLIT_RATIO = TRAIN_VALIDATION_SPLIT_RATIO,
            VALIDATION_BATCH_SIZE = VALIDATION_BATCH_SIZE,
            N_TRAINING_ITER = N_TRAINING_ITER,
            TRAINING_BATCH_SIZE = TRAINING_BATCH_SIZE,
            PRE_OPTIMIZATION_ITERATIONS = PRE_OPTIMIZATION_ITERATIONS,
            GAMMA = GAMMA ,
            EPS_START = EPS_START,
            EPS_END = EPS_END,
            EPS_DECAY = EPS_DECAY,
            MODEL_UPDATE_ITERATIONS = MODEL_UPDATE_ITERATIONS,
            NUM_EPISODES = NUM_EPISODES,
            DATA_DIR = DATA_DIR,
            TEST_DIR = "\n".join(TEST_DIRS),
            ),
        REWARDS = dict(
                SOLUTION_REWARD = SOLUTION_REWARD,
                CONFLICT_REWARD = CONFLICT_REWARD,
                INDETERMINATE_REWARD = INDETERMINATE_REWARD
        ),
        LOSS = dict(
            LOSS_FN = LOSS_FUNCTION.__name__,
            OPTIMIZER=OPTIMIZER_TECHNIQUE.__name__,
            REGULARIZATION_PARAMETER = REGULARIZATION_PARAMETER,
            LEARNING_RATE = LEARNING_RATE
        ),
        FEATURES = dict(
                    NODE_BIAS = NODE_BIAS,
                    NODE_FEATURE_LIST = NODE_FEATURE_LIST,
        ),
        TIME = datetime.datetime.strftime(datetime.datetime.now(), "%d %B %H:%M")
    )