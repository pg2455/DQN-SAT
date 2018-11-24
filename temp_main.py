# %load /home/satgpupg/OvalSAT/main_dqn.py
import sys
if '/home/satgpupg/OvalSAT/' not in sys.path: sys.path.append('/home/satgpupg/OvalSAT/') 
if '/home/satgpupg/' not in sys.path: sys.path.append('/home/satgpupg/') 
import math, random, os, copy, datetime, gc, time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
import sys, getopt
import resource
from pdb import set_trace as bp
from pympler import summary, muppy, asizeof


from replay_memory import ReplayMemory, Transition
from cnf_compiler import CNFCompiler
from sat_solver_interface import SatSolverInterface
from event_handler import EventHandler
from proxy_configuration import *
from validator import Validator
from utils import *
from Q_network import QNetwork

from graph_template import Graph
# from utility_graph import UtilityGraph

TRAIN_VALIDATION_SPLIT_RATIO = 0.1

class Environment(object):
    
    def __init__(self):
        self.sat_solver_interface = SatSolverInterface(VARIABLE_SELECTION_FILE, RESULTS_FILE)
        self.i_episode = 0
    
    def step(self, action, embedding_updated = False, new_embedding = None):
        """
        action: variable, bool_value
        """
        # take action based on the state
        
        # if embeddings are updates right after .update_state(); there needs to be a separate message passing module
        if embedding_updated:
            self.graph.set_current_embedding(new_embedding.data)
            del new_embedding

        assert self.sat_solver_interface.assign_value(*action) == True
        
        if self.ready:
                        # run the sat solver
            print 'starting {}\n'.format(self.i_episode)
            self._pid = run_minisat(EXECUTABLE, filename, original = False)
            self.ready = False

        reward, new_assignment, learnt_clauses, terminal_state = self.sat_solver_interface.get_results()
        
        
        self.graph.update_state(new_assignment, learnt_clauses)
        done = terminal_state or self.graph.is_satisfied()
        if done:
            self._pid.terminate()
            self._pid.wait()
            #print_usage(self.i_episode)
        return self.graph.get_current_state(), reward, done
            
    def reset(self, filename):
        self.i_episode += 1
        with open(RESULTS_FILE, 'w') as f:
            f.write("")

        clauses, n_variables, n_clauses = cnf_compiler.get_cnf_from_dimacs(filename)
        self.graph = Graph(n_variables, clauses)
        
        self.ready = True



        return self.graph.get_current_state()

class Agent(object):
    def __init__(self, model):
        self.steps_done = 0
        self.model = model
        self.n_exploitations, self.n_explorations = 0, 0
    
    def choose_action(self, state, model = False):
        # implement the exploration policy here
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold or model :
            self.n_exploitations += 1 if not model else 0
            new_embedding, q_values, delta = self.model(*convert_to_pytorch_variables(*state))
            values, indices = torch.max(q_values,1)
            action_value, row = values.max(0)
            variable = state[4][indices[row].data[0]] + 1 # since the variable = index + 1;

#             event_handler.log_metrics([('norm_delta_embedding', matrix_norm(delta))], index = steps_done )
#             if sample > 0.7:
#                 event_handler.plot_heatmap(q_values.data, 'action_#_Variable', "{}_{}_{}".format(steps_done, variable, row.data[0]), "q_values_in_the_middle" )

            return 1, new_embedding, (variable, row.data[0])
        else:
            self.n_explorations += 1
            return 0, None, (random.choice(state[4])+1, random.choice([0,1]))



from pdb import set_trace as bp
def compute_loss(sample, backward = False):
    loss_o, loss_th = 0,0 
    for state,action,next_state,reward in sample:
        state_action_value = model(*convert_to_pytorch_variables(*state))[1][action[1], state[4].index(action[0]-1)]
        
        if next_state[4] == []: # i.e. terminal state (not indeterminate since unassigned will be non-empty)
            next_state_value = Variable(torch.Tensor([0.0])) # only reward is needed
        else:
            next_state_value = target_model(*convert_to_pytorch_variables(*next_state))[1].max()
        expected_state_value = GAMMA * next_state_value + Variable(torch.Tensor([reward]))
        expected_state_value = expected_state_value.detach()

        loss = loss_fn(state_action_value, expected_state_value)
        
        loss_o += loss.item()
        
        _a = model(*convert_to_pytorch_variables(*next_state))[1].max()
        loss_th +=  loss_fn(state_action_value, reward + GAMMA * _a).item()

        if backward:        
            loss.backward(retain_graph = False) # backward() inside the loop doesn't need to hold graph

    return loss_th/len(sample), loss_o/len(sample)

def optimize(sample):
    optimizer.zero_grad()
    loss_o, loss_th = compute_loss(sample, backward = True)
    
    if REGULARIZATION_PARAMETER:
        lp_norm = get_lp_norm(model, p=P_NORM)
        lp_norm.backward(torch.Tensor([REGULARIZATION_PARAMETER]))

    optimizer.step()
    
    
    
def validate(sample):
    loss_o, loss_th = compute_loss(sample, backward = False)

def test():
    for filename in os.listdir(TEST_DIR):
        state = env.reset(TEST_DIR + filename)
        done = False
        for t in count():
            embedding_updated, new_embedding, action = agent.choose_action(state, model = True)
            next_state, reward, done = env.step(action, embedding_updated, new_embedding)
            state = next_state

            if done:
                break


if __name__ == "__main__":
    # global counters
    steps_done = 0 # for exploration and exploitation
    n_optimizations = 0 # to update target model
    hp_dict = get_hyperparameters_dict()
    n_explorations, n_exploitations = 0,0

    model = QNetwork(EMBEDDING_DIMENSION, EMBEDDING_UPDATE_ITERATIONS, NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MODEL_PATH)
    target_model = copy.deepcopy(model)

    loss_fn = LOSS_FUNCTION
    optimizer = OPTIMIZER(model.parameters())
    memory = ReplayMemory(MEMORY_CAPACITY)

    cnf_compiler = CNFCompiler()

    env = Environment()
    agent = Agent(model)

    # dataset generation and training (using exploration)
    training_memory = ReplayMemory(1000)
    validation_memory = ReplayMemory(100)
    optimization_count = 0 
    for x in range(100):
        for filename in os.listdir(DATA_DIR):
            state = env.reset(DATA_DIR + filename)
            done = False
            for t in count():
                embedding_updated, new_embedding, action = agent.choose_action(state)
                next_state, reward, done = env.step(action, embedding_updated, new_embedding)
                training_memory.push(state, action, next_state, reward)
                print reward
                state = next_state
                
                if done:
                    break
            
                    if len(training_memory.memory) % PRE_OPTIMIZATION_ITERATIONS == 0 and len(training_memory.memory) > BATCH_SIZE:
                
                        # train-validation split
                        if len(validation_memory.memory) < validation_memory.capacity:
                            for i in training_memory.memory:
                                if random.random() < TRAIN_VALIDATION_SPLIT_RATIO:
                                    validation_memory.push(i)
                                    training_memory.memory.pop(i)
                        
                        optimize(training_memory.sample(TRAINING_BATCH_SIZE))
                        validate(validation_memory.sample(VALIDATION_BATCH_SIZE))
                
                        optimization_count += 1
                        if optimization_count % MODEL_UPDATE_ITERATIONS == 0:
                            test()
                            target_model = copy.deepcopy(model)
