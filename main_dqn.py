# /usr/bin/env python2
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

from utility_graph import UtilityGraph
from replay_memory import ReplayMemory, Transition
from Q_network import QNetwork
from cnf_compiler import CNFCompiler
from sat_solver_interface import SatSolverInterface
from event_handler import EventHandler
from proxy_configuration import *
from validator import Validator
from utils import *

# global counters
steps_done = 0 # for exploration and exploitation
n_optimizations = 0 # to update target model
hp_dict = get_hyperparameters_dict()
n_explorations, n_exploitations = 0,0

model = QNetwork(EMBEDDING_DIMENSION, EMBEDDING_UPDATE_ITERATIONS, NODE_FEATURE_DIMENSION, EDGE_FEATURE_DIMENSION, MODEL_PATH)
bellman_model = copy.deepcopy(model)

loss_fn = LOSS_FUNCTION
optimizer = OPTIMIZER(model.parameters())
memory = ReplayMemory(MEMORY_CAPACITY)
sat_solver_interface = SatSolverInterface(VARIABLE_SELECTION_FILE, RESULTS_FILE)
cnf_compiler = CNFCompiler()

def print_usage(i_episode):
    denominator = (1024*1024.0) if "linux" in sys.platform else (1024*1024*1024.0)
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / denominator
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss /denominator
    event_handler.log_metrics([('self_rss', self_rss), ('child_rss', child_rss)], index = i_episode)

    print "============================== MEMORY STATISTICS =============================="
    print "SELF RSS: ", self_rss
    print "CHILDREN RSS: ", child_rss
    all_objects = muppy.get_objects()
    print "Summary: \n", summary.print_(summary.summarize(all_objects))
    print "Memory Objects length: ", len(memory)
    print "Memory size: Object: {}, List: {}".format(get_size(memory), get_size(memory.memory))
    print "Size of models: M:{}, B: {} ".format(get_size(model), get_size(bellman_model))
    print "Size of memory's one element: {}".format(get_size(memory.memory[0]))
    print "GC COUNT: {}".format(gc.get_count())
    print "==============================================================================="

def select_action(state):
    global steps_done, n_explorations, n_exploitations
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        n_exploitations += 1
        new_embedding, q_values, delta = model(*convert_to_pytorch_variables(*state))
        values, indices = torch.max(q_values,1)
        action_value, row = values.max(0)
        variable = state[4][indices[row].data[0]] + 1 # since the variable = index + 1;

        event_handler.log_metrics([('norm_delta_embedding', matrix_norm(delta))], index = steps_done )
        if sample > 0.7:
            event_handler.plot_heatmap(q_values.data, 'action_#_Variable', "{}_{}_{}".format(steps_done, variable, row.data[0]), "q_values_in_the_middle" )

        return 1, new_embedding, variable, row.data[0]
    else:
        n_explorations += 1
        return 0, None, random.choice(state[4])+1, random.choice([0,1])

def optimize_model():
    global n_optimizations
    global bellman_model
    n_optimizations += 1

    if len(memory) < BATCH_SIZE:
        return None, None

    sum_loss, sum_next_state_value, sum_state_action_value = 0, 0, 0

    t1 = time.time()
    transitions = memory.sample(BATCH_SIZE)
    optimizer_sampling_time = time.time() - t1

    optimizer.zero_grad()
    for state, action, next_state, reward in transitions:
        t1 = time.time()
        state_action_value = model(*convert_to_pytorch_variables(*state))[1][action[1], state[4].index(action[0]-1)]
        model_computation_time = time.time() - t1

        if next_state[4] == []: # i.e. terminal state (not indeterminate since unassigned will be non-empty)
            next_state_value = Variable(torch.Tensor([0.0])) # only reward is needed
        else:
            next_state_value = bellman_model(*convert_to_pytorch_variables(*next_state))[1].max()

        sum_state_action_value += state_action_value.data[0]
        sum_next_state_value += next_state_value.data[0]
        expected_state_value = GAMMA * next_state_value + Variable(torch.Tensor([reward]))
        expected_state_value = expected_state_value.detach()

        loss = loss_fn(state_action_value, expected_state_value)
        sum_loss += loss.data[0]

        t1 = time.time()
        loss.backward(retain_graph = False) # backward() inside the loop doesn't need to hold graph
        loss_backward_time = time.time() - t1

        event_handler.update_metrics([('model_computation_time', model_computation_time),
            ('loss_backward_time', loss_backward_time)])

    # for param in model.parameters():
    #     param.grad.data.clamp_(-1,1)
    if REGULARIZATION_PARAMETER:
        lp_norm = get_lp_norm(model, p=P_NORM)
        lp_norm.backward(torch.Tensor([REGULARIZATION_PARAMETER]))

    t1 = time.time()
    optimizer.step()
    event_handler.log_metrics([('optimizer_step_time', time.time()- t1), ('optimizer_sampling_time', optimizer_sampling_time)], index = n_optimizations)

    event_handler.log_metrics([
    ('optimizer_avg_main_model_state_action_value', 1.0*sum_state_action_value/BATCH_SIZE),
    ('optimizer_avg_predicted_next_state_value', 1.0*sum_next_state_value/BATCH_SIZE),
    ], index = n_optimizations)

    return 1.0*sum_loss/BATCH_SIZE, 1.0*sum_next_state_value/BATCH_SIZE


def _optimize(i_episode):
    global steps_done, n_optimizations, n_exploitations, n_explorations

    # optimize the model
    if steps_done != 0  and steps_done % PRE_OPTIMIZATION_ITERATIONS == 0:
        print "Optimizing ... \n"
        t1 = time.time()
        loss, avg_next_state_value = optimize_model()
        event_handler.log_metrics([('total_optimization_time', time.time() - t1)], index = n_optimizations)

        if loss != None:
            norm = get_lp_norm(model, p=P_NORM).data[0]
            event_handler.log_metrics([ ('loss', loss),('norm', norm) ], index = n_optimizations)
            for name, param in model.named_parameters():
                event_handler.plot_heatmap(param.data,'episode', i_episode, name )
                if hasattr(param.grad, 'data') :
                    event_handler.plot_heatmap(param.grad.data,'episode', i_episode, name + "_gradient")

        if  n_optimizations %  MODEL_UPDATE_ITERATIONS == 0:
            bellman_model = copy.deepcopy(model)

        event_handler.log_metrics([('avg_memory_reward', memory.get_average_reward())], index = i_episode)        #bellman_model = copy.deepcopy(model)
        event_handler.update_metrics([('n_explorations', n_explorations), ('n_exploitations', n_exploitations)], difference = True)

def _validate(i_episode):
    if steps_done % PRE_OPTIMIZATION_ITERATIONS == 0 or i_episode % PRE_OPTIMIZATION_ITERATIONS == 0:
        t1 = time.time()
        validator.validate_model(model)
        event_handler.log_metrics([('validation_time', time.time() - t1)], index = n_optimizations)


def main():

    for i_episode in range(NUM_EPISODES):

        _validate(i_episode)

        with open(RESULTS_FILE, 'w') as f:
             f.write("")

        all_states = torch.zeros(EMBEDDING_DIMENSION,1)
        sum_reward = 0
        sat_start = False # it needs to run after taking first action

        # set the environment
        dimacs_filepath = get_dimacs_filename(DATA_DIR, ['barabasi'])
        clauses, n_variables, n_clauses = cnf_compiler.get_cnf_from_dimacs(dimacs_filepath)
        graph = UtilityGraph(n_variables, n_clauses, clauses, graph_type = 'cvig')


        for t in count():
            # for larger problems, solver might get stuck here; optimize and get better in decision taking
            _optimize(i_episode)

            # take action based on the state
            state = graph.get_current_state()
            embedding_updated, new_embedding, selected_variable, bool_value = select_action(state)
            if embedding_updated:
                graph.set_current_embedding(new_embedding.data)
                del new_embedding

            action = (selected_variable, bool_value)
            assert selected_variable > 0 and selected_variable <= n_variables, "Action not defined - New Assigned Variable {}".format(slected_variable if bool_value else -selected_variable)
            assert selected_variable - 1 not in state[5] and selected_variable - 1 in state[4], "Action already in state..."
            assert sat_solver_interface.assign_value(*action) == True

            # observe reward and next state
            if not sat_start:
                print 'starting {}\n'.format(i_episode)
                _pid = run_minisat(EXECUTABLE, dimacs_filepath)
                sat_start = True
            print t,
            reward, new_assignment, learnt_clauses, terminal_state = sat_solver_interface.get_results()

            # sometimes c++ writes gibberish values in results.txt; break and continue to new episode
            max_variable_check = max([0] + map(abs,new_assignment))
            if terminal_state == 'Stuck' or max_variable_check > n_variables:
                print "\nSolver stuck! Running new episode..."
                _pid.terminate()
                _pid.wait()
                event_handler.write_error(filename = 'error_log_{}.txt'.format(hp_dict['TIME']),
                                string = "#Episode: {} Terminal State: {}, Max Variable: {}, n_variables: {}\n".format(i_episode, terminal_state, max_variable_check, n_variables))
                break

            graph.update_state(new_assignment, learnt_clauses)
            next_state = graph.get_current_state()
            memory.push(state, action, next_state, reward)


            if t%LOGGING_AVERAGE == 0:
                event_handler.plot_heatmap(graph.get_current_embedding(), 'episode', i_episode, "Current Embedding")

            all_states = torch.cat([all_states, model.get_current_state(next_state[3]).expand(1,EMBEDDING_DIMENSION).permute(1,0)], dim = 1)
            sum_reward += reward
            del state,  next_state, action, reward

            if terminal_state or graph.is_satisfied():
                print "\nTerminal state reached- Result: {}, Total actions taken so far:{} ".format(terminal_state, steps_done)
                _pid.terminate()
                _pid.wait()
                print_usage(i_episode)

                # visdom
                # event_handler.plot_heatmap(graph.get_current_embedding(), 'episode', i_episode, "Current Embedding")
                event_handler.log_metrics([('best_n_steps', t+1)], index = i_episode)
                event_handler.update_metrics([('avg_steps', t+1), ('avg_reward', sum_reward)])
                event_handler.plot_heatmap(all_states, "episode", i_episode, "state_t")

                del graph, clauses,  embedding_updated
                gc.collect()
                break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        hp_dict['EXPERIMENT_NAME'] = sys.argv[1]
        hp_dict['AACOMMENT']  = sys.argv[2]

    event_handler = EventHandler(session_name = "{}_{}".format(hp_dict['TIME'], hp_dict['EXPERIMENT_NAME'] ))
    event_handler.make_a_text_window(string = get_configuration_text(hp_dict))
    validator = Validator(VALIDATION_DATA_DIR, N_VALIDATION, VARIABLE_SELECTION_FILE, RESULTS_FILE, GAMMA, event_handler)

    try:
        main()
    except KeyboardInterrupt:
        raise
    finally:
        with open("q_model_{}_{}".format(hp_dict['EXPERIMENT_NAME'], hp_dict['TIME']),'wb') as f:
            torch.save(model,f)
