#! /usr/bin/env python2
import random, torch, gc, os
from itertools import count

from utility_graph import UtilityGraph
from cnf_compiler import CNFCompiler
from utils import convert_to_pytorch_variables, run_minisat, get_dimacs_filename
from sat_solver_interface import SatSolverInterface
from proxy_configuration import *
if '/home/satgpupg/' not in sys.path: sys.path.append('/home/satgpupg/') 
from graph_template import Graph

class Validator(object):

    def __init__(self,VALIDATION_DATA_DIR, N_VALIDATION, VARIABLE_SELECTION_FILE, RESULTS_FILE, GAMMA, event_handler):
        self.N_VALIDATION = N_VALIDATION
        self.cnf_compiler  = CNFCompiler()
        self.solver = SatSolverInterface(VARIABLE_SELECTION_FILE, RESULTS_FILE)
        self.event_handler = event_handler
        self.counter = 0
        self.data_dir = VALIDATION_DATA_DIR
        self.GAMMA = GAMMA

    def select_action(self, model, state):
        sample = random.random()

        new_embedding, q_values, _= model(*convert_to_pytorch_variables(*state))

        values, indices = torch.max(q_values,1)
        action_value, row = values.max(0)
        variable = state[4][indices[row].data[0]] + 1 # since the variable = index + 1;
        return new_embedding, variable, row.data[0], action_value

    def validate_model(self, model):
        if os.listdir(self.data_dir) == []:
            return None

        self.counter += 1
        sum_reward, steps, avg_discounted_action_value = 0, 0, 0

        for i_episode in range(self.N_VALIDATION):
            with open(RESULTS_FILE, 'w') as f:
                 f.write("")

            sat_start = False # it needs to run after taking first action
            discounted_action_value = 0
            # set the environment
            dimacs_filepath = get_dimacs_filename(self.data_dir, ['barabasi'])
            clauses, n_variables, n_clauses = self.cnf_compiler.get_cnf_from_dimacs(dimacs_filepath)
            graph = Graph(n_variables, clauses)

            for t in count():
                # take action based on the state
                state = graph.get_current_state()
                new_embedding, selected_variable, bool_value, action_value = self.select_action(model, state)
                graph.set_current_embedding(new_embedding.data)
                del new_embedding

                action = (selected_variable, bool_value)
                assert selected_variable > 0 and selected_variable <= n_variables, "Action not defined - New Assigned Variable {}".format(slected_variable if bool_value else -selected_variable)
                assert selected_variable - 1 not in state[5] and selected_variable - 1 in state[4], "Action already in state..."
                assert self.solver.assign_value(*action) == True

                # observe reward and next state
                if not sat_start:
                    print '@VALIDATION starting {}\n'.format(i_episode)
                    _pid = run_minisat(EXECUTABLE, dimacs_filepath)
                    sat_start = True
                print t,
                reward, new_assignment, learnt_clauses, terminal_state = self.solver.get_results()

                # sometimes c++ writes gibberish values in results.txt; break and continue to new episode
                max_variable_check = max([0] + map(abs,new_assignment))
                if terminal_state == 'Stuck' or max_variable_check > n_variables:
                    print "\nSolver stuck! Running new episode..."
                    _pid.terminate()
                    _pid.wait()
                    break

                graph.update_state(new_assignment, learnt_clauses)
                next_state = graph.get_current_state()

                sum_reward += reward
                discounted_action_value += self.GAMMA*action_value + reward
                del state,  next_state, action, reward


                if terminal_state or graph.is_satisfied():
                    print "\n@VALIDATION: Terminal state reached- Result: {} ".format(terminal_state)
                    _pid.terminate()
                    _pid.wait()

                    steps += t+1
                    del graph, clauses
                    avg_discounted_action_value += 1.0*discounted_action_value/(steps)
                    gc.collect()
                    break


        self.event_handler.log_metrics([
            ("validation_steps_per_problem", 1.0*steps/self.N_VALIDATION), ("validation_reward_per_problem", 1.0*sum_reward/self.N_VALIDATION),
        ("validation_avg_discounted_action_value",1.0*avg_discounted_action_value/self.N_VALIDATION )], index = self.counter)