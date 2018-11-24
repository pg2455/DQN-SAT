#! /usr/bin/env python2
from configuration import GET_REWARD
from pdb import set_trace as bp

class SatSolverInterface(object):
    """
    A dummy class that writes literal to be assigned to the file variable_selection_file . It also reads and interpret the
    results from the file results_file.

    Parameters:
    ______________
        None
    """

    def __init__(self, variable_selection_file_path, results_file_path):
        self.variable_selection_file = variable_selection_file_path
        self.results_file = results_file_path
        self.last_results_line = ''


    def assign_value(self, variable, bool_value):
        """
        Writes the literal to variable_selection_file which is read by MiniSAT solver to perform
        propagation.

        Parameters:
        ______________
            variable: int
                variable number to which the value needs to be assigned

            bool_value: bool
                value to assign
        Returns:
        ______________
            bool

        """
        vfile = open(self.variable_selection_file, 'r')
        status = 1 - int(vfile.readline().strip())
        assert status in [0,1]
        new_lit = variable if bool_value else -variable

        with open(self.variable_selection_file, 'w') as vfile:
            assert new_lit != 0, "Assignning 0 literal"
            vfile.write("{}\n{}\n".format(status, new_lit))
        return True

    def get_results(self):
        """
        Detects changes in results_file and returns reward, new assignments, learnt_clauses interpreted from the
        new addition to the file.

        Parameters:
        _______________
            None

        Returns:
        _______________
            reward: int
            learnt_clauses: list(list(int))
            new_assignments: list(int)
        """
        i = 0
        while True:
            with open(self.results_file, 'r') as f:
                rfile_lines = f.readlines()

            if len(rfile_lines) <= 1 or ('READING FILE' not in rfile_lines[-1] and rfile_lines[-1] != 'SOLVER CLOSING\n'):
                continue

            _ = rfile_lines.pop()
            last_line = rfile_lines[-1]

            i+=1
            if i  == 1000000:
                return None, [], None, "Stuck" # non-None values indicate error
                raise LookupError("Solver is not writing new results to the file. ")

            if last_line != self.last_results_line:
                #file has changed
                with open(self.results_file, 'r') as f:
                    lines = f.readlines()

                # lines = open(self.results_file,'r').readlines()
                new_lines = []

                while len(lines) >= 1 and lines[-1] != self.last_results_line:
                    new_lines += [lines.pop().strip()]

                self.last_results_line = last_line
                break

        return self.get_reward_and_new_state(new_lines[::-1])

    def get_reward_and_new_state(self, new_lines):
        """
        Analyzes the state change (new_lines) and returns learnt_clauses(if any), new_assignment, and
        reward related to the state change

        Parameters:
        ____________
            new_lines: list
                each element represent change in state due to previous assignment

        Returns:
        ____________
            reward: int
                reward due to state change

            new_assignment: list(int)
                list of literals

            learnt_clauses: list(list(int))
                list of clauses which is again represented as list of literals

            solver_result: string
                to indicate terminal state
        """
        conflict = 0
        learnt_clauses = []
        solver_result = ''
        new_assignment = []

        # new assignment
        for line in new_lines[::-1]:
            if 'DECISION' in line:
                new_assignment = map(int,line[line.find('ASSIGNMENT ') + len('ASSIGNMENT '): ].strip().split())
                break

        for line in new_lines:
            if 'CONFLICT' in line:
                conflict += 1
                learnt_clauses.append(map(int,line[line.find("CLAUSE ")+ len("CLAUSE ") : line.find("BACKTRACK")].strip().split()))

            if line == 'SAT' or line == 'UNSAT' or line == 'INDET':
                solver_result = line

        # print 'CONFLICT: ', conflict , "solver_result: ", solver_result, 'learnt_clauses: ', learnt_clauses

        if (conflict == 0)  and (solver_result == ""):
            reward = GET_REWARD(conflict, state = 'NOT_TERMINAL')
        elif solver_result in ['SAT', 'UNSAT']:
            self.last_results_line = ''
            reward = GET_REWARD(conflict, state = 'SOLUTION')
            
        else:
            reward = GET_REWARD(conflict, state = 'CONFLICT')


        return reward, new_assignment, learnt_clauses, solver_result