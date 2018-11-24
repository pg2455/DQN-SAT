#! /usr/bin/env python2

class CNFCompiler(object):
    """
    converts to and fro cnf string to clauses i.e. list

    Parameters:
    ______________
        None
    """
    def __init__(self):
        pass

    def get_cnf_from_dimacs(self, dimacs_file):
        """
        Get the CNF clauses from the dimacs file.

        Parameters:
        _______________
            dimacs_file: str
                file path of the dimacs file. Look online for DIMACS standard.

        Returns:
        _______________
            clauses: list(list)
                each list contains integers.
        """
        with open(dimacs_file, 'r') as f:
            text = f.read().split('\n')

        for line in text:
            if line[0] == 'p':
                n_variables, n_clauses = int(line.split()[2]), int(line.split()[3])
                break

        cnf = []
        for line in text:
            if line!= '' and line[0] not in ('c', 'p') :
                cnf.append([int(n) for n in line.split()][:-1])
        return (cnf, n_variables, n_clauses)

    def get_dimacs_string_from_cnf(self, clauses, n_variables, save_file = None, comments = []):
        """
        Get the string of all the clauses that will go into the dimac file.

        Parameters:
        _______________
            clauses:list(list)
                each list i.e. clause is, as per the DIMACS standard, disjunction of variables in it

            n_variables: int
                number of variables in the cnf

            save_file: str
                file path where the dimacs string to be saved. if none it simply returns the string.

            comments: list(str)
                each string goes in a comment line starting with c
        """
        dimacs_string = ""
        dimacs_string += "p cnf {0} {1}\n".format(n_variables, len(clauses))
        for comment in comments:
            dimacs_string += "c {}\n".format(comment)

        for clause in clauses:
            dimacs_string += " ".join(map(str,clause)) + " 0\n"

        if save_file:
            with open(save_file, 'w') as f:
                f.write(dimacs_string)
            # open(save_file, 'w').write(dimacs_string)
        return dimacs_string
