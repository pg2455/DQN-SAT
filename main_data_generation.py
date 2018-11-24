from hamiltonian_cnf import HamiltonianCNF
from get_graphs import GenerateRandomGraphs
from graph_sat import GraphSAT
import random, sys


def generate_barabasi_sats(iterations, n_max, solve = False):
    """
    Generate iterations of sats based on barabasi distribution.
    """
    for x in xrange(iterations):
        distinct_set = set()

        if x%100 == 0:
            print "Barabasi@iteration:", x

        # to generate distinct combinations
        while True:
            n = random.randint(3,n_max)
            m = random.randint(1,n-1)
            if not (n,m) in distinct_set:
                distinct_set.add((n,m))
                break

        graph = graph_generator.get_barabasi_graph(n,m)
        _ = sat_generator.get_hamiltonian_sat(graph, DATA_DIRECTORY + 'barabasi_n{}_m{}.dimacs'.format(n,m), solve)
    return True

def generate_erdos_sats(iterations, n_max, solve = False):
    """
    Generate iterations of sats based on erdos distribution
    """
    for x in xrange(iterations):
        if x%100 == 0:
            print "erdos@iteration:", x

        n = random.randint(3,n_max)
        p = random.random()
        graph = graph_generator.get_erdos_renyi_graph(n,p)
        _ = sat_generator.get_hamiltonian_sat(graph, DATA_DIRECTORY + 'erdos_n{}_p{}.dimacs'.format(n,round(p,2)), solve)

if __name__ == "__main__":
    ham_sat = HamiltonianCNF()
    graph_generator = GenerateRandomGraphs()
    sat_generator = GraphSAT()

    if len(sys.argv) > 1:
        DATA_DIRECTORY = sys.argv[1]
        DATA_DIRECTORY += "/" if DATA_DIRECTORY[-1] != '/' else ""
    else:
        DATA_DIRECTORY = "./data/"


    generate_barabasi_sats(20, n_max= 7)
    generate_erdos_sats(20, n_max=7)
