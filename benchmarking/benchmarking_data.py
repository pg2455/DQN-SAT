#! /usr/bin/env python2

import subprocess, os, sys
MINISAT_EXECUTABLE = "../minisat2/core/minisat_release"
MINISAT_OUTPUT = "../minisat_output.txt"
GLUCOSE_EXECUTABLE = "../../../glucose-syrup-4.1/simp/glucose_release"
GLUCOSE_OUTPUT = "../glucose_output.txt"

if "linux" in sys.platform:
    MINISAT_EXECUTABLE = "/home/satpg/OvalSAT/minisat2/core/minisat_release"
    MINISAT_OUTPUT = "/home/satpg/OvalSAT/benchmarking/minisat_output.txt"
    GLUCOSE_EXECUTABLE = "/home/satpg/OvalSAT/glucose-syrup-4.1/simp/glucose_release"
    GLUCOSE_OUTPUT = "/home/satpg/OvalSAT/benchmarking/glucose_output.txt"

def get_results(filename):
    # minisat
    _mpid = subprocess.Popen([MINISAT_EXECUTABLE, '-original',filename])
    _mpid.wait()
    output = open(MINISAT_OUTPUT).readlines()
    mc = "c MINISAT "+ ", ".join([x.strip() for x in output])

    # glucose
    _gpid = subprocess.Popen([GLUCOSE_EXECUTABLE,filename])
    _gpid.wait()
    output = open(GLUCOSE_OUTPUT).readlines()
    gc = "c GLUCOSE "+ ", ".join([x.strip() for x in output])

    return mc, gc


def main(CNF_DIRECTORY):
    for filename in os.listdir(CNF_DIRECTORY):
        mc, gc = get_results(CNF_DIRECTORY + filename)
        result = 'UNSAT' if "UNSATISFIABLE" in gc else "SAT" if "SATISFIABLE" in gc else "INDET"

        with open(CNF_DIRECTORY + filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            if "GLUCOSE" not in content:
                content = gc + '\n' + content
            if "MINISAT" not in content:
                content = mc + '\n' + content

            f.write(content)
        new_name = filename[:filename.find(".")] + "_" + result + ".dimacs" if result not in filename else filename
        os.rename(CNF_DIRECTORY + filename, CNF_DIRECTORY + new_name)



if __name__ == "__main__":
    CNF_DIRECTORY = "../server_data/"

    if len(sys.argv) > 1:
        CNF_DIRECTORY = sys.argv[1]
        CNF_DIRECTORY += "/" if CNF_DIRECTORY[-1] != '/' else ""

    main(CNF_DIRECTORY)
