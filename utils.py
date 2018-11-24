#! /usr/bin/env python2
import resource, torch
import subprocess, sys, os, random
from torch.autograd import Variable
from pympler import summary, muppy, asizeof
from numpy.linalg import norm

def _cuda(variable, use_cuda = True):
    if use_cuda and torch.cuda.is_available():
        return variable.cuda()
    return variable

def run_minisat(EXECUTABLE, filename, original = False):
    if not original:
        return subprocess.Popen([EXECUTABLE, '-no-original',filename])
    return subprocess.Popen([EXECUTABLE, filename])

def convert_to_pytorch_variables(*args):
    return tuple([Variable(x) if type(x) != list else x for x in args])

def get_lp_norm(model, p):
    # returns Variable()
    return sum([parameter.norm(p) for name, parameter in model.named_parameters() if parameter.requires_grad])

def get_configuration_text(hp_dict):
    html = ""
    for cat, _dict in sorted(hp_dict.items(), key = lambda x:x[0]):
        html += "<h3>{}</h3>:<br>".format(cat)
        if type(_dict)  == dict:
            html += "\n".join(["<b>{}</b> = {}<br>".format(k,v) for k,v in _dict.items()])
        if type(_dict) == str:
            html += _dict + "<br>"
    return html

def get_size(obj, units = 'm'):
    if units == 'm':
        return asizeof.asizeof(obj)/(1024*1024.0)
    if units == 'g':
        return asizeof.asizeof(obj)/(1024*1024*1024.0)
    return asizeof.asizeof(obj)

def get_dimacs_filename(data_dir, filetype):
    """
    Fetches the filepath of the file with specific parameters.

    Parameters:
    _______________
        data_dir : str
            directory in which to search for the file

        filetype: list(str)
            each element is a requirement for the filename to start with
    """
    while True:
        filename = random.choice(os.listdir(data_dir))
        if sum([filename.startswith(x) for x in filetype]) != 0:
            return data_dir + filename

def matrix_norm(matrix):
    return norm(matrix, 1)
