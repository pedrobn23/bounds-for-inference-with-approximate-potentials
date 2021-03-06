import collections
import logging
import os
import pprint
import difflib

from pgmpy import readwrite
from pgmpy import models


def _is_bayesian(path: str) -> bool:
    """Check if a .uai network is bayesian.
    A .uai network can be bayesian o markovian. This method
    check whether the network is bayesian.
    Args:
        path: path to the file to he open.
    
    Raises:
        OSError: if there is any problem opening the file in 
            the provided path.
    """
    try:
        with open(path, 'rb') as net_file:

            # the first line in a UAI file contains type
            net_type = net_file.readline().strip()
            ret = net_type == b'BAYES'
            net_file.close()

    except OSError as ose:
        raise OSError(f'Error ocurred reading network file {path!r}') from ose

    return ret


def read(path: str) -> models.BayesianModel:
    """
    Read a bayesian network from a file.
    Read method uses pgmpy to read a bayesian network from either
    BIF file or UAI file.
    Args:
        path: path to the file to he open.
    Raises:
        OSError: if there is any problem opening the file in 
            the provided path.
        ValueError: if file provided is not supported by read.
    """

    try:
        if path.endswith(".uai"):
            if not _is_bayesian(path):
                raise ValueError(f'network in {path!r} is not BAYES.' +
                                 f' Only networks of type BAYES are allowed.')

            reader = readwrite.UAIReader(path=path)

        elif path.endswith(".bif"):
            reader = readwrite.BIFReader(path=path)
        else:
            raise ValueError(
                f'File extension for path {path!r} is not supported.')

    except OSError as ose:
        raise OSError(f'Error ocurred reading network file {path!r}') from ose

    return reader

def nets(path, networks):
    nets = []
    for net in networks:
        print( f' --> reading new net: {os.path.join(path, net)}.')
        if net.endswith('.bif'):
            fullpath = os.path.join(path, net)
            reader = read(fullpath)
            model = reader.get_model()
            nets.append((net, model.get_cpds()))
            
        else:
            raise ValueError(f'Only .bif nets are accepted, got : {net}')

    return nets
