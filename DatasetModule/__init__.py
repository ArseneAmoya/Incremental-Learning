from .CUB_200 import CUB_200_2011, MyData

# from .transforms import *
import os 

__factory = {
    'cub': CUB_200_2011
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))

    if name not in __factory:

        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)