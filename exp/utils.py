import torch
from collections import defaultdict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def move_to_gpu(arg):
    if isinstance(arg, dict):
        return {k: move_to_gpu(v) for k, v in arg.items()}
    try:
        return arg.to(device)
    except:
        return arg


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d
