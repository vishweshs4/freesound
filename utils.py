from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *

from pathlib import Path
import pandas as pd
import numpy as np

def path2fn(path): return path.name
def top_3_accuracy(preds, targs): return top_k_accuracy(preds, targs, 3)

def name(n=10, print_it=True):
    name = "".join(random.choice(string.ascii_lowercase) for _ in range(n))
    if print_it: print(name)
    return name
