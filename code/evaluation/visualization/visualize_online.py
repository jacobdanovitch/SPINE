from multiprocessing import Queue
from queue import PriorityQueue
from numpy import linalg as LA
from scipy import stats
import numpy as np
import pandas as pd
import argparse
import time
import sys
import os
from tqdm import tqdm
from functools import partial
import warnings

top_k_words = []
threshold = 0.001
total = None
width = 10
vectors = None

def top_k_contrib(k, dim):
    topk = dim.nlargest(k)
    return list(zip(topk.values, topk.index.tolist()))


def load_vectors(filename):
    global vectors, dimensions, total, top_k_words
    
    vectors = pd.read_csv(filename, sep="\s", engine='python', header=None, index_col=0)
    if vectors.isnull().values.any():
        warnings.warn("Your vectors contain null inputs. Replacing with 0.")
        vectors = vectors.fillna(0)
    
    get_topk = partial(top_k_contrib, width)
    top_k_words = vectors.apply(get_topk, axis=0).values.tolist()
    
    print ("Sparsity =", (100. * (vectors.values <= threshold).sum()) / vectors.count().sum())
    total = len(vectors)
    print ('done loading vectors')


def find_top_participating_dimensions(word, k):
    if word not in vectors.index:
        print('word not found')
        return

    dims = vectors.loc[word].values.argsort()[::-1][:k]
    vals = vectors.loc[word].values[dims]

    

    print ("Word of interest = ", word)
    print (" -----------------------------------------------------")
    for i, j in zip(vals, dims):
        print ("The contribution of the word '%s' in dimension %d = %f" %(word, j, i))
        print ('Following are the top words in dimension', j, 'along with their contributions')
        for (v, w) in top_k_words[j]:
            print(f"\t{w} = {v}")
        print("\n")
    return