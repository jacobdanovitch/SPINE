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

top_k_words = []
zeros = 0.0
threshold = 0.001
h_dim = None
total = None
vectors = {}
num = 5
width = 10
emb = None

def top_k_contrib(k, dim):
    topk = dim.nlargest(k)
    return list(zip(topk.values, topk.index.tolist()))


def load_vectors(filename):
    global vectors, dimensions, zeros, h_dim, total, top_k_words, emb
    
    emb = pd.read_csv(filename, sep=" ", header=None).set_index(0)
    vectors = dict(zip(emb.index, emb.values.tolist()))
    
    get_topk = partial(top_k_contrib, width)
    top_k_words = emb.apply(get_topk, axis=0).values.tolist()
    
    print ("Sparsity =", (100. * (emb.values <= threshold).sum()) / emb.count().sum())
    total = len(vectors)
    print ('done loading vectors')


def find_top_participating_dimensions(word, k):
    if word not in emb.index:
        print('word not found')
        return

    dims = emb.loc[word].values.argsort()[::-1][:k]
    vals = emb.loc[word, dims]

    print ("Word of interest = ", word)
    print (" -----------------------------------------------------")
    for i, j in zip(vals, dims):
        print ("The contribution of the word '%s' in dimension %d = %f" %(word, j, i))
        print ('Following are the top words in dimension', j, 'along with their contributions')
        for (v, w) in top_k_words[j]:
            print(f"\t{w} = {v}")
        print("\n")
    return