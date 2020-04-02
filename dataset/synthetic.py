import numpy as np
from igraph import *
from collections import deque
import random as rd


def generate_plain_erdos(n=10000, lamda=10):
	return Graph.Erdos_Renyi(n=n, p=1.*lamda/n)

def generate_plain_sbm(n=10000, lamda=10, k=10, beta=0.25):
	# diagnomal entries will be p / (1 + beta) while non-diagonal entries will be {beta / (1+beta)} * {p / (k-1)}
	B            = int(1.*n/k)
	p            = 1.*lamda/B
	pref_matrix  = np.full((k, k), 1.*(beta / (1.+beta)) * (p / (k-1.)))
	np.fill_diagonal(pref_matrix, 1.*p/(1.+beta))
	block_sizes  = np.full((k,), B)
	return Graph.SBM(n=n, pref_matrix=pref_matrix.tolist(), block_sizes=block_sizes.tolist()) , \
		[block_id for block_id, block_size in enumerate(block_sizes) for _ in range(int(block_size))]


