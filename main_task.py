import sys
sys.path.insert(0, './dataset/')
import synthetic
sys.path.insert(0, './sampling_algo/')
import baselines
import gts


from sklearn import linear_model
import numpy as np
import random as rd
import igraph as ig
import pandas as pd
import pickle as pk


#### Standard community task implementation

def pred_random(sample_X, add_X):
    return rd.random()

lr = linear_model.LinearRegression()
def pred_linear(sample_X, add_X):
    # predicts new_distance - old distance
    return -1.*lr.predict([sample_X+add_X])[0]

n              = 2000
lamda          = 10
beta           = 0.25
k              = 50
sample_percent = 10


# loading the graph
g, communities = synthetic.generate_plain_sbm(n=n, lamda=lamda, k=k, beta=beta)
g = g.components().giant()
g.vs['id'] = [v.index for v in g.vs]
g.vs['tp'] = communities
n = g.vcount()
print(g.summary())

# distance function 
def eval_community_coverage(g, orig_g):
    return 1. - 1. * len(set(g.vs['tp'])) / len((set(orig_g.vs['tp'])))



# training the prediction sampler for community detection task
# the training code will be released on the acceptance of the paper

# load the pre-trained model
lr = pk.load(open('./sampling_algo/pretrained_gts_context_sbm_communitycoverage.pk', 'rb'))

sample_size  = 1.*sample_percent/100*n
seed_node    = np.random.randint(0, n)


# testing on one simulation
print ('bfs', eval_community_coverage(g.subgraph(baselines.breadth_first_walk(g, seed_node, sample_size)), g))
print ('dfs', eval_community_coverage(g.subgraph(baselines.depth_first_walk(g, seed_node, sample_size)), g))
print ('rw', eval_community_coverage(g.subgraph(baselines.random_walk(g, seed_node, sample_size)), g))
print ('mhrw', eval_community_coverage(g.subgraph(baselines.metropolis_random_walk(g, seed_node, sample_size)), g))
print ('ff', eval_community_coverage(g.subgraph(baselines.forest_fire_walk(g, seed_node, sample_size)), g))
print ('rd', eval_community_coverage(g.subgraph(baselines.rank_degree_walk(g, seed_node, sample_size)), g))
print ('xs', eval_community_coverage(g.subgraph(baselines.expansion_walk(g, seed_node, sample_size)), g))

init_samples = set([seed_node])
X, Y, P, pred_samples = gts.prediction_sampler(g, init_samples, feature_fn=gts.generalization_features, eval_fn=eval_community_coverage, \
                                                  prediction_fn=pred_linear, sample_size=sample_size, verbose=False)

print ('gts', eval_community_coverage(g.subgraph(pred_samples), g))


