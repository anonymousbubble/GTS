import numpy as np
import igraph as ig
import random as rd


# distance function for community coverage
def eval_community_coverage(g, orig_g):
    return 1. * len(set(g.vs['tp'])) / len((set(orig_g.vs['tp'])))


# sampling features

def generalization_features(g=None, nodes=None, kind='add_node'):
    fts = []
    if kind == 'sample':
        gi = g.subgraph(nodes)
        fts.append(np.mean(gi.degree()))
        fts.append(gi.transitivity_avglocal_undirected(mode='zero'))
        fts.append(gi.vcount())

    elif kind == 'add_node':
        fts.append(g.vs[nodes].degree())
        fts.append(len([1 for n in g.vs[nodes].neighbors() if n['sampled']]))
        fts.append(1./ g.vs[nodes].degree()* len([1 for n in g.vs[nodes].neighbors() if n['sampled']]))
        fts.append(np.max([n['timestamp'] for n in g.vs[nodes].neighbors() if n['sampled']])) 
        fts.append(g.vs[nodes]['layer']) 
        fts.append(g.vs[nodes]['dlayer'])
        fts.append(len([1 for n in g.vs[nodes].neighbors() if not n['sampled'] and not n['frontier']])) 
        fts.append(rd.random())

    elif kind == 'names':
        return ['g_deg', 'g_cc', 'g_num', 'n_deg', 'n_indeg', 'n_normdeg', 'n_time', 'n_lyr', 'n_dlyr', 'n_degxs', 'n_rd' ]

    return fts


# maximum prediction based sampler (no exploration)
def prediction_sampler(g, init_sample, feature_fn, eval_fn, prediction_fn, sample_size=None, reservoir_size=1000, recording=True, verbose=True):

    # init
    X, Y, P, front_ctr = [], [], [], 0

    sampled_set   = init_sample.copy()

    g.vs['sampled'], g.vs['frontier']   = 0, 0
    g.vs['timestamp'], g.vs['layer'], g.vs['dlayer']    = np.inf, np.nan, np.nan
    for i_sample in init_sample:
        g.vs[i_sample]['sampled'] = 1
        g.vs[i_sample]['layer']   = 0
        g.vs[i_sample]['dlayer']   = 0
        g.vs[i_sample]['timestamp']= len(init_sample)
        for n in g.vs[i_sample].neighbors():
            if not n['sampled']: n['dlayer'] = 1

    while len(sampled_set) < sample_size:
        if verbose: print (len(sampled_set))
        sample_X = feature_fn(g, sampled_set, 'sample')
        add_Yo  = eval_fn(g.subgraph(sampled_set), g)

        frontier_set = set()
        for s_sample in sampled_set:
            for n in g.vs[s_sample].neighbors():
                if not n['sampled']:
                    frontier_set.add(n['id'])
                    n['frontier'] = 1
                    if np.isnan(n['layer']):
                        n['layer'] = g.vs[s_sample]['layer']+1


        # find the best predicted action
        top_node, top_val = np.inf, -1.*np.inf
        for frontier_node in frontier_set:
            add_node = frontier_node
            add_X    = feature_fn(g, add_node, 'add_node')
            add_P    = prediction_fn(sample_X, add_X)

            if len(X) < reservoir_size:
                add_Y    = eval_fn(g.subgraph(sampled_set|set([add_node])), g) - add_Yo
                if not np.isinf(add_Y) and not np.isnan(add_Y):
                    X.append(sample_X+ add_X)
                    Y.append(add_Y)
                    P.append(add_P)
            else:
                prob_new = 1.*reservoir_size/(front_ctr+1)
                if rd.random() < prob_new:
                    rand_idx = rd.randint(0, reservoir_size-1)
                    add_Y    = eval_fn(g.subgraph(sampled_set|set([add_node])), g) - add_Yo
                    if not np.isinf(add_Y) and not np.isnan(add_Y):
                        X[rand_idx], Y[rand_idx], P[rand_idx] = sample_X+ add_X, add_Y, add_P

            front_ctr+=1

            if add_P > top_val:
                top_val  = add_P
                top_node = add_node


        # update the sample
        sampled_set.add(top_node); g.vs[top_node]['sampled'] = 1; g.vs[top_node]['frontier'] = 0; g.vs[top_node]['timestamp'] = len(sampled_set)
        for n in g.vs[top_node].neighbors():
            if not n['sampled']:
                n['dlayer'] = g.vs[top_node]['dlayer']+1

    return np.array(X), np.array(Y), np.array(P), sampled_set

