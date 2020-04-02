# basic exploration samplers
import random as rd
from collections import deque
import numpy as np
import operator

# correct the bias to form RWRW
def random_walk(g, seed_node, sample_size):
	sampled_nodes = set()
	current = g.vs[seed_node]

	while len(sampled_nodes) < sample_size:
		sampled_nodes.add(current['id'])
		current = rd.choice(current.neighbors())

	return sampled_nodes

# rw with rejump and restarts
def random_walk_restart_rejump(g, seed_node, sample_size, restart=0., rejump=0.):
	sampled_nodes = set()
	current_seed  = seed_node
	current = g.vs[current_seed]
	sampled_nodes.add(current['id'])
	N       = g.vcount()

	while len(sampled_nodes) < sample_size:
		coin = rd.random()
		if coin < restart: # restarts
			current = g.vs[current_seed]
		elif coin < restart+rejump: # rejumps
			current_seed = rd.randint(0, N-1)
			current = g.vs[current_seed]
		else: # walks
			current = rd.choice(current.neighbors())
		sampled_nodes.add(current['id'])

	return sampled_nodes


def metropolis_random_walk(g, seed_node, sample_size):
	sampled_nodes = set()
	current = g.vs[seed_node]

	while len(sampled_nodes) < sample_size:
		sampled_nodes.add(current['id'])
		next = rd.choice(current.neighbors())
		current = next if rd.random()<1.*current.degree()/next.degree() else current

	return sampled_nodes

def breadth_first_walk(g, seed_node, sample_size):
	sampled_nodes = set()
	current = g.vs[seed_node]
	queue   = deque([current['id']])
	while len(sampled_nodes) < sample_size:
		top = queue.popleft()
		sampled_nodes.add(top)
		queue.extend([n['id'] for n in g.vs[top].neighbors() if n['id'] not in sampled_nodes])
	return sampled_nodes

def depth_first_walk(g, seed_node, sample_size):
	sampled_nodes = set()
	current = g.vs[seed_node]
	stack   = [current['id']]
	visited_nodes = set([seed_node])
	while len(sampled_nodes) < sample_size:
		top = stack.pop()
		sampled_nodes.add(top)
		for n in g.vs[top].neighbors():
			if n['id'] not in visited_nodes:
				stack.append(n['id'])
				visited_nodes.add(n['id'])
	return sampled_nodes

def forest_fire_walk(g, seed_node, sample_size, p=0.7):
	sampled_nodes = set()
	current = g.vs[seed_node]
	queue   = deque([current['id']])
	while len(sampled_nodes) < sample_size:
		if not queue:
			queue.append(rd.choice(g.vs)['id'])
		top = queue.popleft()
		sampled_nodes.add(top)
		x   = np.random.geometric(p)
		extensions = [n['id'] for n in g.vs[top].neighbors() if n['id'] not in sampled_nodes]
		queue.extend(rd.sample(extensions, min(len(extensions), int(x)) ))
	return sampled_nodes



def rank_degree_walk(g, seed_node, sample_size, p=0.01, rho=0.0):
	sampled_nodes, seed_nodes= set(), set([seed_node])
	seed_nodes |= set([v['id'] for v in np.random.choice(g.vs, size=min(int(sample_size), int(p*g.vcount())-1), replace=False)])
	sampled_nodes |= seed_nodes

	# init
	seed_nodes = deque(seed_nodes)

	while len(sampled_nodes) < sample_size:
		if not seed_nodes: seed_nodes.extend([v['id'] for v in np.random.choice(g.vs, size=min(int(sample_size), int(p*g.vcount())-1), replace=False)])
		selected_seed_node = seed_nodes.popleft()
		ngb_list           = [(n['id'], n.degree()) for n in g.vs[selected_seed_node].neighbors()]
		rd.shuffle(ngb_list)
		ngb_list.sort(key=operator.itemgetter(1), reverse=True)
		seed_nodes.extend([n[0] for n in ngb_list[:max(1, int(rho*g.vs[selected_seed_node].degree()))] if n[0] not in sampled_nodes])
		sampled_nodes.add(selected_seed_node)
	return sampled_nodes


def expansion_walk(g, seed_node, sample_size, coverage=False):
	sampled_nodes, new_changes, frontier_nodes_val = set(), set(), dict()
	g.vs['sampled'], g.vs['frontier'] = 0, 0

	current = g.vs[seed_node]
	frontier_nodes_val[current['id']]=0; new_changes.add(current['id']); g.vs[current['id']]['frontier'] = 1
	while (len(sampled_nodes) < sample_size):
		top, top_val = np.inf, -1.*np.inf
		for frontier in frontier_nodes_val:
			if frontier in new_changes:
				val = len([1 for n in g.vs[frontier].neighbors() if not n['sampled'] and not n['frontier']])
				frontier_nodes_val[frontier] = val
			else:
				val = frontier_nodes_val[frontier]
			if val > top_val:
				top_val = val
				top = frontier
		# print top, top_val
		sampled_nodes.add(top)
		g.vs[top]['sampled'] = 1; g.vs[top]['frontier'] = 0
		frontier_nodes_val.pop(top)


		new_changes = set()
		new_tmp = []
		for ngb in g.vs[top].neighbors():
			if not ngb['frontier'] and not ngb['sampled']:
				frontier_nodes_val[ngb['id']] = 0
				new_changes.add(ngb['id'])
				new_tmp.append(ngb['id'])
			for nn in ngb.neighbors():
				if nn['frontier']:
					new_changes.add(nn['id'])
		for new in new_tmp:
			g.vs[new]['frontier'] = 1

		if coverage and g.vcount() == len(sampled_nodes)+len(frontier_nodes_val):
			break

	return sampled_nodes


def random_expansion_walk(g, seed_node, sample_size, coverage=False):
	sampled_nodes, new_changes, frontier_nodes_val = set(), set(), dict()
	g.vs['sampled'], g.vs['frontier'] = 0, 0

	current = g.vs[seed_node]
	frontier_nodes_val[current['id']]=0; g.vs[current['id']]['frontier'] = 1
	while (len(sampled_nodes) < sample_size):
		top = np.random.choice(list(frontier_nodes_val.keys()))
		sampled_nodes.add(top)
		g.vs[top]['sampled'] = 1; g.vs[top]['frontier'] = 0
		frontier_nodes_val.pop(top)


		new_changes = set()
		new_tmp = []
		for ngb in g.vs[top].neighbors():
			if not ngb['frontier'] and not ngb['sampled']:
				frontier_nodes_val[ngb['id']] = 0
				new_tmp.append(ngb['id'])
		for new in new_tmp:
			g.vs[new]['frontier'] = 1

		if coverage and g.vcount() == len(sampled_nodes)+len(frontier_nodes_val):
			break

	return sampled_nodes

