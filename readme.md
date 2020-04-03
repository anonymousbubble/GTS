# GTS

This package contains a pre-trained implementation of GTS. The full implementation will be released after publication. 


## Dependencies
- python2.7
- igraph
- numpy
- sklearn
- pickle
- pandas
- scipy


## Dataset

The graph families can be generated using synthetic network generators like Erdos Renyi, Barabasi and Stochastic Block Model. The real world graph families can be found in network collections such as [Network Repository](https://snap.stanford.edu/data/index.html), [SNAP](https://snap.stanford.edu/data/index.html) and [KONECT](http://konect.uni-koblenz.de/).

In this implementation, we share the SBM graph generator code in [dataset](https://github.com/anonymousbubble/GTS/tree/master/dataset) folder.


## Training and Validation

The code to train and validating GTS on a graph family, which corresponds to finding the best found set of hyper-paramters will be released later. 

In the current version, we release a pre-trained sampling policy in `sampling_algo/` directory. 


## Baseline

We release the code for testing the baseline algorithms in `sampling_algo/baselines.py` code. 


## Testing

To test the samplers on SBM graph family and community coverage task (smaller value is better), run:

```
  $ python main_task.py 
```

## Author

- anonymous


