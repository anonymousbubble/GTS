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



######## main_task.py ########
This code has three components:
1) train for learning different samplers using different policy functions (lr, rr, nr & svr) : to be released.
2) validating the learned samplers for choosing the best sampler = GTS : to be released.
3) testing the GTS sampler on unseen graph. : released

For real-world graph families, we maintain 60%, 20%, 20% split. 


######## ./dataset ########

Code for generating synthetic graph families or loading real-world graph families.



######## ./sampling_algo ########

Code for learning and executing sampler. We provide pre-trained model for GTS in this implementation. Train, validation and testing code will be released on publication. 



