# author: anonymous

This package contains a pre-trained implementation of GTS. The full implementation will be released on publication. 


######## required packages ########
igraph
numpy
sklearn
pickle
pandas
scipy



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



