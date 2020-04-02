# author: anonymous

######## required packages ########
igraph
numpy
sklearn
pickle
pandas
scipy



######## main_task.py ########
This function has three components:
1) train for learning different samplers using different policy functions (lr, rr, nr & svr) : will be released later
2) validating the learned samplers for choosing the best sampler = GTS : will be realeased later 
3) testing the GTS sampler on unseen graph. : released

For real-world graph families, we maintain 40%, 40%, 20% split. 


######## ./dataset ########

Code for generating synthetic graph families or loading real-world graph families. (real-world datasets will be realeased later)



######## ./sampling_algo ########

Code for learning and executing sampler. We provide pre-trained model for GTS in this implementation. Train, validation and testing code will be released on publication. 



