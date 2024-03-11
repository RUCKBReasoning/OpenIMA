# OpenIMR
The PyTorch implementation of OpenIMR.

OpenIMR is proposed for open-world semi-supervised learning for node classification (ICDE 2024). 

# Overview
You can enter the folder `run/` and then run run_ours.sh (run_ours_large.sh) for open-world semi-supervised learning on the Coauthor CS dataset (ogbn-arxiv dataset).

Specifically, the repository is organized as follows:

* `losses/` contains the implementation of supervised contrastive loss, which can be used for implementing the proposed PLCL loss.

* `networks/` contains the implementation of a GAT backbone.

* `networks_large/` contains the implementation of a GAT backbone for large graph datasets.
 
* `util.py` is used for loading and pre-processing the dataset, and also includes the functions for computing metrics.

* `train_ours.py` is used for implementing the pipeline of OpenIMR.

* `train_ours_large.py` is used for implementing the pipeline of OpenIMR for large graph datasets.

# Requirements
Before running the code, you should install PyTorch, dgl, scipy, sklearn, numpy, and ogb.

# Running the code
To run OpenIMA on the example dataset Coauthor CS
```
$ cd run/
$ sh run_ours.sh
```

To run OpenIMA on the larger example dataset ogbn-arxiv
```
$ cd run/
$ sh run_ours_large.sh
```
All the experiments are repeated ten times under ten different data splits, and the reported accuracy is averaged over the ten runs.

# Reference
If you follow our idea in your work, please cite the following paper:
```
 @inproceedings{Wang2024OpenIMR,
     author = {Yanling Wang and Jing Zhang and Lingxi Zhang and Lixin Liu and Yuxiao Dong and Cuiping Li and Hong Chen and Hongzhi Yin},
     title = {Open-World Semi-Supervised Learning for Node Classification},
     booktitle = {ICDE},
     year = {2024}
   }
```
