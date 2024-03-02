# Geometric Deep Learning: Applying Exphormers to NeuroGraph dataset

## Setup
1. Install `pyenv` to manage Python version using [this guide](https://realpython.com/intro-to-pyenv/).
2. Install Python 3.11 using `pyenv`.
3. Create a new virtual environment:
```
pyenv virtualenv 3.11.5 gdl-venv
```
4. Connect the virtual environment to the local directory and activate:
```
pyenv local gdl-venv 
```
5. Install the necessary dependencies:
```
pip3 install -r requirements.txt
```

## Experiments
1. To test with the gender dataset for Neurograph, run:
```
python3 main.py --cfg configs/Exphormer/neural-Gender.yaml wandb.use False
```

## Results
1. Choose experimental results by clicking through the seeded logs in the `results` directory.
2. From Terminal, run:
```
tensorboard --logdir results/[YOUR_EXPERIMENT_NAME]
```
3. Naviage to http://localhost:6006/ (or whatever link is output by Terminal) and view the loss curve plots.


____________________________________





# Exphormers: Sparse Transformers for Graphs
## (Original README for Exphormers paper)


[![arXiv](https://img.shields.io/badge/arXiv-2303.06147-b31b1b.svg)](https://arxiv.org/abs/2303.06147)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exphormer-sparse-transformers-for-graphs/graph-classification-on-cifar10-100k)](https://paperswithcode.com/sota/graph-classification-on-cifar10-100k?p=exphormer-sparse-transformers-for-graphs)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exphormer-sparse-transformers-for-graphs/node-classification-on-coco-sp)](https://paperswithcode.com/sota/node-classification-on-coco-sp?p=exphormer-sparse-transformers-for-graphs)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exphormer-sparse-transformers-for-graphs/graph-classification-on-malnet-tiny)](https://paperswithcode.com/sota/graph-classification-on-malnet-tiny?p=exphormer-sparse-transformers-for-graphs)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exphormer-sparse-transformers-for-graphs/graph-classification-on-mnist)](https://paperswithcode.com/sota/graph-classification-on-mnist?p=exphormer-sparse-transformers-for-graphs)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exphormer-sparse-transformers-for-graphs/node-classification-on-pascalvoc-sp-1)](https://paperswithcode.com/sota/node-classification-on-pascalvoc-sp-1?p=exphormer-sparse-transformers-for-graphs)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exphormer-sparse-transformers-for-graphs/link-prediction-on-pcqm-contact)](https://paperswithcode.com/sota/link-prediction-on-pcqm-contact?p=exphormer-sparse-transformers-for-graphs)



![Exphormer-viz](./Exphormers.png)


In this work we introduce new sparse transformers for graph data, and use them in the [GraphGPS](https://github.com/rampasek/GraphGPS) framework. Our sparse transformers outperform BigBird and Performer in all cases we tried, which have been mainly designed for the natural language processing context; in many cases, we even get better results than full (dense attention) transformers. Our sparse transformer has three components: actual edges, expander graphs, and universal connectors or virtual nodes. We combine these components into a single sparse attention mechanism.


### Python environment setup with Conda

```bash
conda create -n exphormer python=3.9
conda activate exphormer

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb

conda clean --all
```


### Running Exphormer
```bash
conda activate exphormer

# Running Exphormer for LRGB Datasets
python main.py --cfg configs/Exphormer_LRGB/peptides-struct-EX.yaml  wandb.use False

# Running Exphormer for Cifar10
python main.py --cfg configs/Exphormer/cifar10.yaml  wandb.use False
```
You can also set your wandb settings and use wandb.

### Guide on configs files

Most of the configs are shared with [GraphGPS](https://github.com/rampasek/GraphGPS) code. You can change the following parameters in the config files for different parameters and variants of the Exphormer:
```
prep:
  exp: True  # Set True for using expander graphs, set False otherwise. 
    # Alternatively you can set use_exp_edges to False.
    # In this case expander graphs will be calculated but not used in the Exphormer. 
  exp_deg: 5 # Set the degree of the expander graph.
    # Please note that if you set this to d, the algorithm will use d permutations 
    # or d Hamiltonian cycles, so the actual degree of the expander graph will be 2d
  exp_algorithm: 'Random-d' # Options are ['Random-d', 'Random-d2', 'Hamiltonian].
    # Default value is 'Random-d'
  add_edge_index: True # Set True if you want to add real edges beside expander edges
  num_virt_node: 1 # Set 0 for not using virtual nodes 
    # otherwise set the number of virtual nodes you want to use.
```

## Citation

Our work can be cited using the following bibtex:
```bibtex
@inproceedings{shirzad2023exphormer,
  title={Exphormer: Sparse transformers for graphs},
  author={Shirzad, Hamed and Velingker, Ameya and Venkatachalam, Balaji and Sutherland, Danica J and Sinop, Ali Kemal},
  booktitle={International Conference on Machine Learning},
  eprinttype={arXiv}, eprint={2303.06147},
  year={2023}
}
```
