# CC-Top

This is the code accompanying the submission "CC-Top - Leveraging pairwise constraints for topic classification" for review at EMNLP 2022. 

## Setup

* create a fresh conda environment  
* pip install all dependencies via `requirements.txt`
* install the local helper package `cctop` via `pip install -e .` in elastic mode
* test your install via `python run_text.py --config configs/test_cctop.yaml`

## Training runs

* Clustering training: `python run_text.py --config configs/dbpedia_clustering.yaml`
* Supervised training: `python run_text.py --config configs/dbpedia_supervised.yaml`
* Constrained training: `python run_text.py --config configs/dbpedia_constrained.yaml`
* Overclustering with constraints: `python run_text.py --config configs/dbpedia_constrained_overclustering.yaml`
* Dynamic Topic Discovery with constraints: 

## Replicate benchmarks

## Extend towards new dataset

* add `<dataset_name>.py` file to `cctop/data` folder and implement `download` function
* adapt `get_data()` function in `cctop/data/utils`
