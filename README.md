# CC-Top

This is the anonymous codebase accompanying the submission "CC-Top - Leveraging pairwise constraints for topic classification" for review at EMNLP 2022. 

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
* Dynamic Topic Discovery with constraints: refer to next section

## Replicate Dynamic Topic Discovery Setting

### Phase 1

* Training: `python run_text.py --config configs/dbpedia_td_p1.yaml`
* Evaluation: 
    * Change `model_uri` path in `configs/dbpedia_td_p1.yaml`to the corresponding training run
    * Change `train_type`from `training` to `testing`
    * Choose `test_set` between `{d_test_1, d_test_2, d_test_combined}`
    * Execute `python run_text.py --config configs/dbpedia_td_p1.yaml`

### Phase 2

* Training:
    * Change `model_uri` path in `configs/dbpedia_td_p2.yaml`to the corresponding training run from Phase 1
    * Choose `new_split` between `{2v2, 3v1}`
    * Execute `python run_text.py --config configs/dbpedia_td_p2.yaml`
* Evaluation: 
    * Change `model_uri` path in `configs/dbpedia_td_p2.yaml`to the corresponding training run from Phase 2
    * Choose `test_set` between `{d_test_1, d_test_2, d_test_combined}`
    * Execute `python run_text.py --config configs/dbpedia_td_p2.yaml`

## Replicate benchmarks

To replicate the results from the paper, run the respective bash scripts `table<2-8>.sh` in the `experiments` folder. 

## Extend towards new dataset

* Add `<dataset_name>.py` file to `cctop/data` folder and implement the respectice `download` function
* Adapt `get_data()` function in `cctop/data/utils`