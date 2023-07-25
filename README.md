# CC-Top: Constrained Clustering for Dynamic Topic Discovery

## Abstract
Research on multi-class text classification of short texts mainly focuses on supervised (transfer) learning approaches, requiring a finite set of pre-defined classes which is constant over time. This work explores deep constrained clustering (CC) as an alternative to supervised learning approaches in a setting with a dynamically changing number of classes, a task we introduce as dynamic topic discovery (DTD).We do so by using pairwise similarity constraints instead of instance-level class labels which allow for a flexible number of classes while exhibiting a competitive performance compared to supervised approaches. First, we substantiate this through a series of experiments and show that CC algorithms exhibit a predictive performance similar to state-of-the-art supervised learning algorithms while requiring less annotation effort.Second, we demonstrate the overclustering capabilities of deep CC for detecting topics in short text data sets in the absence of the ground truth class cardinality during model training.Third, we showcase that these capabilities can be leveraged for the DTD setting as a step towards dynamic learning over time and finally, we release our codebase to nurture further research in this area.

### [Link to paper 📝](https://aclanthology.org/2022.evonlp-1.5/)


This is the anonymous codebase accompanying the submission "CC-Top: Constrained Clustering for Dynamic Topic Discovery". 

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
