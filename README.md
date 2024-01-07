# Population Level Label Distribution Learning

Clustering of Crowdsourced Labels in Subjective Domains. 

## Components 
 * Pre Processing of Dataset. CSV/JSON of Features+Labels to a JSON with Feature and Label Count
 * Unsupervised/Label refining methods
	* LDA - Gensim and Scikit Learn
	* FMM - BNPY (buggy)
	* KMeans - Scikit-learn
	* GMM - Scikit-learn
	* NBP (from [Weerasooriya et al.](https://arxiv.org/abs/2003.07406))
 * Supervised Learning Models
	 * CNN 
	 * LSTM
* Model evaluation methods as discussed in the NBP Paper.

## Research Papers related to this code
* [Learning to Predict Population-Level Label Distributions by Liu et al.](https://dl.acm.org/doi/fullHtml/10.1145/3308560.3317082)
* [Neighborhood-based Pooling for Population-level Label Distribution Learning by Weerasooriya et al.](https://arxiv.org/abs/2003.07406)
  
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites

```
git clone https://github.com/Homan-Lab/pldl
```

The pipeline depends on a MongoDB for storing results (it is meant to be deployed at multiple instances). Make sure you modify [mongo_utils.py](refact_code/mongo_utils.py) and `get_current_mongodb_credentials` function with your MongDB hostname and credentials.

This pipeline runs entirely on Python 2.7 (transition to Py3 is in the works). You need to create a Py2.7 virtual environment and then install dependencies. The packages required for this to work is included in `requirements.txt`. [Here is a brief guide](https://help.dreamhost.com/hc/en-us/articles/215489338-Installing-and-using-virtualenv-with-Python-2) on setting up a virtual python environment. A helpful article on how to find your [Python folder](https://askubuntu.com/questions/262063/how-to-find-python-installation-directory-on-ubuntu). Once that is setup, activate the virtualenv, install requirements using `pip install -r requirements.txt`.

**bnpy** (needed for FMM) and **pwlf** (piece wise linear fit for model selection) has to be installed manually. The modified version is included in this repo under folder (from `./bnpy` and `./piecewise_linear_fit_py`, install it using `pip install .`

**GloVe** embeddings are needed for the NLP experiments (due to the size of lexicon of ~2GB, it is not included in this repo). It is used by [helper_functions_nlp.py](https://github.com/Homan-Lab/pldl/blob/master/experimental_code/helper_functions_nlp.py) and [CNN_train.py](https://github.com/Homan-Lab/pldl/blob/master/experimental_code/CNN_train.py). You can download **glove.twitter.27B.zip**  lexicon through [their website](https://nlp.stanford.edu/projects/glove/) and store it in `data/lexicons/glove.twitter.27B/glove.twitter.27B.100d.txt`.

## Scripts to Run
Experimental code is included inside `experimental_code` folder. 
### Language based experiments
* `facebook_experiments.sh` - This is for running our experiments for Facebook Dataset. 
* `jobs_experiments.sh` - This is for running our experiments for Jobs Dataset. 
* `suicide_experiments.sh` - This is for running our experiments for Suicide Dataset. 
### Experiments without language
* `geng_experiments.sh` - This is for running experiments with datasets from [Geng et al.](http://palm.seu.edu.cn/xgeng/LDL/index.htm) 
## Data
This pipeline is designed to work with three datasets (Facebook, Jobs, and Suicide) which are included in this[repo](https://github.com/Homan-Lab/pldl_data). 

### Data split process for Jobs

| Split process | Abbreviation |
| --------------------------------------- |:-------------:|
| Broad split (50% *train* + 25% *dev* + 25% *test*) | shuffle |
| Deep split (Only for the job-themed dataset) | broad |


## Tasks

| Split process | Abbreviation |
| --------------------------------------- |:-------------:|
| Train the model using train (50%) + dev (25%) | train |
| Test the model using test (25%) | test |


## Models
### Neighborhood Based Pooling type methods
| Measure | Abbreviation |
| ------------- |:-------------:|
| KL-divergence | KL |
| Jensen Shannon divergence | JS |
| Chebyshev distance| CH |
| Canaberra metric| CA |

### Clustering type models (adopted from Tong)

| Model | Abbreviation |
| ------------- |:-------------:|
| Latent Dirichlet allocation | lda |
| Finite mixture model | fmm |


### Classification type models

| Model | Abbreviation |
| ---------- |:-------------:|
| CNN | cnn |
| LSTM | lstm |


## Parameters in the bash scripts

|Paramter| Supported Values |Description |
|--|--|--|
|dataset_id|jobQ123_BOTH_deep, jobQ123_BOTH|Name of the dataset|
|db||Name of the mongo database|
|measure|all, KL, CA, CH, JS|Information theoritic measures supported for NBP|
|EXP|NBP, LDA, FMM, RAW|Unsupervised learning method. Clustering methods (LDA,FMM), label distributions (RAW) and Neighborhood based pooling (NBP)|
|SPLIT|NONE, RND, ORI|Split mode. RND = random splits, ORI = read original splits, None = reads as it is.|


## FAQ, Problems and Fixes
1. FMM Clustering - BNPY is a buggy package on 2.7. It has to be installed through the folder in this repo as ```munkres==1.0.12``` is needed for BNPY to work on Py2.7. (Feb 2021, there is an updated version of BNPY which hasn't been tested with this pipeline yet)
2. BNPY also depends on scipy, in the package developers has planned to depreciate a **matrix** datatype, which is causing to crash. This has to be fixed by editing the core of numpy. This warning is ignored in the **FMM_utils.py**
```PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.```

