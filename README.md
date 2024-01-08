# CrowdOpinion

Clustering of Crowdsourced Labels in Subjective Domains, source code for the [paper](https://virtual2023.aclweb.org/paper_P197.html).

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
* [Subjective Crowd Disagreements for Subjective Data: Uncovering Meaningful CrowdOpinion with Population-level Learning](https://virtual2023.aclweb.org/paper_P197.html)
* [Learning to Predict Population-Level Label Distributions by Liu et al.](https://dl.acm.org/doi/fullHtml/10.1145/3308560.3317082)
* [Neighborhood-based Pooling for Population-level Label Distribution Learning by Weerasooriya et al.](https://arxiv.org/abs/2003.07406)
  
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites

```
git clone https://github.com/Homan-Lab/crowdopinion
```

This pipeline runs entirely on Python 3.19 (originally designed for py2, FMM might need py2). You need to create a Py3.6(or newer) virtual environment and then install dependencies. The packages required for this to work is included in `requirements.txt`. [Here is a brief guide](https://help.dreamhost.com/hc/en-us/articles/215489338-Installing-and-using-virtualenv-with-Python-2) on setting up a virtual python environment. A helpful article on how to find your [Python folder](https://askubuntu.com/questions/262063/how-to-find-python-installation-directory-on-ubuntu). Once that is setup, activate the virtualenv, install requirements using `pip install -r requirements.txt`.

**bnpy** (needed for FMM) and **pwlf** (piece wise linear fit for model selection) has to be installed manually. The modified version is included in this repo under folder (from `./bnpy` and `./piecewise_linear_fit_py`, install it using `pip install .`

Once you have taken care of this, install `tensorflow-gpu` and `pytorch-gpu` on the environment. Since this is dependent on how you've originally setup (conda env or native python env), better to follow whichever guide that fits the env setup. Next step is to install the "easy" packages on `requirements.txt`.

Before running experiments, everything is designed to be logged and tracked through [WandB](https://wandb.ai/home). We recommend creating an account and setting up API access for it so the remaining setups will be easier. 

## Scripts to Run
Experimental code is included inside root folder. 
### Language based experiments
* `mr_experiments.sh` - This is for running our experiments for Movie Reviews dataset

This script is an interface for the underlying experiments thats kicked off through `bert_experiments.sh`. Reading both scripts will help you on how to create the folders. We have included the data files for MovieReviews for reference. 

## Data
This pipeline is designed to work with three datasets (Facebook, Jobs, and Suicide) which are included in this [repo](https://github.com/Homan-Lab/pldl_data). 

Preprocessing scripts for the datasets supported in this package (as mentioned in ACL paper) is inside the `pre_scripts` folder.

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

### Clustering type models (adopted from Liu et al.)

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

## How to cite this work?

```
@inproceedings{weerasooriya-etal-2023-subjective,
    title = "Subjective Crowd Disagreements for Subjective Data: Uncovering Meaningful {C}rowd{O}pinion with Population-level Learning",
    author = "Weerasooriya, Tharindu Cyril  and
      Luger, Sarah  and
      Poddar, Saloni  and
      KhudaBukhsh, Ashiqur  and
      Homan, Christopher",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.54",
    doi = "10.18653/v1/2023.acl-long.54",
    pages = "950--966",
    abstract = "Human-annotated data plays a critical role in the fairness of AI systems, including those that deal with life-altering decisions or moderating human-created web/social media content. Conventionally, annotator disagreements are resolved before any learning takes place. However, researchers are increasingly identifying annotator disagreement as pervasive and meaningful. They also question the performance of a system when annotators disagree. Particularly when minority views are disregarded, especially among groups that may already be underrepresented in the annotator population. In this paper, we introduce CrowdOpinion, an unsupervised learning based approach that uses language features and label distributions to pool similar items into larger samples of label distributions. We experiment with four generative and one density-based clustering method, applied to five linear combinations of label distributions and features. We use five publicly available benchmark datasets (with varying levels of annotator disagreements) from social media (Twitter, Gab, and Reddit). We also experiment in the wild using a dataset from Facebook, where annotations come from the platform itself by users reacting to posts. We evaluate CrowdOpinion as a label distribution prediction task using KL-divergence and a single-label problem using accuracy measures.",
}
```