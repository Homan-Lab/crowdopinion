import os
#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')

import argparse
import sys
import pdb
import numpy as np
from tqdm import tqdm
import math
import random
import pandas as pd
import collections
from ldl_utils import get_data_dict,read_json
# from mongo_utils import write_results_to_mongodb_only
from helper_functions_LSTM_TF import read_labels_json,compile_tweet_dict,check_label_frequency,build_text_labels,keras_feature_prep,write_predictions_to_json_cnn,KLdivergence,write_results_to_json_only
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
from helper_functions import build_prob_distribution,generate_pd,get_data_labels_only,convert_pd_to_labels_sampling
from functools import partial
import multiprocessing
from scipy.stats import multinomial
from os.path import exists

lda_flag = 1 #gensim starts the clusters from 0 while others start from 1. 

def bootstrap_sampler(pred_data_pool,empirical_labels,n_samples,n_votes):
    # Samples from empirical and compares the sample with the predicition
    # Algorithm 5 Bootstrap Sampler AI Stats Paper
    # The empirical labels are read using the message_id in model_selection() function
    total = 0.0

    for i in range(n_samples):
        random_item = random_selector(pred_data_pool)
        pred_dist = get_data_labels_only(random_item["labels"])
        source_labels = empirical_labels[int(random_item["message_id"])]
        if sum(source_labels)==0:
            source_dist = source_labels
            # pdb.set_trace()
        else:
            source_dist = generate_pd(source_labels)
        #source_dist = round_up_label_values(source_dist) #sampler throws an error saying the total is not 1 but total is 1.0001
        random_label = sample_from_dist(source_dist,n_votes)
        random_label_distribution = generate_pd(random_label)
        KL = KL_pred2sample(pred_dist, random_label_distribution)
        total +=KL
    L_SDash = float(total/n_samples)
    return L_SDash

def bootstrap_sampler_with_preds(data_pool,n_samples,n_votes):
    # Samples from predicitions
    # Algorithm 5 Bootstrap Sampler AI Stats Paper
    # The empirical labels are read using the message_id in model_selection() function
    total = 0.0
    for i in range(n_samples):
        random_item = random_selector(pred_data_pool)
        source_dist = get_data_labels_only(random_item["labels"])
        #source_dist = round_up_label_values(source_dist) #sampler throws an error saying the total is not 1 but total is 1.0001
        random_label = sample_from_dist(source_dist,n_votes)
        random_label_distribution = generate_pd(random_label)
        KL = KL_pred2sample(source_dist, random_label_distribution)
        total +=KL
    L_SDash = float(total/n_samples)
    return L_SDash

def cluster_sampler(data_pool,n_samples,clusters_dists,n_votes,n_topics):
    # Algorithm 3 Cluster Sampler AI Stats Paper
    # Randomly picks data item then reads it cluster and repeats for the n_samples needed.
    # The empirical labels are read using the message_id in model_selection() function
    total_KL = 0.0
    total_MD = 0.0
    for i in range(n_samples):
        random_cluster = random_selector(data_pool)
        cluster = random_cluster['cluster']
        predicted_label_distribution = extract_dist_of_cluster(cluster,clusters_dists,n_topics)
        random_sample = sample_from_dist(predicted_label_distribution,n_votes)
        #items_in_cluster = select_items_in_cluster(data_pool,cluster)
        random_label_distribution = generate_pd(random_sample)#bootstrap_sampler(items_in_cluster,10)
        KL = KL_pred2sample(predicted_label_distribution, random_label_distribution)
        # Begin Multinomial Distribution
        MD = multinomial_distribution(random_sample,n_votes)
        #End Multinomial 
        total_KL +=KL
        total_MD +=MD
    L_SDash = float(total_KL/n_samples)
    L_SDash_MD = float(total_MD/n_samples)
    return L_SDash,L_SDash_MD

def data_to_write_generator_LSDash(result,L_S,L_SdashSet):
    results_to_write = []
    for L_Sdash in L_SdashSet:
        results = defaultdict(list)
        results["Experiment"] = result["Model Type"]
        results["NSamples"] = result["NSamples"]
        results["NIterations"] = result["NIterations"]
        results["Fraction"] = result["Fraction"]
        results["Count"] = result["Count"]
        results["Run Location"] = result["Run Location"]
        results["Sampler"] = result["Sampler"]
        results["Topics"] = result["Topics"]
        results["L_S"] = L_S
        results["L_SDash"] = L_Sdash
        if (result["Sampler"] in "NBP"):
            results["N_Avg"] = result["N_Avg"]
        results_to_write.append(results)
    return results_to_write

def data_to_write_generator_LSDash_MD(result,L_S,L_SdashSet,MD_emp,MD_Set_sample):
    results_to_write = []
    for L_Sdash,MD_dash in zip(L_SdashSet,MD_Set_sample):
        results = defaultdict(list)
        results["Experiment"] = result["Model Type"]
        results["NSamples"] = result["NSamples"]
        results["NIterations"] = result["NIterations"]
        results["Fraction"] = result["Fraction"]
        results["Count"] = result["Count"]
        results["Run Location"] = result["Run Location"]
        results["Sampler"] = result["Sampler"]
        results["Topics"] = result["Topics"]
        results["L_S"] = L_S
        results["L_SDash"] = L_Sdash
        results["MD_S"] = MD_emp
        results["MD_SDash"] = MD_dash
        if (result["Sampler"] in "NBP"):
            results["N_Avg"] = result["N_Avg"]
        results_to_write.append(results)
    return results_to_write

def generate_pd_of_cluster(random_samples,empirical_labels):
    #needed for Algorithm #3
    empirical_label_set = []
    sampled_label_set = []
    for random_item in random_samples:
        x_id = int(random_item["message_id"])
        empirical_label = empirical_labels[x_id]
        empirical_label_set.append(empirical_label)

    total_of_labels = np.asarray(empirical_label_set)
    total_of_labels = sum(total_of_labels)
    pd_of_labels = generate_pd(total_of_labels)
    return pd_of_labels

def extract_dist_of_cluster(random_cluster,clusters_dists,n_topics):
    # random_cluster_in_dist = random_cluster-lda_flag #the cluster assignments start from 0 (predicts) however the cluster assignments are from 1
    try:
        raw_dist_of_cluster = clusters_dists[str(random_cluster)]
    except:
        raw_dist_of_cluster = clusters_dists[str(random_cluster-1)] #the cluster assignments start from 0 (predicts) however the cluster assignments are from 1
    dist_of_cluster = []
    try:
        dist_sum = sum(raw_dist_of_cluster)
        dist_of_cluster = raw_dist_of_cluster
    except:
        for each in raw_dist_of_cluster.split(' + '):
            proba = float(each.split('*')[0])
            choice_index = int(each.split('*')[1].replace('"', ''))
            dist_of_cluster.append(round(proba,2))
    return dist_of_cluster

def model_selection_preprocess(input_data_file):

    JSONfile = read_json(input_data_file)

    data_dict = JSONfile["data"] #for cluster sampling to get the originating cluster
    (fdict, choices) = get_data_dict(JSONfile["dictionary"])
    
    label_data = read_labels_json(fdict, JSONfile["data"])
    label_values_only = build_prob_distribution(get_data_labels_only(label_data))

    try:
        cluster_info = JSONfile["topics_dict"]
    except:
        cluster_info = 0.0

    return data_dict,label_data,label_values_only,cluster_info

def cluster_counter(dataset):
    dframe = pd.DataFrame(dataset)
    dframe_cluster = dframe['cluster']
    cluster_counts = dframe_cluster.value_counts()
    cluster_counts = cluster_counts.sort_index()
    return cluster_counts

def measure_loss_sampling(random_samples,empirical_labels):
    #needed for Algorithm #2
    empirical_label_set = []
    sampled_label_set = []
    for random_item in random_samples:
        x_id = int(random_item["message_id"])
        empirical_label = empirical_labels[x_id]
        empirical_label_distribution = generate_pd(empirical_label)
        empirical_label_set.append(empirical_label_distribution)

        sampled_item_label_distribution = get_data_labels_only(random_item["labels"])
        sampled_label_set.append(sampled_item_label_distribution)

    L_Sdash = KL_empirical2pred(empirical_label_set,sampled_label_set)
    return L_Sdash

def multinomial_distribution(item_counts,n_samples):
    item_counts = np.array(item_counts)
    sum_items = item_counts.sum()
    item_counts = item_counts.astype(float) #To avoid any ints not converting to float
    pd_counts = (item_counts/sum_items) #PJ
    # md_value = multinomial.pmf(item_counts, n=sum_items, p=pd_counts)  
    md_value = multinomial.logpmf(item_counts, n=sum_items, p=pd_counts)

    return float(md_value)

# def measure_loss_sampling_for_cluster(random_samples,empirical_labels):
#     #needed for Algorithm #2
#     Loss = []
#     # Multi Processing for the loss measurement
#     sampling_process = partial(KL_empirical2cluster, empirical_labels)
#     iterables = random_samples
#     pool = multiprocessing.Pool()
#     Loss = pool.map(sampling_process, iterables) #fraction,count,
#     pool.close()
#     pool.join()
#
#     L_Sdash = np.mean(Loss)
#
#     return L_Sdash

# Sequential approach
# def measure_loss_sampling_for_cluster(random_samples,empirical_labels):
#     #needed for Algorithm #2
#     Loss = []
#     for random_item in random_samples:
#         Loss.append(KL_empirical2cluster(empirical_labels,random_item))
#
#     L_Sdash = np.mean(Loss)
#
#     return L_Sdash

def model_selection_for_pooling(empirical_labels,cluster_predict_labels,n_samples,L_S,n,sample_type,clusters_dists,n_votes,n_topics):
    # Algorithm 2 from AI Stats Paper
    count = 0
    print ("Model Selection for Pooling, using "+sample_type+" sampler")
    L_Sdash_Set = []
    L_Sdash = 0.0
    MD_Set = []
    tqdm_label = "Sampling "+str(sample_type)
    for i in tqdm(range(n),desc=tqdm_label):
        if (sample_type == "cluster"):
            L_Sdash,MD = cluster_sampler(cluster_predict_labels,n_samples,clusters_dists,n_votes,int(n_topics)) #S_dash
            MD_Set.append(MD)
        elif (sample_type =="bootstrap"):
            L_Sdash = bootstrap_sampler(cluster_predict_labels,empirical_labels,n_samples,n_votes) #S_dash
        elif (sample_type == "NBP"):
            L_Sdash = neighborhood_sampler(cluster_predict_labels,n_samples,n_votes) #S_dash
        L_Sdash_Set.append(L_Sdash)
        if (L_Sdash>L_S):
            count+=1

    count = float(count)
    n = float(n)
    fraction = float(count/n)
    print ("Count: "+str(count))
    print ("Percentage: "+str(fraction*100.0))

    return fraction,count,L_Sdash_Set,MD_Set

# def round_up_label_values(labels):
#     label = []
#     sum_of_labels = sum(labels)
#     diff = 1-sum_of_labels
#     diff_round = int(round(diff,0))
#     if (diff_round==0):
#       for item in labels:
#         label.append(round(item,2))
#     else:
#       for item in labels:
#         label.append(round(item,1))
#     return label

def round_up_label_values(labels):
    label = []
    sum_of_labels = sum(labels)

    if (abs(sum_of_labels)>0):
      for item in labels:
        label.append(round(item,2))
    else:
      for item in labels:
        label.append(round(item,1))
    return label


def neighborhood_sampler(pred_data_pool,n_samples,n_votes):
    # Algorithm 4 Neighborhood Sampler AI Stats Paper
    # The empirical labels are read using the message_id in model_selection() function
    # The empirical labels are read using the message_id in model_selection() function
    total = 0.0
    for i in range(n_samples):
        random_item = random_selector(pred_data_pool)
        source_dist = get_data_labels_only(random_item["labels"])
        #source_dist = round_up_label_values(source_dist) #sampler throws an error saying the total is not 1 but total is 1.0001
        random_label = sample_from_dist(source_dist,n_votes)
        random_label_distribution = generate_pd(random_label)
        KL = KL_pred2sample(source_dist, random_label_distribution)
        total +=KL
    L_SDash = float(total/n_samples)
    return L_SDash

def KL_empirical2pred(empirical_pcts, prediction_proba):
    KLsum = []

    for pair in zip(empirical_pcts, prediction_proba):
        empirical_pct = np.asarray(pair[0])
        prediction_pct = np.asarray(pair[1])

        KL = KLdivergence(empirical_pct, prediction_pct)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)

    KL = np.mean(KLsum)
    #print('KL divergence: ', KL)
    return KL

def KL_pred2sample(predicted_ldl, sampled_ldl):
    predicted_ldl = np.asarray(predicted_ldl)
    sampled_ldl = np.asarray(sampled_ldl)
    KL = KLdivergence(predicted_ldl,sampled_ldl)
    return KL

def KL_empirical2cluster(empirical_pcts, cluster):
    KLsum = []

    for empirical in empirical_pcts:
        empirical_pct = np.asarray(empirical_pcts[empirical])
        empirical_pct = generate_pd(empirical_pct)
        prediction_pct = np.asarray(cluster)

        KL = KLdivergence(empirical_pct, prediction_pct)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)

    KL = np.mean(KLsum)
    #print('KL divergence: ', KL)
    return KL

def sample_from_dist(dist,n_votes):
    no_choices = len(dist)
    #the converstions to the distribution is due to the way how the np.random.choice handles things
    #when the sum is not equal to 1 (absolute) it throws and error
    #in our PDs the sum is 1.0 or 1.00000001 or 0.999999 due to our computations
    #https://stackoverflow.com/questions/25985120/numpy-1-9-0-valueerror-probabilities-do-not-sum-to-1
    dist = round_up_label_values(dist)
    dist = np.array(dist)
    dist /= dist.sum()
    dist = dist.astype('float64')

    try:
        sample_assignments = np.random.choice(no_choices, n_votes, p=dist)
    except:
        dist = [1.00/no_choices for i in range(no_choices)]
        sample_assignments = np.random.choice(no_choices, n_votes, p=dist)
    samples = collections.Counter(sample_assignments)
    sample = []
    for choice in range(no_choices):
        if (samples[choice]):
            sample.append(samples[choice])
        else:
            sample.append(0)
    return sample

def select_items_in_cluster(data_pool,cluster):
    sequence = []

    for data_item in data_pool:
        if data_item['cluster'] == cluster:
            sequence.append(data_item)
    return sequence

def random_selector(data_items):
    item = random.choice(data_items)
    return item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Model type")
    parser.add_argument("--sampler", help="Sampling Method",default = "bootstrap")
    parser.add_argument("--topics", help="Number of Topics",default = 0.0)
    parser.add_argument("--votes", help="Number of Votes",default = 0)
    parser.add_argument("--input_test_file", help="Input test file JSON name")
    parser.add_argument("--n_samples_to_draw", help="Number of samples to draw")
    parser.add_argument("--n_iterations", help="Number of iterations")
    parser.add_argument("--input_pred_file", help="Input JSON predictions")
    parser.add_argument("--json_log_file", help="Input JSON location for log")
    parser.add_argument("--run_location", help="Identifier for Workstation",default = "PC")
    parser.add_argument("--process_id", help="Process identifier")
    parser.add_argument("--db_name", help="Database identifier", default = "ds_jobs_original")

    args = parser.parse_args()
    process_id= args.process_id
    sample_type = args.sampler
    process_id = process_id+"_"+sample_type
    n_samples = int(args.n_samples_to_draw)
    n_iterations = int(args.n_iterations)
    n_topics = float(args.topics)
    n_votes = int(args.votes)
    json_log_file = args.json_log_file
    if "lda" not in process_id or "fmm" not in process_id:
        global lda_flag
        lda_flag = 0

    # pred_data_dict is extracted for the cluster sampling as it contains the predicitions and their original cluster information
    empirical_data_dict,empirical_data,empirical_label_values_only,empirical_cluster_info = model_selection_preprocess(args.input_test_file)
    pred_data_dict,pred_data,pred_label_values_only,pred_cluster_info = model_selection_preprocess(args.input_pred_file)
    clusters_dist = pred_cluster_info #cluster info is only stored in pred-dataset
    L_S = KL_empirical2pred(empirical_label_values_only, pred_label_values_only)

    fraction,count,L_Sdash_Set,MD_Sampling_Set = model_selection_for_pooling(empirical_data,pred_data_dict,n_samples,L_S,n_iterations,sample_type,clusters_dist,n_votes,n_topics)

    results = defaultdict(list)
    results["Model Type"] = args.model_type
    results["NSamples"] = n_samples
    results["NIterations"] = n_iterations
    results["Fraction"] = fraction
    results["Count"] = count
    results["Run Location"] = args.run_location
    results["Sampler"] = sample_type
    results["Topics"] = n_topics
    if (sample_type in "NBP"):
        results["N_Avg"] = clusters_dist
    # write_results_to_json_only(results,args.json_log_file)
    if (sample_type in "cluster"):
        #Begin Multinomial Distribution
        cluster_counts = cluster_counter(pred_data_dict) #KJ
        sum_items = cluster_counts.sum()
        pd_cluster_counts = (cluster_counts/sum_items).values #PJ
        cluster_counts = cluster_counts.values #X

        # multinomial.pmf(cluster_counts, n=sum_items, p=pd_cluster_counts)
        
        # pdb.set_trace()
        md_empirical = multinomial_distribution(cluster_counts,sum_items)
        #Ends
        results_db = data_to_write_generator_LSDash_MD(results,L_S,L_Sdash_Set,md_empirical,MD_Sampling_Set)
    else:
        results_db = data_to_write_generator_LSDash(results,L_S,L_Sdash_Set) 
    df_results = pd.DataFrame(results_db)
    if exists(json_log_file):
        current_df = pd.read_json(json_log_file)
        combined = [df_results,current_df]
        combined_df = pd.concat(combined)
        combined_df = combined_df.reset_index(drop=True)
        df_results = combined_df
        # df_results = df_results.append(current_df,ignore_index=True,sort=False)
    df_results.to_json(json_log_file)

    # write_oresults_to_mongodb_only(results_db,process_id,args.db_name)

if __name__ == '__main__':
    main()
