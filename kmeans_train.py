#python kmeans_train.py --train_file data/jobQ123_BOTH/processed/jobQ1_BOTH/split/jobQ1_BOTH_train.json --dev_file data/jobQ123_BOTH/processed/jobQ1_BOTH/split/jobQ1_BOTH_dev.json --lower 2 --upper 12 --iterations 5 --output_file jobQ1_BOTH_split_kmeans  --folder_name data/jobQ1_BOTH/kmeans

#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
 	mpl.use('Agg')

from sklearn.cluster import KMeans
# import gensim
from tqdm import tqdm
import os, math, sys, json, collections
from scipy.stats import entropy
import numpy as np
import joblib
from label_vectorization import get_ans_pct_vectors,get_assignments,tests,get_perplexity
from helper_functions import write_model_logs_to_json,read_labeled_data_KMeans,create_folder,get_index_of_best_iteration,save_trained_model_joblib,save_max_sklearn_model_trained,save_trained_model_joblib_sklearn,KLdivergence,median,create_folder
from helper_functions import sklearn_find_kl,iteration_selection_sklearn,find_item_distribution_clusters_sklearn,get_ids_only
from helper_functions_nlp import clean_text_for_sklean,build_bag_of_words,data_in_cluster_sklearn,save_trained_model_joblib_sklearn_nlp,prep_tokens_for_doc2vec,embed_to_vect,build_glove_embed,glove_embed_vects,text_hybrid_labels,hybrid_flag
import argparse
import sys
from collections import Counter
import pdb
import pandas as pd
# from sklearn.externals import joblib
from ldl_utils import read_json
import shutil

pretrained_emb = "data/lexicons/glove.twitter.27B/glove.twitter.27B.100d.txt"
#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processe
model_selection_measure = "cross"
iterations = 10
# v = {"entropy": entropee, "max": maxy, "distance": scores, "centroid": centroidy, "cross": cross}

def train_dev_kmeans_selection(train_answer_counters,dev_answer_counters, ITERATIONS, LOWER, UPPER, output_name, folder_name):

    # # Read data splits from file, NOT generate each time
    # with open(SPLIT_LOG_DIR + output_name + "_" + split_prep + ".json") as fp:
    #     results_dict = json.load(fp)
    # train_items = results_dict['train_set']
    # dev_items = results_dict['dev_set']
    #
    # train_answer_counters = {}
    # for k in train_items:
    #     train_answer_counters[k] = tweetid_answer_counters[k]
    train_vectors = get_ans_pct_vectors(train_answer_counters)
    train_message_ids = get_ids_only(train_answer_counters)
    dev_vectors = get_ans_pct_vectors(dev_answer_counters)
    results_log_dict = {}
    results_dict = {}

    for n_clusters in tqdm(range(LOWER, UPPER)):
        # print(n_clusters)
        # maxy = []
        # entropee = []
        # scores = []
        # cross = []
        # centroidy = []
        kl = []
        results = {}
        for i in range(iterations):
        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility
            clusterer = KMeans(n_clusters=n_clusters) #Default 300 iteration
            # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            train_predict = clusterer.fit_predict(train_vectors)
            cluster_distributions = data_in_cluster_sklearn(train_predict,n_clusters,train_message_ids,train_answer_counters)
            kl.append(sklearn_find_kl(train_answer_counters,train_predict, cluster_distributions))

            results[i] = find_item_distribution_clusters_sklearn(train_predict)
            create_folder(folder_name + "/logs/models/CL"+str(n_clusters)+"/temp"+str(i))
            write_model_logs_to_json(folder_name + "/logs/models/CL"+str(n_clusters)+"/temp"+str(i),cluster_distributions,"cluster_info_"+str(n_clusters))
            save_trained_model_joblib_sklearn_nlp(folder_name + "/logs/models/CL"+str(n_clusters)+"/temp"+str(i), clusterer, output_name, n_clusters)
        
        model,cluster_distributions,results_log_dict[n_clusters] = iteration_selection_sklearn(kl,results,folder_name + "/logs/models/CL"+str(n_clusters)+"/temp",n_clusters)
        shutil.rmtree(folder_name + "/logs/models/CL"+str(n_clusters))
        write_model_logs_to_json(folder_name + "/logs/models",cluster_distributions,"cluster_info_"+str(n_clusters))

        # clusterer = KMeans(n_clusters=n_clusters)
        # clusterer.fit(train_vectors)
        save_trained_model_joblib_sklearn_nlp(folder_name + "/logs/models", model, output_name, n_clusters)
    results_log_dict["exp_name"] = output_name
    write_model_logs_to_json(folder_name + "/logs/models/",results_log_dict,"cluster_log")
    print ("Completed KMeans Training")
    return 1


def train_dev_kmeans_nlp(train_answer_counters,dev_answer_counters, ITERATIONS, LOWER, UPPER, output_name, folder_name,label_dict,train_message_dict,dev_message_dict,glove,hybrid,train_vects,dev_vects):
    train_messages,train_message_ids,train_cleaned_messages,train_tokens = clean_text_for_sklean(train_message_dict)
    dev_messages,dev_message_ids,dev_cleaned_messages,dev_tokens = clean_text_for_sklean(dev_message_dict)
    if glove == "bert":
        train_vectors = train_vects
        dev_vectors = dev_vects
    if glove == True:
        vec_model = build_glove_embed(train_cleaned_messages)
        train_vectors,_ = glove_embed_vects(train_tokens,vec_model)
        vec_model.save(folder_name + "/logs/models/km_glove.dict")
    # else:
    #     train_vectors,sklearn_bow_model = build_bag_of_words(train_cleaned_messages)
    #     dev_vectors = sklearn_bow_model.transform(dev_cleaned_messages)
    
    if hybrid:
        train_vectors = text_hybrid_labels(train_vectors,train_answer_counters,float(hybrid))

    results_log_dict = {}
    results_dict = {}
    for n_clusters in tqdm(range(LOWER, UPPER)):
        # print(n_clusters)
        # maxy = []
        # entropee = []
        # scores = []
        # cross = []
        # centroidy = []
        kl = []
        results = {}
        for i in range(iterations):
        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility
            clusterer = KMeans(n_clusters=n_clusters) #Default 300 iteration
            # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            train_predict = clusterer.fit_predict(train_vectors)
            cluster_distributions = data_in_cluster_sklearn(train_predict,n_clusters,train_message_ids,train_answer_counters)
            kl.append(sklearn_find_kl(train_answer_counters,train_predict, cluster_distributions))
            results[i] = find_item_distribution_clusters_sklearn(train_predict)
            create_folder(folder_name + "/logs/models/CL"+str(n_clusters)+"/temp"+str(i))
            write_model_logs_to_json(folder_name + "/logs/models/CL"+str(n_clusters)+"/temp"+str(i),cluster_distributions,"cluster_info_"+str(n_clusters))
            save_trained_model_joblib_sklearn_nlp(folder_name + "/logs/models/CL"+str(n_clusters)+"/temp"+str(i), clusterer, output_name, n_clusters)
        
        model,cluster_distributions,results_log_dict[n_clusters] = iteration_selection_sklearn(kl,results,folder_name + "/logs/models/CL"+str(n_clusters)+"/temp",n_clusters)
        shutil.rmtree(folder_name + "/logs/models/CL"+str(n_clusters))
        write_model_logs_to_json(folder_name + "/logs/models",cluster_distributions,"cluster_info_"+str(n_clusters))
        save_trained_model_joblib_sklearn_nlp(folder_name + "/logs/models/", model, output_name, n_clusters)
    results_log_dict["exp_name"] = output_name
    write_model_logs_to_json(folder_name + "/logs/models/",results_log_dict,"cluster_log")
    print ("Completed KMeans NLP Training")
    return 1


def model_selection(cluster_log,output_dir,output_name,LOWER, UPPER):
    max_cluster_id,max_iteration = model_selection_kmeans(cluster_log, model_selection_measure,LOWER, UPPER)
    model_dir = output_dir + '/logs/models/CL' + str(max_cluster_id) + '/'
    model_path = model_dir + "Iter" + str(max_iteration) +'.pkl'
    save_max_sklearn_model_trained(model_path,output_dir,output_name)
    print ("Model training for KMeans completed cluster number: "+str(max_cluster_id)+" and saved to "+model_path)

def model_selection_kmeans(cluster_log, measure_name,LOWER, UPPER):

    # Select model by the Maximum of **measure_name**
    # measure_name = "entropy" or "likelihood"
    print(measure_name)
    max_meas = cluster_log[LOWER][measure_name]
    max_meas_idx = 0
    for n_clusters in range(LOWER, UPPER):
        # v = {"entropy": entropee, "max": maxy, "likelihood": likelies, "centroid": centroidy}
        target_values = cluster_log[n_clusters][measure_name]
        if measure_name == "cross":
            if target_values <= max_meas:
                max_meas_idx = n_clusters
                max_meas = target_values
                max_iteration = cluster_log[n_clusters]["max_iteration"]
        else:
            if target_values >= max_meas:
                max_meas_idx = n_clusters
                max_meas = target_values
                max_iteration = cluster_log[n_clusters]["max_iteration"]
    print(max_meas_idx, max_meas,max_iteration)
    return max_meas_idx,max_iteration
    #save_max_lda_model_trained(output_folder + "/" + str(max_meas_idx) + "_topic.lda",max_model_location)

def preprocess_data(input_train_file_name,input_dev_file_name,folder_name):

    create_folder(folder_name)
    create_folder(folder_name + "/logs")
    create_folder(folder_name + "/logs/models")

    train_answer_counters,train_message_dict,label_dict = read_labeled_data_KMeans(input_train_file_name)

    dev_answer_counters,dev_message_dict,label_dict = read_labeled_data_KMeans(input_dev_file_name)

    return train_answer_counters,dev_answer_counters,label_dict,train_message_dict,dev_message_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="Input training file JSON name")
    parser.add_argument("--train_file_vects", help="Input training vects .npy",default=False)
    parser.add_argument("--dev_file", help="Input dev file JSON name")
    parser.add_argument("--dev_file_vects", help="Input dev vects .npy",default=False)
    parser.add_argument("--lower", help="Lower Limit")
    parser.add_argument("--upper", help="Upper Limit")
    parser.add_argument("--iterations", help="Number of iterations")
    parser.add_argument("--output_file", help="Output file name", default = False)
    parser.add_argument("--folder_name", help="Main folder name",default = False)
    parser.add_argument("--nlp_data", help="NLP Data",default = False)
    parser.add_argument("--glove", help="Glove Embeddings",default=False)
    parser.add_argument("--hybrid", help="Hybrid of Text + Labels", default=False)
    args = parser.parse_args()
    nlp_flag = args.nlp_data
    glove = args.glove
    hybrid = hybrid_flag(args.hybrid)
    train_vects = args.train_file_vects
    dev_vects = args.dev_file_vects

    if train_vects:
        train_vects = np.load(train_vects,allow_pickle=True)
    if dev_vects:
        dev_vects = np.load(dev_vects,allow_pickle=True)
    #Reading Data
    train_answer_counters,dev_answer_counters,label_dict,train_message_dict,dev_message_dict = preprocess_data(args.train_file,args.dev_file,args.folder_name)
    if (nlp_flag):
        train_dev_kmeans_nlp(train_answer_counters,dev_answer_counters,int(args.iterations), int(args.lower), int(args.upper),args.output_file,args.folder_name,label_dict,train_message_dict,dev_message_dict,glove,hybrid,train_vects,dev_vects)
    else:
        train_dev_kmeans_selection(train_answer_counters,dev_answer_counters,int(args.iterations), int(args.lower), int(args.upper),args.output_file,args.folder_name)

if __name__ == '__main__':
	main()
