import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
 	mpl.use('Agg')

from sklearn.cluster import KMeans
from tqdm import tqdm
import os, math, sys, json, collections
from scipy.stats import entropy
import numpy as np
# from sklearn.externals import joblib
import joblib
from ldl_utils import read_json
from label_vectorization import get_ans_pct_vectors,get_assignments
from helper_functions import read_labeled_data_KMeans,generate_pd,map_probability_to_label,generate_topics_dict,save_to_json_foldercheck
from helper_functions_nlp import clean_text_for_sklean,build_bag_of_words,data_in_cluster_sklearn,prep_tokens_for_doc2vec,embed_to_vect,text_hybrid_labels,glove_embed_vects,hybrid_flag
# from mongo_utils import retrive_model_from_sampling_db
import argparse
import sys
import pdb
from collections import defaultdict,OrderedDict
from gensim.models import Doc2Vec
model_selection_measure = "entropy"
# v = {"entropy": entropee, "max": maxy, "distance": scores, "centroid": centroidy, "cross": cross}

def kmeans_predict(answer_counters,message_dict,label_dict,model_file):
    predictions_to_write = []
    data_to_write = {}
    model = joblib.load(model_file)
    pred_vectors = get_ans_pct_vectors(answer_counters)

    new_pred_vectors = model.transform(pred_vectors)
    cluster_labels = model.predict(pred_vectors)
    cluster_assignments, dist_by_cluster, assignments_per_cluster = get_assignments(pred_vectors, new_pred_vectors)

    #Match the cluster assignment with the cluster Distribution
    for cluster_assignment,message_id in zip(cluster_assignments,message_dict):
        raw_label_distribution = dist_by_cluster[cluster_assignment]
        label_distribution = generate_pd(raw_label_distribution)
        labels_mapped = map_probability_to_label(label_dict,label_distribution)
        cluster_id = cluster_assignment.item()+1
        predictions_to_write.append(OrderedDict([("message_id", message_id),("message", message_dict[int(message_id)]),("cluster",cluster_id),("labels", labels_mapped)]))

    data_to_write["data"] = predictions_to_write
    data_to_write["dictionary"] = label_dict
    topics_dist = [generate_pd(x) for x in dist_by_cluster]
    topics_dict = generate_topics_dict(topics_dist)
    data_to_write['topics_dict'] = topics_dict
    return data_to_write

def kmeans_predict_nlp(answer_counters,message_dict,label_dict,model_file,cluster_info,glove,hybrid,pred_vectors,embeddings):
    predictions_to_write = []
    data_to_write = {}
    model = joblib.load(model_file)
    messages,message_ids,cleaned_messages,test_tokens = clean_text_for_sklean(message_dict)
    # train_messages,train_message_ids,train_cleaned_messages,train_tokens = clean_text_for_sklean(train_message_dict)
    if embeddings == "glove":
        test_vecs = list(prep_tokens_for_doc2vec(cleaned_messages,tokens_only=True))
        glove_model = Doc2Vec.load(glove)
        test_vecs,_ = glove_embed_vects(test_vecs,glove_model)
        pred_vectors = test_vecs
    # else:
    #     train_vectors,sklearn_bow_model = build_bag_of_words(train_cleaned_messages)
    #     pred_vectors = sklearn_bow_model.transform(cleaned_messages)
    if hybrid:
        pred_vectors = text_hybrid_labels(pred_vectors,answer_counters,float(hybrid))
    cluster_labels = model.predict(pred_vectors)
 
    # cluster_assignments, dist_by_cluster, assignments_per_cluster = get_assignments(pred_vectors, new_pred_vectors)

    #Match the cluster assignment with the cluster Distribution
    for cluster_assignment,message_id in zip(cluster_labels,message_dict):
        # raw_label_distribution = dist_by_cluster[cluster_assignment]
        # label_distribution = generate_pd(raw_label_distribution)
        label_distribution = cluster_info[str(cluster_assignment)]
        labels_mapped = map_probability_to_label(label_dict,label_distribution)
        cluster_id = cluster_assignment.item()+1
        predictions_to_write.append(OrderedDict([("message_id", message_id),("message", message_dict[int(message_id)]),("cluster",int(cluster_id)),("labels", labels_mapped)]))

    data_to_write["data"] = predictions_to_write
    data_to_write["dictionary"] = label_dict
    data_to_write['topics_dict'] = cluster_info
    return data_to_write

def preprocess_data(input_file_name):

    answer_counters,message_dict,label_dict = read_labeled_data_KMeans(input_file_name)

    return answer_counters,message_dict,label_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="Input training JSON name for BoW",default=False)
    parser.add_argument("--file_to_predict", help="Input JSON name")
    parser.add_argument("--file_to_predict_vects", help="Input vects name",default=False)
    parser.add_argument("--model_file", help="Input model location",default = False)
    parser.add_argument("--result_db_name", help="Database with results", default = False)
    parser.add_argument("--result_exp_name", help="Collection of the results", default = False)
    parser.add_argument("--output_file", help="Output file name", default = False)
    parser.add_argument("--nlp_data", help="Experiment on NLP", default = False)
    parser.add_argument("--cluster_info", help="LD of the clusters", default=False)
    parser.add_argument("--embeddings",help="Embeddings Type",default='glove')
    parser.add_argument("--glove", help="Flag for Glove", default=False)
    parser.add_argument("--hybrid", help="Hybrid of Text + Labels", default=False)
    args = parser.parse_args()
    results_db = args.result_db_name
    model_path = args.model_file
    exp_name = args.result_exp_name
    nlp_flag = args.nlp_data
    cluster_info = args.cluster_info
    glove = args.glove
    hybrid = hybrid_flag(args.hybrid)
    output_file = args.output_file
    embeddings = args.embeddings
    file_to_predict_vects = args.file_to_predict_vects
    if file_to_predict_vects:
        file_to_predict_vects = np.load(file_to_predict_vects,allow_pickle=True)

    if results_db:
        # selected_cluster = str(retrive_model_from_sampling_db(results_db,exp_name))
        results_log = read_json(results_db)
        selected_cluster = str(results_log["model_selected"])
        model_path = model_path.replace("X",selected_cluster)
        cluster_info = cluster_info.replace("X",selected_cluster)
    #Reading Data
    answer_counters,message_dict,label_dict = preprocess_data(args.file_to_predict)
    if nlp_flag:
        cluster_info = read_json(cluster_info)
        # train_answer_counters,train_message_dict,train_label_dict = preprocess_data(args.train_file)
        predictions = kmeans_predict_nlp(answer_counters,message_dict,label_dict,model_path,cluster_info,glove,hybrid,file_to_predict_vects,embeddings)    
    else:
        predictions = kmeans_predict(answer_counters,message_dict,label_dict,model_path)
    save_to_json_foldercheck(predictions,output_file)


if __name__ == '__main__':
	main()
