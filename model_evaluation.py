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
import keras
import math
#from dit.divergences import jensen_shannon_divergence
from keras.models import load_model
from sklearn import metrics
from ldl_utils import get_data_dict, vectorize,read_json
# The line `from mongo_utils import write_results_to_mongodb` is importing the
# `write_results_to_mongodb` function from the `mongo_utils` module. This function is used to write
# the results of the model testing to a MongoDB database.
# from mongo_utils import write_results_to_mongodb
from helper_functions_LSTM_TF import get_feature_vectors,compile_tweet_dict,check_label_frequency,build_text_labels,keras_feature_prep,write_predictions_to_json_cnn,KLdivergence,JSdivergence,write_results_to_json,gpu_fix_cuda,write_results_to_json_pandas,write_results_to_wandb
from LSTM_utils import decoderize
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
from helper_functions import create_folder,validate_file_location,build_prob_distribution,convert_to_majority_array
import datetime
from CNN_train_geng import image_feature_extraction
from model_selection_pooling import random_selector
from sklearn.metrics import accuracy_score,classification_report,f1_score
import pandas as pd


MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 # words limit in each doc
ITERATIONS = 100
EPOCHS = 25
BATCHSIZE = 32



def convert_to_majority_index(labels):
    output = []
    for label_set in labels:
        label_set = np.array(label_set)
        max_index = np.argmax(label_set)
        output.append(max_index)
    return output

def measure_f1(y_test,y_pred):
    # pdb.set_trace()
    y_test = convert_to_majority_index(y_test)
    y_pred = convert_to_majority_index(y_pred)
    precision = {}
    recall = {}
    f1_macro = f1_score(y_test, y_pred, average='macro')

    f1_micro = f1_score(y_test, y_pred, average='micro')

    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    results = classification_report(y_test, y_pred, digits=3,output_dict=True)
    precision['macro'] = results['macro avg']['precision']
    precision['weighted'] = results['weighted avg']['precision']

    recall['macro'] = results['macro avg']['recall']
    recall['weighted'] = results['weighted avg']['recall']
    # pdb.set_trace()
    results['accuracy'] = accuracy_score(y_test, y_pred)

    return f1_macro,f1_micro,f1_weighted,precision,recall,results['accuracy']

def euclideanDistance(instance1, instance2):
    distance = 0
    length = len(instance1)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def model_testing(model_type,input_test_file, model_file,input_pred_file,json_location,run_location,process_id,db_name,epsilon,weight):

    result = defaultdict(list)
    gpu_fix_cuda()
    results = defaultdict(list)
    model_info = defaultdict(list)
    test_answer_counters = defaultdict(list)
    JSONfile = read_json(input_test_file)
    test_message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, choices) = get_data_dict(JSONfile["dictionary"])
    test_answer_counters = get_feature_vectors(fdict, JSONfile["data"])

    JSONfile = read_json(input_pred_file)
    pred_message_dict = compile_tweet_dict(JSONfile["data"])
    pred_answer_counters = get_feature_vectors(fdict, JSONfile["data"])
    pred_text,pred_labels = build_text_labels(pred_message_dict,pred_answer_counters)
    y_pred = build_prob_distribution(pred_labels)
    test_text = []
    test_labels = []
    test_text,test_labels = build_text_labels(test_message_dict,test_answer_counters)
    y_test = build_prob_distribution(test_labels)

    print(test_labels, test_labels.shape)
    check_label_frequency(test_labels)
    acc = measure_accuracy(y_test,y_pred)
    score = 0.0
    # Rest of lines are depreciated. (70 to 92) Replaced by module on 68
    # if model_file:
    #     if (model_type == "LSTM"):
    #         test_features, test_word_index = keras_feature_prep(test_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    #         x_test = test_features
    #         print(x_test, x_test.shape)
    #         model = load_model(model_file)
    #         score, acc = LSTM_evaluation(model, x_test, y_test)
    #     elif(model_type == "CNN"):
    #         if ( "geng" in input_test_file):
    #             x_test = image_feature_extraction(test_message_dict)
    #             print(len(x_test))
    #             x_test = np.array(x_test)
    #             x_test = np.expand_dims(x_test,axis=-1)
    #             model = load_model(model_file)
    #             score, acc = model.evaluate(x_test, test_labels, batch_size=BATCHSIZE)
    #         else:
    #             test_features, test_word_index = keras_feature_prep(test_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    #             x_test = test_features
    #             print(x_test, x_test.shape)
    #             model = load_model(model_file)
    #             score, acc = model.evaluate(x_test, test_labels, batch_size=BATCHSIZE)
    #     else:
    #         score,acc = 0.0,0.0
    print('Test score: ', score)
    results = {}
    results['f1_macro'],results['f1_micro'],results['f1_weighted'],results['precision'],results['recall'],acc = measure_f1(y_test,y_pred)
    print('Test accuracy: ', acc)
    y_test_KL, y_test_Mis, y_test_Nmis = KL_PMI_empirical2pred(y_test, y_pred)
    wandb_project = db_name
    write_results_to_wandb(model_file,model_type,results,acc,y_test_KL,y_test_Mis,y_test_Nmis,epsilon,process_id,weight,wandb_project)
    write_results_to_json_pandas(model_file,model_type,results,acc,y_test_KL,y_test_Mis,y_test_Nmis,epsilon,json_location)
    # write_results_to_mongodb(model_file,process_id,score,acc,y_test_KL,y_test_Mis,y_test_Nmis,run_location,db_name,epsilon)


def LSTM_evaluation(model,x_test,y_test):
    y_test_in, y_test_decorded = decoderize(y_test)
    score, acc = model.evaluate([x_test, y_test_in], y_test_decorded, batch_size=BATCHSIZE)
    return score,acc

def measure_accuracy(y_test,y_pred):
    total_items = len(y_test)
    label_choices = len(y_test[0])
    matched = 0
    y_test = convert_to_majority_array(y_test)
    y_pred = convert_to_majority_array(y_pred)
    for y_test_item,y_pred_item in zip(y_test,y_pred):
        if accuracy_score(y_test_item, y_pred_item) == 1:
            matched += 1
    accuracy = float(matched/total_items)
    return accuracy

def KL_PMI_empirical2pred(empirical_pcts, prediction_proba):

    KLsum = []
    MIsum = []
    adjusted_MIsum = []
    normalized_MIsum = []
    lowest_kl = highest_kl = 0.0
    lowest_kl_item = highest_kl_item = 0

    for pair in zip(empirical_pcts, prediction_proba):

        empirical_pct = pair[0]
        prediction_pct = np.asarray(pair[1])
        
        # KL = entropy(empirical_pct, prediction_pct)
        # from prediction_pct to empirical_pct
        KL = KLdivergence(empirical_pct, prediction_pct)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)

        # https://datascience.stackexchange.com/a/9271/30372
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
        # http://scikit-learn.org/stable/modules/clustering.html#mutual-info-score
        MIsum.append(metrics.mutual_info_score(empirical_pct, prediction_pct))
        normalized_MIsum.append(metrics.normalized_mutual_info_score(empirical_pct, prediction_pct))
        # adjusted_MIsum.append(metrics.adjusted_mutual_info_score(pair[0], np.asarray(pair[1])))

    KL = np.mean(KLsum)
    MIS = np.mean(MIsum)
    Nmis = np.mean(normalized_MIsum)

    print('KL divergence: ', KL)
    print('Mutual information score: ', MIS)
    print('Normalized mutual information score: ', Nmis)

    return KL, MIS, Nmis

def JS_PMI_empirical2pred(empirical_pcts, prediction_proba):

    JSsum = []
    MIsum = []
    adjusted_MIsum = []
    normalized_MIsum = []

    for pair in zip(empirical_pcts, prediction_proba):
        empirical_pct = pair[0]
        prediction_pct = np.asarray(pair[1])

        # KL = entropy(empirical_pct, prediction_pct)
        # from prediction_pct to empirical_pct
        JSsum.append(JSdivergence(empirical_pct, prediction_pct))

        # https://datascience.stackexchange.com/a/9271/30372
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
        # http://scikit-learn.org/stable/modules/clustering.html#mutual-info-score
        MIsum.append(metrics.mutual_info_score(empirical_pct, prediction_pct))
        normalized_MIsum.append(metrics.normalized_mutual_info_score(empirical_pct, prediction_pct))
        # adjusted_MIsum.append(metrics.adjusted_mutual_info_score(pair[0], np.asarray(pair[1])))

    JS = np.mean(JSsum)
    MIS = np.mean(MIsum)
    Nmis = np.mean(normalized_MIsum)

    print('KL divergence: ', JS)
    print('Mutual information score: ', MIS)
    print('Normalized mutual information score: ', Nmis)

    return JS, MIS, Nmis

def EU_empirical2pred(empirical_pcts, prediction_proba):

    EUsum = []

    adjusted_MIsum = []
    normalized_MIsum = []

    for pair in zip(empirical_pcts, prediction_proba):
        empirical_pct = pair[0]
        prediction_pct = np.asarray(pair[1])

        # KL = entropy(empirical_pct, prediction_pct)
        # from prediction_pct to empirical_pct
        EUsum.append(euclideanDistance(empirical_pct, prediction_pct))

        # https://datascience.stackexchange.com/a/9271/30372
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
        # http://scikit-learn.org/stable/modules/clustering.html#mutual-info-score
        # adjusted_MIsum.append(metrics.adjusted_mutual_info_score(pair[0], np.asarray(pair[1])))

    EU = np.mean(EUsum)

    print('Euclidean distance: ', EU)

    return EU

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Model type")
    parser.add_argument("--input_test_file", help="Input test file JSON name")
    parser.add_argument("--input_model_file", help="Output model file name",default=False)
    parser.add_argument("--input_pred_file", help="Input JSON predictions")
    parser.add_argument("--json_log_file", help="Input JSON location for log")
    parser.add_argument("--run_location", help="Identifier for Workstation",default = "PC")
    parser.add_argument("--process_id", help="Process identifier")
    parser.add_argument("--epsilon", help="Epsilon for NBP", default = 0.0)
    parser.add_argument("--db_name", help="Database identifier", default = "TempDB")
    parser.add_argument("--weight", help="Weight Parameter", default = 0.0)

    args = parser.parse_args()
    epsilon = float(args.epsilon)
    model_testing(args.model_type,args.input_test_file,args.input_model_file,args.input_pred_file,args.json_log_file,args.run_location,args.process_id,args.db_name,epsilon,args.weight)

if __name__ == '__main__':
    main()
