#!/usr/bin/env python
import os, json, nltk, re, string
# from sklearn.externals import joblib

import joblib
import numpy as np
import pickle, gzip
import pdb
import h5py
from collections import defaultdict,OrderedDict
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from ldl_utils import get_data_dict, vectorize,read_json
from gensim.models import LdaModel #https://radimrehurek.com/gensim/utils.html#gensim.utils.SaveLoad.load
import shutil #copy lda max model to final folder
import datetime
# import pymongo
import math
import pandas as pd
import warnings
# from mongo_utils import get_current_mongodb_credentials
#LDA on Language
import gensim
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer('english')
#End on LDA Language

def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def write_results_to_json_only(results,outputdir):

    results["Timestamp"] = datetime.datetime.utcnow()

    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    try:
        os.remove(outputdir)
    except:
        pass
    with open(outputdir, 'a') as outfile:
        json_dump = json.dumps(results, indent=4,default = myconverter)
        outfile.write(json_dump)
        print ("JSON file saved to "+outputdir)

def get_ids_only(data_items):
    message_ids = []
    for data_item in data_items:
        message_ids.append(data_item)
    return message_ids

def read_json_log(json_log_file):
    data_file = open(json_log_file)
    log = json.load(data_file)
    return log['model_selected']

def find_item_distribution_clusters_sklearn(cluster_predicts):
    clusters,cluster_counts = np.unique(cluster_predicts,return_counts=True)
    max_clusters = max(cluster_counts)
    min_clusters = min(cluster_counts)
    std_dev_clusters = np.std(cluster_counts).item()
    max_items = clusters[np.where(cluster_counts==max_clusters)][0].item()
    min_items = clusters[np.where(cluster_counts==min_clusters)][0].item()
    result_row = {"max_cluster_items":max_clusters,"max_cluster":max_items,"min_cluster_items":min_clusters,"min_cluster":min_items,"std_cluster_items":std_dev_clusters}
    return result_row

def iteration_selection_sklearn(kl_values,results,all_model_path,n_clusters):
    #selects the median KL -> selects the iteration -> moves to the final path and deletes rest to save space
    median_kl_index = np.argsort(kl_values)[len(kl_values)//2]
    median_kl = kl_values[median_kl_index]
    result = results[median_kl_index]
    result["avg_kl"] = np.mean(kl_values).item()
    result["std_kl"] = np.std(kl_values).item()
    model_path = all_model_path+str(median_kl_index)

    model = joblib.load(model_path+"/CL"+str(n_clusters)+".pkl")
    
    result['max_cluster_items'] = result['max_cluster_items'].item() #fix for error numpy int64 dump to jsob
    result['min_cluster_items'] = result['min_cluster_items'].item()
    
    cluster_info = read_json(model_path+"/cluster_info_"+str(n_clusters)+".json")
    
    return model,cluster_info,result

def iteration_selection_gensim(kl_values,results,all_model_path,n_clusters):
    #selects the median KL -> selects the iteration -> moves to the final path and deletes rest to save space
    median_kl_index = np.argsort(kl_values)[len(kl_values)//2]
    median_kl = kl_values[median_kl_index]
    result = results[median_kl_index]
    result["avg_kl"] = np.mean(kl_values).item()
    result["std_kl"] = np.std(kl_values).item()
    model_path = all_model_path+"/CL"+str(n_clusters)+"/"+str(median_kl_index)+"_topic.lda"
    model = LdaModel.load(model_path, mmap='r')
    cluster_info = read_json(all_model_path+"/CL"+str(n_clusters)+"/"+str(median_kl_index)+"_cluster_info.json")
    shutil.rmtree(all_model_path+"/CL"+str(n_clusters))
    model.save(all_model_path + "/" + str(n_clusters) + "_topic.lda")
    write_model_logs_to_json(all_model_path,cluster_info,"cluster_info_"+ str(n_clusters))
    del model
    return result

def iteration_selection_bnpy(kl_values,results,all_model_path,n_clusters,process_id):
    #selects the median KL -> selects the iteration -> moves to the final path and deletes rest to save space
    median_kl_index = np.argsort(kl_values)[len(kl_values)//2]
    median_kl = kl_values[median_kl_index]
    result = results[median_kl_index]
    result["avg_kl"] = np.mean(kl_values).item()
    result["std_kl"] = np.std(kl_values).item()
    model_path = all_model_path+str(median_kl_index)

    train_pred = read_json(model_path+"/"+process_id+"_train.json")
    dev_pred = read_json(model_path+"/"+process_id+"_dev.json")
    test_pred = read_json(model_path+"/"+process_id+"_test.json")

    return result,train_pred,dev_pred,test_pred

def sklearn_find_kl(train_vectors,train_preds, cluster_distributions):
    KLsum = []

    for train_vector,train_pred in zip(train_vectors,train_preds):
        train_pred = np.asarray(cluster_distributions[str(train_pred)])
        train_vector = np.asarray(train_vectors[train_vector])
        KL = KLdivergence(train_vector, train_pred)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)
    return np.mean(KLsum)

def gensim_find_kl(train_vectors,train_preds, cluster_distributions):
    KLsum = []
    for train_vector,train_pred in zip(train_vectors,train_preds):
        train_pred = generate_pd(cluster_distributions[train_pred])
        train_vector = generate_pd(train_vectors[train_vector])
        KL = KLdivergence(train_vector, train_pred)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)
    return np.mean(KLsum)


def bnpy_find_kl(train_vectors,train_preds):
    KLsum = []

    for train_vector,train_pred in zip(train_vectors,train_preds):
        train_vector = generate_pd(train_vectors[train_vector])
        KL = KLdivergence(train_vector, train_pred)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)
    return np.mean(KLsum)

def write_model_logs_to_json(MODEL_LOG_DIR, results_dict, output_name):
    create_folder(MODEL_LOG_DIR)
    with open(MODEL_LOG_DIR +"/"+ output_name + ".json", "w") as fp:
        json_export = json.dumps(results_dict)
        fp.write(json_export)
        # json.dump(results_dict, fp, sort_keys=True, indent=4)

def save_lda_model(LDA_LOG_DIR, model, output_name, i):
    path = LDA_LOG_DIR + output_name + '/models/'
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path + 'CL' + str(i) + '.lda')

def load_lda_model(path):
    model = LdaModel.load(path)
    return model

def save_pickle(path,bow):
    with open(path,'wb') as fp:
        pickle.dump(bow,fp)
    fp.close()
    print ("Saved "+path)

def load_pickle(path):
    with open(path,'rb') as fp:
        bow = pickle.load(fp)
    return bow

def create_folder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)

def save_trained_model_joblib(MODEL_LOG_DIR, model, output_name, i, j):
    # http://scikit-learn.org/stable/modules/model_persistence.html
    # i in range(LOWER, UPPER)
    # j in range(ITERATIONS)
    model_dir = MODEL_LOG_DIR + '/CL' + str(i) + '/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_dir + "Iter" + str(j) +'.pkl')
    #model.close()

def save_trained_model_joblib_sklearn(MODEL_LOG_DIR, model, output_name, i):
    # http://scikit-learn.org/stable/modules/model_persistence.html
    # i in range(LOWER, UPPER)
    # j in range(ITERATIONS)
    model_dir = MODEL_LOG_DIR + '/CL' + str(i) + '/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_dir + '.pkl')
    #model.close()


def check_if_model_trained(MODEL_LOG_DIR, model, output_name, LOWER, UPPER):

    logs = []

    for i in range(LOWER, UPPER):
        model_dir = MODEL_LOG_DIR + output_name + '/CL' + str(i) + '/'
        if not os.path.exists(model_dir):
            logs.append(1)
        else:
            logs.append(0)

    if sum(logs) != 0:
        return False
    else:
        return True

def build_prob_distribution(dataset):
    results  = []
    for data in dataset:
        total = sum(data)
        data = np.array(data)
        data = data.astype(float)
        row_to_write = data/total
        results.append(row_to_write)
    return results

def get_index_of_maximum(values):

    return values.index(max(values))

def save_keras_predict(MODEL_LOG_DIR, prediction, cnn_name):

    output_name = cnn_name.split("_")[0]
    model_dir = MODEL_LOG_DIR + output_name

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    path = model_dir + '/' + cnn_name + '_predict.pkl'

    with open(path, 'wb') as fp:
        pickle.dump(prediction, fp)
    fp.close()
    del prediction  # deletes the existing prediction

def load_keras_predict(MODEL_LOG_DIR, cnn_name):

    output_name = cnn_name.split("_")[0]
    model_dir = MODEL_LOG_DIR + output_name

    path = model_dir + '/' + cnn_name + '_predict.pkl'

    with open(path, 'rb') as fp:
        prediction = pickle.load(fp)

    return prediction


def read_original_split(jsonfile):

    #SPLIT_LOG_DIR = "data/split/"

    # Read data splits from file, NOT generate each time
    #jsonfile = SPLIT_LOG_DIR + output_name + "_" + split_prep + ".json"
    #print(jsonfile)
    with open(jsonfile) as fp:
        results_dict = json.load(fp)
    train_items = results_dict['train_set']
    dev_items = results_dict['dev_set']
    test_items = results_dict['test_set']

    return train_items, dev_items, test_items

def data_prep_bnpy(choice_counts, choices):
    '''
    Structure data in Bag of words format
    :param choice_counts: dictionary object {message_id : list_of_answer_counts}
    :param choices: possible answer choices
    :return:
    '''

    vocab_list = choices

    word_ids_per_doc = [x for x in range(len(vocab_list))]
    nWords = len(word_ids_per_doc)
    word_id = []
    word_count = []
    doc_range = [0]
    i = 0

    # create a list of word ids and non zero word counts for each document
    for doc_id in choice_counts.keys():
        ans_counts = np.array(choice_counts[doc_id])

        # find words with count > 0
        nz_word_ids = np.flatnonzero(ans_counts)
        nz_word_counts = ans_counts.ravel()[nz_word_ids]
        # print(ans_counts, nz_word_ids, nz_word_counts)
        # array([2, 1, 0, 4]), array([0, 1, 3]), array([2, 1, 4])

        word_id.extend(nz_word_ids.tolist())
        word_count.extend(nz_word_counts.tolist())

        nWords_in_doc = len(nz_word_ids)
        i += nWords_in_doc
        doc_range.append(i)

    bow_info = {
        'word_id' : np.array(word_id),
        'word_count' : np.array(word_count),
        'doc_range' : np.array(doc_range),
        'vocab_size' : np.array(nWords),
        'vocabList' : np.array(choices),
        'logFunc' : False
    }

    return bow_info

def validate_file_location(path):
    return os.path.isfile(path)

def language_prep_bnpy(message_dict, subitems):

    vocab_list = []
    for msg_id, message in message_dict.items():
        # Naive tokenization
        tokens = message.split()
        # Advanced tokenization
        # tokens = get_normalized_tokens(message, set())
        for token in tokens:
            if token not in vocab_list:
                vocab_list.append(token)

    word_ids_per_doc = [x for x in range(len(vocab_list))]
    nWords = len(word_ids_per_doc)

    word_id = []
    word_count = []
    doc_range = [0]
    i = 0

    # create a list of word ids and non zero word counts for each document
    for index, msg_id in enumerate(subitems):
        message = message_dict[msg_id]
        # Naive tokenization
        tokens = message.split()
        # Advanced tokenization
        # tokens = get_normalized_tokens(message, set())

        ans_counts = [0] * nWords
        for token in tokens:
            ans_counts[vocab_list.index(token)] += 1
        ans_counts = np.array(ans_counts)

        # find words with count > 0
        nz_word_ids = np.flatnonzero(ans_counts)
        nz_word_counts = ans_counts.ravel()[nz_word_ids]

        word_id.extend(nz_word_ids.tolist())
        word_count.extend(nz_word_counts.tolist())

        nWords_in_doc = len(nz_word_ids)
        if nWords_in_doc != 0:
            i += nWords_in_doc
            doc_range.append(i)
        else:
            print(index, tokens)
            print(ans_counts, nz_word_ids, nz_word_counts, nWords_in_doc, i)

    bow_info = {
        'word_id' : np.array(word_id),
        'word_count' : np.array(word_count),
        'doc_range' : np.array(doc_range),
        'vocab_size' : np.array(nWords),
        'vocabList' : vocab_list,
        'logFunc' : False
    }

    return bow_info

def word_normalizer(w):

    p = re.compile(r'^#*[a-z]+[\'-/]*[a-z]*$', re.UNICODE)
    pLink = re.compile(r'https*:\S+\.\w+', re.IGNORECASE)
    pMention = re.compile(r'@[A-Za-z0-9_]+\b')
    pNewLine = re.compile(r'[\r\n]+')
    pRetweet = re.compile(r'\brt\b', re.IGNORECASE)
    punctuation = {0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22}

    """
    Returns normalized word or None, if it doesn't have a normalized representation.
    """
    if pLink.match(w):
        return '[http://LINK]'
    if pMention.match(w):
        return '[@SOMEONE]'
    if len(w) < 1:
        return None
    if w[0] == '#':
        w = w.strip('.,*;-:"\'`?!)(').lower()
    else:
        w = w.strip(string.punctuation).lower()
    if not(p.match(w)):
        return None
    return w

def stemmer_lemmatizer(tokens):

    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in tokens]

def get_normalized_tokens(text, stopset):

    # naive tokenization
    words = text.split()

    tokens = []
    for w in words:
        normalized_word = word_normalizer(w)
        # remove stopwords from normalized tweet
        if (normalized_word is not None) and (normalized_word not in stopset):
            tokens.append(normalized_word)

    return stemmer_lemmatizer(tokens)

def save_bnpy_model(model_dir, trained_model, info_dict):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # store the object
    model_dict = {'best_model': trained_model, 'info': info_dict}
    with gzip.open(os.path.join(model_dir, 'best_model.pklz'), 'wb') as f:
        pickle.dump(model_dict, f)

def load_bnpy_model(model_location):

    model_dict = dict()
    for (dirpath, dirnames, filenames) in os.walk(model_location):
        for di in dirnames:
            pickle_fpath = os.path.join(model_location, di) + '/best_model.pklz'
            with gzip.open(pickle_fpath, 'rb') as model:
                m = pickle.load(model)

            nClusters = int(di.strip("CL"))

            model_dict[nClusters] = (m['best_model'], m['info'])  # tuple

    return model_dict

def KLdivergence(P, Q):
    # from Q to P
    # https://datascience.stackexchange.com/a/26318/30372
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001

    P = P + epsilon
    Q = Q + epsilon
    
    return np.sum(P * np.log(P/Q))

def answer_counters2idxstr_token(answer_counters, message_id, choices):

    labels = []

    for item in zip(answer_counters[message_id], range(len(choices))):
        labels.append(item[0] * str('%02d ' % (item[1]+1)))
    return ''.join(labels).split()

def save_max_lda_model(model_file,LDAproba_Y,max_meas_idx,lda_folder,final_lda_dst,proba_mode):
    lda_src = model_file
    lda_dst = final_lda_dst
    shutil.copyfile(lda_src, lda_dst)
    with open(lda_folder + "/" + proba_mode+ "_LDAproba_Y.pkl", 'wb') as fp:
        pickle.dump(LDAproba_Y, fp)
    fp.close()

def save_max_lda_model_trained(model_file,final_lda_dst):
    lda_src = LdaModel.load(model_file)
    lda_src.save(final_lda_dst)

def save_to_json(output_file,data_to_write):
    with open(output_file, "w") as fp:
        json.dump(data_to_write, fp, sort_keys=True, indent=4)
    print ("Saved to "+output_file)

def save_to_json_foldercheck(data,outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))
        #print ("JSON file saved to "+outputdir)

def copy_json_files(original_file,output_file):
    src_data = read_json(original_file)
    save_to_json(output_file,src_data)

def load_LDA_proba_Y(path,proba_mode):
    with open(path + "/" + proba_mode + "_LDAproba_Y.pkl" ,'rb') as fp:
        LDA_proba_Y = pickle.load(fp)
    return LDA_proba_Y

def plot_NN_history(history_NN, NN_name, kind):

    # plt.style.use('ggplot')
    plt.plot(history_NN.history['acc'])
    plt.plot(history_NN.history['val_acc'])

    plt.legend(['Learning Curve', 'Validation Curve'], loc='best')

    plt.title('%s accuracy' % kind)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.xticks(range(0, 26, 5))
    plt.yticks()

    plt.savefig("figures/" + NN_name + ("_%s.pdf" % kind))
    plt.close()

def read_labeled_data(filename):
    answer_counters = defaultdict(list)
    JSONfile = read_json(filename)
    message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, label_dict) = get_data_dict(JSONfile["dictionary"])
    answer_counters = get_feature_vectors(fdict, JSONfile["data"])

    return answer_counters,message_dict,label_dict

def read_labeled_data_NBP(filename):
    answer_counters = defaultdict(list)
    JSONfile = read_json(filename)
    message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, label_dict) = get_data_dict(JSONfile["dictionary"])
    answer_counters = get_feature_vectors_NBP(fdict, JSONfile["data"])

    return answer_counters,message_dict,label_dict

def read_labeled_data_KMeans(filename):
    answer_counters = defaultdict(list)
    JSONfile = read_json(filename)
    message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, label_dict) = get_data_dict(JSONfile["dictionary"])
    answer_counters = get_feature_vectors_only(fdict, JSONfile["data"])
    return answer_counters,message_dict,label_dict

def read_labeled_data_sklearn(filename):
    answer_counters = defaultdict(list)
    JSONfile = read_json(filename)
    message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, label_dict) = get_data_dict(JSONfile["dictionary"])
    answer_counters = get_feature_vectors_only(fdict, JSONfile["data"])
    return answer_counters,message_dict,label_dict

def get_feature_vectors(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        item["message_id"] = int(item["message_id"])
        output[item["message_id"]] = vect
    return output

def get_data_labels_only(label_data):
    data = []
    for label_value in sorted(label_data.keys()):
        data.append(label_data[label_value])
    return np.asarray(data)

def get_feature_vectors_NBP(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        total_labels = float(sum(vect))
        vect[:] = [x /total_labels for x in vect]
        item["message_id"] = item["message_id"]
        output[item["message_id"]] = vect
    return output

def get_feature_vectors_only(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        total_labels = float(sum(vect))
        vect[:] = [x /total_labels for x in vect]
        item["message_id"] = item["message_id"]
        output[item["message_id"]] = vect
    return output

def id_to_label(measure):
    labels = []
    for measure_mode in measure:
        if "KL" in measure_mode:
            labels.append("KL-Divergence")
        elif "CH" in measure_mode:
            labels.append("Chebyshev distance")
        elif "EU" in measure_mode:
            labels.append("Euclidean distance")
        elif "CA" in measure_mode:
            labels.append("Canberra metric")
        elif "CS" in measure_mode:
            labels.append("Cosine similarity")
    return labels

def generate_pd_data(result):
	total = float(sum(result[0]))
	result = result/total
	return result

def compile_tweet_dict(json_list):
    result = {int(x["message_id"]): x["message"] for x in json_list}
    return result

def plot_graphs_NBP_results(dframe_results,data_item_name,output_dir):

    plt.plot(dframe_results['Epsilon'],dframe_results['KL-divergence'])
    #plt.plot(dframe_results['Epsilon'],dframe_results['Test accuracy'])

    plt.legend(['KL-Divergence'], loc='best')

    #plt.legend(['KL-Divergence','Test accuracy'], loc='best')

    plt.title('%s KL-Divergence' % data_item_name)

    plt.xlabel('Epsilon')
    plt.ylabel('KL-Divergence')

    plt.xticks()

    plt.yticks()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + ("/%s.pdf" % data_item_name))
    print (output_dir + ("/%s.pdf" % data_item_name))

    plt.close()

def plot_graphs_NBP(results,ylabel,measure,epsilon,data_item_name,output_dir):
    results_to_write = {'Epsilon':epsilon}
    results_dframe = pd.DataFrame(results_to_write)
    for measure_mode in measure:
        result = []
        for row in results:
             result.append(row[measure_mode])
        results_dframe[measure_mode] = result
        results_dframe = results_dframe.sort_values(by=['Epsilon'])

        plt.plot(results_dframe['Epsilon'],results_dframe[measure_mode])

    plot_labels = id_to_label(measure)

    plt.legend(plot_labels, loc='best')

    #plt.title('%s KL-Divergence (NBP stage)' % data_item_name)
    plt.xlabel('Radius (r)')
    plt.ylabel(ylabel)

    plt.xticks()

    plt.yticks()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + ("/%s_NBP.pdf" % data_item_name))
    print (output_dir + ("/%s_NBP.pdf" % data_item_name))

    plt.close()

def write_NBP_results_to_mongodb(process_id,measures,results_to_write,run_location,db_name,epsilon):
    mongo_client = pymongo.MongoClient(get_current_mongodb_credentials())
    mongo_db = mongo_client[db_name]
    mongo_col = mongo_db[process_id]
    n = len(epsilon)
    for epsilon_value,i,result in zip(epsilon,range(n),results_to_write):
        results = defaultdict(list)
        results["Run Location"] = run_location
        results["date"] = datetime.datetime.utcnow()

        for measure in measures:
            results[measure] = result[measure]
            results[measure+"_NAvg"] = result[measure+"_NAvg"]
            results[measure+"_NMin"] = result[measure+"_NMin"]
            results[measure+"_NMax"] = result[measure+"_NMax"]
            # if math.isnan(measures[measure][i]):
            #     results[measure] = 0.0
            # else:
            #     results[measure] = measures[measure][i]
        results["Epsilon"] = epsilon_value

        x = mongo_col.insert_one(results)
        print ("Result saved to the database")
    print ("All results saved to DB")

def write_NBP_result_to_mongodb(process_id,measures,results_to_write,run_location,db_name,epsilon):
    mongo_client = pymongo.MongoClient(get_current_mongodb_credentials())
    mongo_db = mongo_client[db_name]
    mongo_col = mongo_db[process_id]
    n = len(epsilon)
    results = defaultdict(list)
    results["Run Location"] = run_location
    results["date"] = datetime.datetime.utcnow()

    for epsilon_value,i,result in zip(epsilon,range(n),results_to_write):
        results = defaultdict(list)
        results["Run Location"] = run_location
        results["date"] = datetime.datetime.utcnow()
        for measure in measures:
            results[measure] = result[measure]
            # if math.isnan(measures[measure][i]):
            #     results[measure] = 0.0
            # else:
            #     results[measure] = measures[measure][i]
        results["Epsilon"] = epsilon_value
        x = mongo_col.insert_one(results)
        print ("Result saved to the database")
    print ("All results saved to DB")

def JSdivergence(P, Q):
    # from Q to P
    # https://datascience.stackexchange.com/a/26318/30372
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    M = 0.5*(P+Q)
    KL1 = KLdivergence(P,M)
    KL2 = KLdivergence(Q,M)

    JS = 0.5*(KL1+KL2)
    return JS

def convert_to_majority(labels):
    output = defaultdict(list)
    for label in labels:
        label_set = labels[label]
        zero_ary = np.zeros(len(label_set))
        max_index = get_index_of_maximum(label_set)
        zero_ary[max_index] = label_set[max_index]
        output[label] = zero_ary
    return output

def convert_to_majority_array(labels):
    output = []
    for label_set in labels:
        label_set = np.array(label_set)
        zero_ary = np.zeros(len(label_set))
        max_index = np.argmax(label_set)
        zero_ary[max_index] = 1
        output.append(zero_ary)
    return output

def data_prep_bnpy(choice_counts, choices):
    '''
    Structure data in Bag of words format
    :param choice_counts: dictionary object {message_id : list_of_answer_counts}
    :param choices: possible answer choices
    :return:
    '''

    vocab_list = choices

    word_ids_per_doc = [x for x in range(len(vocab_list))]
    nWords = len(word_ids_per_doc)
    word_id = []
    word_count = []
    doc_range = [0]
    i = 0

    # create a list of word ids and non zero word counts for each document
    for doc_id in choice_counts.keys():
        ans_counts = np.array(choice_counts[doc_id])

        # find words with count > 0
        nz_word_ids = np.flatnonzero(ans_counts)
        nz_word_counts = ans_counts.ravel()[nz_word_ids]
        # print(ans_counts, nz_word_ids, nz_word_counts)
        # array([2, 1, 0, 4]), array([0, 1, 3]), array([2, 1, 4])

        word_id.extend(nz_word_ids.tolist())
        word_count.extend(nz_word_counts.tolist())

        nWords_in_doc = len(nz_word_ids)
        i += nWords_in_doc
        doc_range.append(i)

    bow_info = {
        'word_id' : np.array(word_id),
        'word_count' : np.array(word_count),
        'doc_range' : np.array(doc_range),
        'vocab_size' : np.array(nWords),
        'vocabList' : np.array(choices),
        'logFunc' : False
    }

    return bow_info

def save_bnpy_model(model_dir, trained_model, info_dict):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # store the object
    model_dict = {'best_model': trained_model, 'info': info_dict}
    with gzip.open(os.path.join(model_dir, 'best_model.pklz'), 'wb') as f:
        pickle.dump(model_dict, f)

def load_bnpy_model(model_location):

    model_dict = dict()
    for (dirpath, dirnames, filenames) in os.walk(model_location):
        for di in dirnames:
            pickle_fpath = os.path.join(model_location, di) + '/best_model.pklz'
            with gzip.open(pickle_fpath, 'rb') as model:
                m = pickle.load(model)

            nClusters = int(di.strip("CL"))

            model_dict[nClusters] = (m['best_model'], m['info'])  # tuple

    return model_dict

def get_index_of_minimum(values):

    return values.index(min(values))


def save_max_sklearn_model_trained(model_file,final_kmeans_dst,output_name):
    kmeans_src = joblib.load(model_file)
    final_dump_dst = final_kmeans_dst +"/"+output_name+".pkl"
    joblib.dump(kmeans_src,final_dump_dst)

def generate_pd(result):
    original_result = result
    try:
        total = np.sum(result)
        result = np.array(result, dtype='float64')
        result = result/total
    except RuntimeWarning:
        result = original_result
    return result

def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2

    if (lstLen % 2):
        return sortedLst[index]
    else:
        return sortedLst[index + 1]

def get_index_of_best_iteration(values,model_selection_measure):
    value_of_selected = median(values)
    index_of_selected = values.index(value_of_selected)
    return index_of_selected

    # old model selection from
    # if model_selection_measure == "cross":
    #     return values.index(min(values))
    # else:
    #     return values.index(max(values))

def map_probability_to_label(choices,prediction):
    result = {}
    for x,y in zip(choices.values(),prediction):
    	result[x] = y
    return result

def convert_pd_to_labels(label_distribution):

    total_labels = len(label_distribution)
    converted = [x*total_labels for x in label_distribution]
    converted = [round(x) for x in converted]
    converted = [int(x) for x in converted]
    return converted

def convert_pd_to_labels_sampling(label_distribution,n_votes):
    total_labels = n_votes
    converted = [x*total_labels for x in label_distribution]
    converted = [round(x) for x in converted]
    converted = [int(x) for x in converted]
    return converted

    topic_dict = {}

def generate_topics_dict(topics_dist):
    topic_dict = {}
    for id,topic in zip(range(len(topics_dist)),topics_dist):
        topic_id = '"0%s"' % str(1)
        text = str(round(topic[0],3))+"*"+topic_id
        for i in range(len(topic)-1):
            topic_id = '"0%s"' % str(i+2)
            text+=" + "+str(round(topic[i+1],3))+"*"+topic_id
        topic_dict.update({id:text})
    return topic_dict

def cluster_dist_to_write(dist_by_cluster):
    topics_dist = [generate_pd(x) for x in dist_by_cluster]
    topics_dict = generate_topics_dict(topics_dist)
    return topics_dict

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess_stem_clean(text):
    result = []
    text = remove_url(text)
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result

def remove_url(text):
    text = re.sub(r"http\S+", "", text) #remove URLs from the text
    return text

def match_cluster_to_label_dist(cluster_assigned_values,cluster_label_dist):
    predicitions = []
    for predicition in cluster_assigned_values:
        cluster_id = np.argmax(predicition)
        label_dist = cluster_label_dist[cluster_id]
        predicitions.append(label_dist)
    return np.array(predicitions)

def map_raw_label_to_label_choice(choices,prediction):
    result = {}
    for x,y in zip(choices,prediction):
    	result[x] = y
    return result

def relu(value):
    if value>0:
        return value
    else:
        return 0

def transform_for_lda(vectors):
    result_vectors = [relu(vector) for vector in vectors] #3 for 50 window size
    return result_vectors


def delete_model_drive(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            print("model file deleted")
        except:
            os.rmdir(path)
            print("model folder deleted")