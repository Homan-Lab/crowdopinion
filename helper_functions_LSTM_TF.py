import os
#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import os, json, nltk, re, string
# from sklearn.externals import joblib
import joblib
import numpy as np
import pickle, gzip
import pdb
import h5py
# import cPickle
import _pickle as cPickle

from ldl_utils import vectorize
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from collections import defaultdict,OrderedDict
from gensim.models import LdaModel #https://radimrehurek.com/gensim/utils.html#gensim.utils.SaveLoad.load

import shutil #copy lda max model to final folder
from helper_functions import save_to_json_foldercheck
import datetime
import pandas as pd
import wandb
# WANDB_NAME = "maxent-experiments" 
# WANDB_NAME = "pooling-experiments"
# WANDB_NAME = "anuj_exps"
# WANDB_NAME = "tr_exps"
# WANDB_NAME = "pp_pldl_party"
# WANDB_NAME = "pp_pldl"
# WANDB_NAME = "DS-experiments"
# WANDB_NAME = "acl_exps"
WANDB_NAME = "crowdeval_exps"
from tqdm import tqdm

def gpu_fix_cuda():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def save_keras_trained_model(MODEL_LOG_DIR, model):
    # output_name = cnn_name.split("_")[0]
    # model_dir = MODEL_LOG_DIR + output_name
    #
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    pdb.set_trace()
    temp = cPickle.dump(model)
    file = open(MODEL_LOG_DIR, "wb")
    with open(MODEL_LOG_DIR, "wb") as file:
        cPickle.dump(model, file)
    del model  # deletes the existing model
    file.close()
#data/LSTM/jobQ1/jobQ1_CF_shuffle_lda_AVGprob.h5

def load_keras_model(MODEL_LOG_DIR, cnn_name):
    output_name = cnn_name.split("_")[0]
    model_dir = MODEL_LOG_DIR + output_name
    model = load_model(model_dir + '/' + cnn_name + '.h5')
    return model

def get_feature_vectors(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        # vect = vectorize(fdict, item["labels"])
        vect = list(item["labels"].values()) #list conversion through vectorize without frills
        item["message_id"] = int(item["message_id"])
        output[item["message_id"]] = vect
    return output

def read_labels_json(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        # vect = vectorize(fdict, item["labels"])
        vect = list(item["labels"].values()) #list conversion through vectorize without frills
        item["message_id"] = int(item["message_id"])
        output[item["message_id"]] = vect
    return output

def compile_tweet_dict(json_list):
    result = {int(x["message_id"]): x["message"] for x in json_list}
    return result

def check_label_frequency(y):

    print(y.sum(axis=0))
    #print(y_test.sum(axis=0))

def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

def write_results_to_json(model_file,model_type,score,acc,y_test_KL,y_test_Mis,y_test_Nmis,epsilon,outputdir):
    result = defaultdict(list)
    results = defaultdict(list)
    model_info = defaultdict(list)

    model_info["Model path"] = model_file
    model_info["Model"] = model_type
    model_info["Timestamp"] = datetime.datetime.now()
    result["Test score"] = score
    result["Test accuracy"] = acc
    result["KL divergence"] = y_test_KL
    result["Mutual information score"] = y_test_Mis
    result["Normalized mutual information score"] = y_test_Nmis
    results["Evaluations"] = result
    results["Model info"] = model_info
    results["Epsilon"] = epsilon
    # pdb.set_trace()
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'a') as outfile:
        outfile.write(json.dumps(results, indent=4,default = myconverter))
        print ("JSON file saved to "+outputdir)

def write_results_to_json_pandas(model_file,model_type,score,acc,y_test_KL,y_test_Mis,y_test_Nmis,epsilon,outputdir):
    result = defaultdict(list)
    results = defaultdict(list)
    model_info = defaultdict(list)

    results["Model path"] = model_file
    results["Model"] = model_type
    results["Timestamp"] = datetime.datetime.now()
    results["Test accuracy"] = acc
    results["KL divergence"] = y_test_KL
    results["Mutual information score"] = y_test_Mis
    results["Normalized mutual information score"] = y_test_Nmis
    # results["Evaluations"] = result
    results["Recall Macro"] = score['recall']['macro']
    results["Precision Macro"] = score['precision']['macro']
    results["F1 Macro"] = score['f1_macro']
    results["Epsilon"] = epsilon
    results_df = pd.DataFrame(results,index=[0])
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    if os.path.exists(outputdir):
        results_df.to_csv(outputdir, mode='a', header=False,index=False)
    else:
        results_df.to_csv(outputdir,index=False)


def write_results_to_wandb(model_file,model_type,score,acc,y_test_KL,y_test_Mis,y_test_Nmis,epsilon,dataset,weight,wandb_project):
    result = defaultdict(list)
    results = defaultdict(list)
    model_info = defaultdict(list)

    wandb.init(project=wandb_project,name=dataset)
    wandb.config = {
    "model": model_type,
    "model_path": model_file,
    "dataset": dataset
    }
    # results["Timestamp"] = datetime.datetime.now()
    results["Test accuracy"] = acc
    results["KL divergence"] = y_test_KL
    results["Mutual information score"] = y_test_Mis
    results["Normalized mutual information score"] = y_test_Nmis

    results["Recall Macro"] = score['recall']['macro']
    results["Precision Macro"] = score['precision']['macro']
    results["F1 Macro"] = score['f1_macro']
    results["Epsilon"] = epsilon
    results["Dataset"] = dataset
    results["Weight"] = weight

    wandb.log(results)


def write_results_to_json_only(results,outputdir):

    results["Timestamp"] = datetime.datetime.utcnow()

    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'a') as outfile:
        outfile.write(json.dumps(results, indent=4,default = myconverter))
        print ("JSON file saved to "+outputdir)


# def write_results_to_mongodb(model_file,process_id,score,acc,y_test_KL,y_test_Mis,y_test_Nmis,run_location,db_name,epsilon):
#     mongo_client = pymongo.MongoClient(MONGODB_URL)
#     mongo_db = mongo_client[db_name]
#     mongo_col = mongo_db[process_id]
#
#     for KL,Mis,Nmis in zip(y_test_KL,y_test_Mis,y_test_Nmis):
#         results = defaultdict(list)
#         results["Model path"] = model_file
#         results["Run Location"] = run_location
#         results["date"] = datetime.datetime.utcnow()
#         results["Test score"] = score
#         results["Test accuracy"] = acc
#         results["KL divergence"] = KL
#         results["Mutual information score"] = Mis
#         results["Normalized mutual information score"] = Nmis
#         results["Epsilon"] = epsilon
#
#         x = mongo_col.insert_one(results)
#
#     print "Result saved to the database"

def build_text_labels(message_dict,answer_counters):
    text = []
    labels = []
    for message_id in message_dict:
        text.append(message_dict[int(message_id)])
        labels.append(answer_counters[int(message_id)])
    labels = np.asarray(labels)
    return text,labels

def build_labels_dict(answer_counters):
    labels = []
    for message_id in answer_counters:
        labels.append(answer_counters[(message_id)])
    labels = np.asarray(labels)
    return labels

def write_predictions_to_json(predictions,data_dict,label_dict,output):
    data_to_write = {}
    predictions_to_write = []
    for message_id,pred_vect in zip(data_dict,predictions):
        message = data_dict[int(message_id)]
        labels = {x:y for x,y in zip(label_dict.values(),pred_vect[0].tolist())}
        predictions_to_write.append(OrderedDict([("message_id", message_id),("message", message),("labels", labels)]))
    data_to_write['dictionary'] = label_dict
    data_to_write['data'] = predictions_to_write
    save_to_json_foldercheck(data_to_write,output)

def write_predictions_to_json_cnn(predictions,data_dict,label_dict,output):
    data_to_write = {}
    predictions_to_write = []
    for message_id,pred_vect in zip(data_dict,predictions):
        message = data_dict[int(message_id)]
        labels = {x:y for x,y in zip(label_dict.values(),pred_vect.tolist())}
        predictions_to_write.append(OrderedDict([("message_id", message_id),("message", message),("labels", labels)]))
    data_to_write['dictionary'] = label_dict
    data_to_write['data'] = predictions_to_write
    save_to_json_foldercheck(data_to_write,output)

def keras_feature_prep(texts,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    # token represented by index
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print(len(sequences), len(word_index))

    features = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(features, features.shape)

    return features, word_index

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
    plt.savefig(NN_name + ("_%s.pdf" % kind))
    #plt.savefig("figures/" + NN_name + ("_%s.pdf" % kind))

def plot_KN_history(x,y1, measure, folder_name, kind):

    # plt.style.use('ggplot')

    plt.plot(x,y1)

    plt.legend([measure], loc='best')

    plt.title('%s KL-Divergence' % kind)
    plt.xlabel('epsilon')
    plt.ylabel('KL-Divergence')

    #plt.xticks(range(0, 26, 5))
    plt.xticks()
    plt.yticks()
    plt.savefig(folder_name + ("/%s.pdf" % kind))
    #plt.savefig("figures/" + NN_name + ("_%s.pdf" % kind))
    plt.close()

def plot_KN_history_all(x,y1,y2,y3,y4,y5, folder_name, kind):

    # plt.style.use('ggplot')

    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.plot(x,y4)
    plt.plot(x,y5)

    plt.legend(['KL-Divergence', 'Chebyshev distance','Euclidean distance','Canaberra distance','Cosine Similarity'], loc='best')

    plt.title('%s KL-Divergence' % kind)
    plt.xlabel('epsilon')
    plt.ylabel('KL-Divergence')

    #plt.xticks(range(0, 26, 5))
    plt.xticks()
    plt.yticks()
    plt.savefig(folder_name + ("/%s.pdf" % kind))
    plt.close()
    #plt.savefig("figures/" + NN_name + ("_%s.pdf" % kind))

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


def JSdivergence(P, Q):
    # from Q to P
    # https://datascience.stackexchange.com/a/26318/30372
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    KL1 = KLdivergence(P,Q)
    KL2 = KLdivergence(Q,P)

    JS = 0.5*(P+Q)
    return JS


# def save_keras_trained_model(MODEL_LOG_DIR, model, cnn_name):
#     output_name = cnn_name.split("_")[0]
#     model_dir = MODEL_LOG_DIR + output_name
#
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#
#     model.save(model_dir + '/' + cnn_name + '.h5')  # creates a HDF5 file
#     del model
