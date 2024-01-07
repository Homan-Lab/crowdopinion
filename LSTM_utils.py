#Utils for LSTM Training Copied from Tong
import nltk, re, string, math, os, json, itertools
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
# from keras.utils import plot_model

# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical,plot_model


from gensim.models import LdaModel
import pdb
import tensorflow as tf
import numpy as np
import os
#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
# import pydot as pydot
from gensim.corpora import Dictionary, MmCorpus
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras
from scipy.stats import entropy
from sklearn import metrics
from tqdm import tqdm
# from sklearn.externals import joblib
import joblib

from collections import Counter
from nltk.corpus import stopwords

from Seq2Seq import Seq2Seq
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.models import load_model

from label_vectorization import *
from helper_functions import *
from helper_functions import read_original_split,load_LDA_proba_Y
from helper_functions_LSTM_TF import load_keras_model,save_keras_trained_model,build_text_labels,write_predictions_to_json,gpu_fix_cuda
import pickle
# from MultinomialCluster import *
# from TestGaussianMixture import *
# from TestLDA import *

split_prep = "shuffle"
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000 # words limit in each doc
EMBEDDING_DIM = 100
EPOCHS = 25
BATCHSIZE = 32
MODEL_LOG_DIR = "data/LSTM"

def LSTM_training(train_answer_counters, train_message_dict,dev_message_dict,dev_answer_counters,test_message_dict,test_answer_counters,choices, model_name,target,pred_output_name):

    train_text = []
    train_labels = []
    train_text,train_labels = build_text_labels(train_message_dict,train_answer_counters)
    print(len(train_text), len(train_labels))

    dev_text = []
    dev_labels = []
    dev_text,dev_labels = build_text_labels(dev_message_dict,dev_answer_counters)

    test_text = []
    test_labels = []
    test_text,test_labels = build_text_labels(test_message_dict,test_answer_counters)

    ### THIS IS THE BEGINNING OF TRAINING
    train_features, train_word_index = keras_feature_prep(train_text)
    dev_features, dev_word_index = keras_feature_prep(dev_text)
    test_features, test_word_index = keras_feature_prep(test_text)
    word_index = train_word_index
    print(train_features.shape, train_labels.shape)

    x_train = train_features
    x_dev = dev_features
    x_test = test_features
    print(x_train, x_train.shape)
    print(x_dev, x_dev.shape)
    #print(x_test, x_test.shape)

    y_train = train_labels
    y_dev = dev_labels
    print(y_train, y_train.shape)
    print(y_dev, y_dev.shape)
    #print(y_test, y_test.shape)
    check_label_frequency(y_train, y_dev, choices)
    #empirical_pcts, _ = get_testset_empirical_label_dist(test_items, answer_counters)
    y_train = np.nan_to_num(y_train) 
    y_dev = np.nan_to_num(y_dev) 
    print ("LSTM Training on "+model_name+" completed and saved to " +model_name)
    #predicted_probabilities = LSTM_and_embedding_layer_train(word_index, x_train, y_train, x_dev, y_dev, train_labels.shape[1], model_name,x_dev) 
    predicted_probabilities = LSTM_and_embedding_layer_train(word_index, x_train, y_train, x_dev, y_dev, train_labels.shape[1], model_name,x_test) 

    write_predictions_to_json(predicted_probabilities,test_message_dict,choices,pred_output_name)

def LSTM_and_embedding_layer_train(word_index, x_train, y_train, x_dev, y_dev, pred_dim, NN_name,x_test):
    gpu_fix_cuda()
    ((y_dev_in, y_dev), (y_train_in, y_train)) = (decoderize(y_dev), decoderize(y_train))
    model = Seq2Seq(num_decoder_tokens=pred_dim, word_index=word_index)
    # increase batch size for actual tests
    history = model.fit([x_train,y_train_in], y_train, batch_size=BATCHSIZE, epochs=EPOCHS, validation_data=([x_dev,y_dev_in], y_dev))
    proba = LSTM_and_embedding_layer_predict(x_test, NN_name,model)
    print (proba.shape)

    # plot_NN_history(history, NN_name, "LSTM")
    #save_keras_trained_model(NN_name,model)
    model.save(NN_name)
    del model #delete model for saving space
    return proba

def LSTM_and_embedding_layer_predict(x_test, NN_name,model):
    proba = model.predict(x_test, batch_size=BATCHSIZE)
    #proba = proba[:, 0, 1:-1]
    return proba

 #Testing
def LSTM_testing(answer_counters, choices, output_name, message_dict, model_name, target):

    train_items, dev_items, test_items = read_original_split(output_name, split_prep)
    total_items = train_items + dev_items + test_items
    # Keep the order of test items (using for loop)
    total_answer_counters = {}
    total_idxstr_token = []
    for message_id in total_items:
        message_id = int(message_id) #Cyril added to convert unicode from test_items
        total_answer_counters[message_id] = answer_counters[message_id]
        answer_idxstrdoc = answer_counters2idxstr_token(answer_counters, message_id, choices)
        total_idxstr_token.append(answer_idxstrdoc)
    total_vectors = get_ans_pct_vectors(total_answer_counters)

    MM_LOG_DIR = "data/" + model_name + "/"
    lda_file = MM_LOG_DIR + output_name + "_" + split_prep + "_" + target + ".json"
    print(lda_file)

    with open(lda_file) as jsonfile:
        cluster_log = json.load(jsonfile)

    # value.keys() = dict_keys(['cross', 'entropy', 'max', 'perplexity', 'topics'])
    max_meas_idx, max_meas, max_iter = bestMM_selection(cluster_log, "entropy")

    model_file = os.path.splitext(lda_file)[0] + '/CL' + str(max_meas_idx) + '.lda'
    lda_model = LdaModel.load(model_file, mmap='r')
    print(model_file)

    value = cluster_log[max_meas_idx]
    # value.keys() = dict_keys(['cross', 'entropy', 'max', 'perplexity', 'topics'])
    # https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.show_topics
    centroids = []
    for k, v in sorted(value['topics'].items(), key=lambda t:int(t[0])):
        # print(k, v)
        proba_list = [0] * len(choices)
        for each in v.split(' + '):
            proba = float(each.split('*')[0])
            choice_index = int(each.split('*')[1].replace('"', ''))
            proba_list[choice_index-1] = proba
        centroids.append(proba_list)
    centroids = np.vstack(centroids)
    print(centroids, centroids.shape)

    # 'Cluster' | 'MAXprob' | 'AVGprob'
    test_results = {}
    for proba_mode in ['MAXprob', 'AVGprob']:

        LDAproba_X = []
        LDAproba_Y = []

        for message_id in total_items:
            LDAproba_X.append(message_dict[int(message_id)])

        lda_dict = Dictionary.load(MM_LOG_DIR + output_name + "_" + split_prep + '_' + target + '.dict')

        # LDA on texts
        if target == 'text':
            total_bow_corpus = [lda_dict.doc2bow(t) for t in total_texts_ngram]
        # LDA on labels
        elif target == 'label':
            total_bow_corpus = [lda_dict.doc2bow(t) for t in total_idxstr_token]

        prediction_proba = []
        for idx, bow in enumerate(total_bow_corpus):
            tuples = lda_model.get_document_topics(bow, minimum_probability=1e-20)
            proba = [x[1] for x in tuples]
            prediction_proba.append(proba)

        for item, true, pred in zip(total_items, total_vectors, prediction_proba):
            # print(item)
            # print(true, type(true), answers2pct(true), true.shape)
            # print(pred, type(pred), max(pred), len(pred))

            if proba_mode == 'MAXprob':
                # Use the distribution with the maximum probability
                pred_maxidx = np.argmax(pred)
                proba_fromMM = centroids[pred_maxidx]
            elif proba_mode == 'AVGprob':
                # Use the distribution with the average probability
                newprobs = []
                for predprob, centroid in zip(pred, centroids):
                    newprobs.append(np.multiply(predprob, centroid))
                proba_fromMM = np.sum(newprobs, axis=0)

            LDAproba_Y.append(proba_fromMM)

        print(proba_mode, len(LDAproba_X), len(LDAproba_Y))

        ### THIS IS THE BEGINNING OF TRAINING

        features, word_index = keras_feature_prep(LDAproba_X)
        labels = np.asarray(LDAproba_Y)
        print(features.shape, labels.shape)

        x_train, rest_features = features[:len(train_items), :], features[len(train_items):, :]
        x_dev, x_test = rest_features[:len(dev_items), :], rest_features[len(dev_items):, :]
        print(x_train, x_train.shape)
        print(x_dev, x_dev.shape)
        print(x_test, x_test.shape)

        y_train, rest_labels = labels[:len(train_items), :], labels[len(train_items):, :]
        y_dev, y_test = rest_labels[:len(dev_items), :], rest_labels[len(dev_items):, :]
        print(y_train, y_train.shape)
        print(y_dev, y_dev.shape)
        print(y_test, y_test.shape)
        check_label_frequency(y_train, y_dev, y_test, choices)
        empirical_pcts, _ = get_testset_empirical_label_dist(test_items, answer_counters)

        NN_name = output_name + "_" + split_prep + "_" + model_name + '_' + proba_mode
        print(NN_name)
        #Train
        probability = LSTM_and_embedding_layer_predict_load(NN_name, MODEL_LOG_DIR)
        score,accuracy = LSTM_and_embedding_layer_test(x_test, y_test, NN_name, MODEL_LOG_DIR)
        print('Compared to empirical_pcts:')
        empirical_KL, empirical_Mis, empirical_Nmis = KL_PMI_empirical2pred(empirical_pcts, probability)

        print('Compared to y_test in ' + NN_name + ':')
        y_test_KL, y_test_Mis, y_test_Nmis = KL_PMI_empirical2pred(y_test, probability)

        test_results[NN_name] = {"score": score, "accuracy": accuracy, "empirical_KLdivergence": empirical_KL, "empirical_Mutual_information": empirical_Mis, "empirical_Normalized_mutual_information": empirical_Nmis, "Ytest_KLdivergence": y_test_KL, "Ytest_Mutual_information": y_test_Mis, "Ytest_Normalized_mutual_information": y_test_Nmis}
    write_model_logs_to_json(MODEL_LOG_DIR, test_results, output_name + "_" + split_prep + "_" + model_name)

#LSTM Training,Testing & Predictions
def LSTM_and_embedding_layer(word_index, x_train, y_train, x_dev, y_dev, x_test, y_test, pred_dim, NN_name, MODEL_LOG_DIR):

    ((y_dev_in, y_dev), (y_test_in, y_test), (y_train_in, y_train)) = (decoderize(y_dev), decoderize(y_test), decoderize(y_train))
    model = Seq2Seq(num_decoder_tokens=pred_dim, word_index=word_index)

    # increase batch size for actual tests
    history = model.fit([x_train,y_train_in], y_train, batch_size=BATCHSIZE, epochs=EPOCHS, validation_data=([x_dev,y_dev_in], y_dev))

    plot_NN_history(history, NN_name, "LSTM")
    save_keras_trained_model(MODEL_LOG_DIR, model, NN_name)
    # proba = model.predict(x_test, batch_size=BATCHSIZE)
    # score, acc = model.evaluate([x_test, y_test_in], y_test, batch_size=BATCHSIZE)
    # print('Test score: ', score)
    # print('Test accuracy: ', acc)
    #
    # proba = proba[:, 0, 1:-1]
    #
    # return score, acc, proba

#LSTM Training,Testing & Predictions



# def LSTM_and_embedding_layer_predict(x_test, NN_name, MODEL_LOG_DIR):
#
#     model = load_keras_model(MODEL_LOG_DIR, NN_name)
#
#     proba = model.predict(x_test, batch_size=BATCHSIZE)
#
#     save_keras_predict(MODEL_LOG_DIR, proba, NN_name)
#
#     return proba



# def LSTM_and_embedding_layer_test(x_train, y_train, x_dev, y_dev, x_test, y_test, NN_name, MODEL_LOG_DIR):
#
#     ((y_dev_in, y_dev), (y_test_in, y_test), (y_train_in, y_train)) = (decoderize(y_dev), decoderize(y_test), decoderize(y_train))
#     model = load_keras_model(MODEL_LOG_DIR, NN_name)
#
#     #proba = model.predict(x_test, batch_size=BATCHSIZE)
#     score, acc = model.evaluate([x_test, y_test_in], y_test, batch_size=BATCHSIZE)
#     print('Test score: ', score)
#     print('Test accuracy: ', acc)
#
#     #proba = proba[:, 0, 1:-1]
#
#     return score, acc

# def LSTM_and_embedding_layer_predict_load(NN_name, MODEL_LOG_DIR):
#
#     proba = load_keras_predict(MODEL_LOG_DIR, NN_name)
#
#     return proba

def LSTM_and_embedding_layer_test(x_test, y_test, NN_name, MODEL_LOG_DIR):

    (y_test_in, y_test) = decoderize(y_test)
    model = load_keras_model(MODEL_LOG_DIR, NN_name)

    #proba = model.predict(x_test, batch_size=BATCHSIZE)
    score, acc = model.evaluate([x_test, y_test_in], y_test, batch_size=BATCHSIZE)
    print('Test score: ', score)
    print('Test accuracy: ', acc)

    #proba = proba[:, 0, 1:-1]

    return score, acc

def bestMM_selection(cluster_log, measure_name):

    # Select model by the Maximum of **measure_name**
    # measure_name = "entropy" or "likelihood"
    print(measure_name)
    max_meas_idx, max_meas, max_iter = 0, -float("inf"), 0

    for k, v in cluster_log.items():
        # v = {"entropy": entropee, "max": maxy, "likelihood": likelies, "centroid": centroidy}
        target_values = v[measure_name]
        if max(target_values) >= max_meas:
            max_meas_idx = k
            max_meas = max(target_values)
            max_iter = get_index_of_maximum(target_values)
    print(max_meas_idx, max_meas, max_iter)

    return max_meas_idx, max_meas, max_iter
#From TextClassificaton
def keras_feature_prep(texts):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    # token represented by index
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print(len(sequences), len(word_index))

    features = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(features, features.shape)

    return features, word_index

def check_label_frequency(y_train, y_dev, choices):

    print(y_train.sum(axis=0))
    print(y_dev.sum(axis=0))
    #print(y_test.sum(axis=0))

def get_testset_empirical_label_dist(test_items, answer_counters):

    empirical_pcts = []
    repeated_pcts = []

    for message_id in test_items:
        message_id = int(message_id) #Cyril added to convert unicode from test_items
        label_pct = answers2pct(answer_counters[message_id])
        empirical_pcts.append(label_pct)
        repeated_labels = []
        for index, item in enumerate(answer_counters[message_id]):
            repeated_labels.append(str(index)*item)
        for label in list(''.join(repeated_labels)):
            repeated_pcts.append(label_pct)

    print(len(empirical_pcts), len(repeated_pcts))

    return np.asarray(empirical_pcts), np.asarray(repeated_pcts)

def decoderize(y):
    #y = np.concatenate((np.zeros((len(y),1)),y,np.zeros((len(y),1))),1)
    y_start = np.zeros_like(y)
    y_start[:,0] = 1
    y_end = np.zeros_like(y)
    y_end[:,-1] = 1
    y_start = np.stack((y_start,y),1)
    y_end = np.stack((y,y_end),1)
    return (y_start, y_end)

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

def KL_PMI_empirical2pred(empirical_pcts, prediction_proba):

    KLsum = []
    MIsum = []
    adjusted_MIsum = []
    normalized_MIsum = []

    for pair in zip(empirical_pcts, prediction_proba):
        empirical_pct = pair[0]
        prediction_pct = np.asarray(pair[1])
        #pdb.set_trace()
        # KL = entropy(empirical_pct, prediction_pct)
        # from prediction_pct to empirical_pct
        KLsum.append(KLdivergence(empirical_pct, prediction_pct))

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
    print()

    return KL, MIS, Nmis
