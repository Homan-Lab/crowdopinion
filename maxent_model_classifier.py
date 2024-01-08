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
# import keras
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from ldl_utils import get_data_dict, vectorize,read_json
from helper_functions_LSTM_TF import get_feature_vectors,compile_tweet_dict,check_label_frequency,build_text_labels,keras_feature_prep,plot_NN_history,gpu_fix_cuda
from LSTM_utils import LSTM_training,LSTM_and_embedding_layer_train
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
from helper_functions import create_folder,validate_file_location,convert_to_majority,delete_model_drive
import os


MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 # words limit in each doc
ITERATIONS = 100
EPOCHS = 25
BATCHSIZE = 32

q1_LOWER = 2
q1_UPPER = 12
q2_LOWER = 2
q2_UPPER = 12
q3_LOWER = 5
q3_UPPER = 20
DS_iter = 1

def max_ent_pre(folder_name, input_train_file_name,input_dev_file_name, output_file_name,majority,train_vects,dev_vects):
    train_answer_counters = defaultdict(list)
    JSONfile = read_json(input_train_file_name)
    create_folder(folder_name) #creates the folder for saving LSTM model
    train_message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, choices) = get_data_dict(JSONfile["dictionary"])
    train_answer_counters = get_feature_vectors(fdict, JSONfile["data"])

    dev_answer_counters = defaultdict(list)
    JSONfile = read_json(input_dev_file_name)
    dev_message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, rdict) = get_data_dict(JSONfile["dictionary"])
    dev_answer_counters = get_feature_vectors(fdict, JSONfile["data"])

    if majority=="True":
        print ("Majority Label")
        train_answer_counters = convert_to_majority(train_answer_counters)
        dev_answer_counters = convert_to_majority(dev_answer_counters)
    #glove_lexicon = open(glove, 'r', encoding='utf-8')

    train_text = []
    train_labels = []
    train_text,train_labels = build_text_labels(train_message_dict,train_answer_counters)
    print(len(train_text), len(train_labels))
    
    dev_text = []
    dev_labels = []
    dev_text,dev_labels = build_text_labels(dev_message_dict,dev_answer_counters)

    # train_features, train_word_index = keras_feature_prep(train_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    # dev_features, dev_word_index = keras_feature_prep(dev_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    # word_index = train_word_index
    # print(train_features.shape, train_labels.shape)

    x_train = np.load(train_vects,allow_pickle=True)
    x_dev = np.load(dev_vects,allow_pickle=True)
    print(x_train, x_train.shape)
    print(x_dev, x_dev.shape)
    #print(x_test, x_test.shape)

    y_train = train_labels
    y_dev = dev_labels
    print(y_train, y_train.shape)
    print(y_dev, y_dev.shape)

    #print(y_test, y_test.shape)
    check_label_frequency(y_train)
    check_label_frequency(y_dev)
    pred_dim = train_labels.shape[1]

    max_ent_training(x_train, y_train, x_dev, y_dev, pred_dim, output_file_name)

def max_ent_training(x_train, y_train, x_dev, y_dev, pred_dim,model_output):
    gpu_fix_cuda()

    tb = [keras.callbacks.TensorBoard(log_dir='./dllogs')]
    main_input = Input(shape=(len(x_train[0]),1), dtype='float32') #changed from jobs dataset for image
    x_train = np.array(x_train)
    x_train = np.expand_dims(x_train,axis=-1)
    y_train = np.array(y_train)
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

    x_dev = np.array(x_dev)
    x_dev = np.expand_dims(x_dev,axis=-1)
    #x_dev = np.reshape(x_dev, (x_dev.shape[0], 1, x_dev.shape[1]))
    y_dev = np.array(y_dev)
    
    model = Sequential()
    model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(pred_dim))
    # model.add(Dropout(0.5))
    model.add(Activation("softmax"))
    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['accuracy'])

    print(model.summary())
    history_NN = model.fit(x_train, y_train, batch_size=BATCHSIZE, epochs=EPOCHS, callbacks=tb, validation_data=(x_dev, y_dev))
    # plot_model(model, to_file="figures/" + NN_name + "_CNN_layers.pdf", show_shapes=False)
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    predicted_probabilities = model.predict(x_train, batch_size=BATCHSIZE)
    delete_model_drive(model_output)
    model.save(model_output)
    del model #delete model for saving space
    # plot_NN_history(history_NN, model_output, "CNN")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_file", help="Input training file JSON name")
    parser.add_argument("--input_train_file_vects", help="Input training file JSON name")
    parser.add_argument("--input_dev_file", help="Input dev file JSON name")
    parser.add_argument("--input_dev_file_vects", help="Input dev file JSON name")
    parser.add_argument("--output_model_file", help="Output model file name")
    parser.add_argument("--folder_name", help="Main folder name")
    parser.add_argument("--majority", help="Flag for majority",default=False)
    #parser.add_argument("--output_pred_name", help="Output JSON predictions")
    args = parser.parse_args()
    input_vects = args.input_train_file_vects

    dev_vects = args.input_dev_file_vects

    max_ent_pre(args.folder_name, args.input_train_file,args.input_dev_file, args.output_model_file,args.majority,input_vects,dev_vects)

if __name__ == '__main__':
    main()
