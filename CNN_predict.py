import argparse
import sys
import pdb
import numpy as np
# import keras
from tensorflow import keras
from keras.models import load_model
from ldl_utils import get_data_dict, vectorize,read_json
from helper_functions_LSTM_TF import gpu_fix_cuda,get_feature_vectors,compile_tweet_dict,check_label_frequency,build_text_labels,keras_feature_prep,write_predictions_to_json_cnn
# from LSTM_utils import LSTM_training,LSTM_and_embedding_layer_train
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
# from helper_functions import create_folder,validate_file_location
# from CNN_train_geng import image_feature_extraction
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

def CNN_predict(input_test_file, model_file,pred_output_name,test_vectors):
    # train_answer_counters = defaultdict(list)
    # JSONfile = read_json(input_train_file_name)
    # create_folder(folder_name) #creates the folder for saving LSTM model
    # train_message_dict = compile_tweet_dict(JSONfile["data"])
    # (fdict, choices) = get_data_dict(JSONfile["dictionary"])
    # train_answer_counters = get_feature_vectors(fdict, JSONfile["data"])
    gpu_fix_cuda()
    test_answer_counters = defaultdict(list)
    JSONfile = read_json(input_test_file)
    test_message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, choices) = get_data_dict(JSONfile["dictionary"])
    test_answer_counters = get_feature_vectors(fdict, JSONfile["data"])

    test_text = []
    test_labels = []
    test_text,test_labels = build_text_labels(test_message_dict,test_answer_counters)

    # test_features, test_word_index = keras_feature_prep(test_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)

    if test_vectors==False:
        test_features, test_word_index = keras_feature_prep(test_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
        x_test = test_features
    else:
        x_test = np.load(test_vectors,allow_pickle=True)
        x_test = np.array(x_test)
        x_test = np.expand_dims(x_test,axis=-1)
    print(x_test, x_test.shape)

    y_test = test_labels

    print(y_test, y_test.shape)
    check_label_frequency(y_test)
    model = load_model(model_file)
    
    predicted_probabilities = model.predict(x_test, batch_size=BATCHSIZE)

    write_predictions_to_json_cnn(predicted_probabilities,test_message_dict,choices,pred_output_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_test_file", help="Input dev file JSON name")
    parser.add_argument("--input_test_file_vects", help="Input dev file npy Vectors name",default=False)
    parser.add_argument("--input_model_file", help="Output model file name")
    parser.add_argument("--output_pred_name", help="Output JSON predictions")
    args = parser.parse_args()
    CNN_predict(args.input_test_file,args.input_model_file,args.output_pred_name,args.input_test_file_vects)

if __name__ == '__main__':
    main()
