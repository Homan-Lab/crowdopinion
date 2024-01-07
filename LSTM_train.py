import argparse
import sys
import pdb
import os
#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
from ldl_utils import get_data_dict, vectorize,read_json
from helper_functions_LSTM_TF import get_feature_vectors,compile_tweet_dict
from LSTM_utils import LSTM_training,LSTM_and_embedding_layer_train
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
from helper_functions import create_folder,convert_to_majority

ITERATIONS = 100
q1_LOWER = 2
q1_UPPER = 12
q2_LOWER = 2
q2_UPPER = 12
q3_LOWER = 5
q3_UPPER = 20
DS_iter = 1
target = 'label'

def LSTM_processing(folder_name, input_train_file_name,input_dev_file_name, output_file_name,pred_output_name,input_test_file_name,majority):
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

    test_answer_counters = defaultdict(list)
    JSONfile = read_json(input_test_file_name)
    test_message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, rdict) = get_data_dict(JSONfile["dictionary"])
    test_answer_counters = get_feature_vectors(fdict, JSONfile["data"])

    if majority=="True":
        print ("Majority Label")
        train_answer_counters = convert_to_majority(train_answer_counters)
        dev_answer_counters = convert_to_majority(dev_answer_counters)
        test_answer_counters = convert_to_majority(test_answer_counters)
    LSTM_training(train_answer_counters, train_message_dict,dev_message_dict,dev_answer_counters,test_message_dict,test_answer_counters,choices, output_file_name, target,pred_output_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_file", help="Input training file JSON name")
    parser.add_argument("--input_dev_file", help="Input dev file JSON name")
    parser.add_argument("--input_test_file", help="Input test file JSON name")
    parser.add_argument("--output_model_file", help="Output file name")
    parser.add_argument("--folder_name", help="Main folder name")
    parser.add_argument("--output_pred_name", help="Output JSON predictions")
    parser.add_argument("--majority", help="Flag for majority",default=False)
    args = parser.parse_args()

    LSTM_processing(args.folder_name, args.input_train_file,args.input_dev_file, args.output_model_file,args.output_pred_name,args.input_test_file,args.majority)

if __name__ == '__main__':
    main()
