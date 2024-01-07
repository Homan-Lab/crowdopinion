#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import argparse
import sys
import pdb
from FMM_utils import bnpy_predict
from ldl_utils import get_data_dict, get_feature_vectors, vectorize,read_json,compile_tweet_dict,save_label_dict,load_label_dict,save_label_vects,load_label_vects
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
from helper_functions import create_folder
import pickle
import argparse
# from mongo_utils import retrive_model_from_sampling_db
from cf_csv_preprocess import save_to_json

#Constants for LDA Data
ITERATIONS = 100
q1_LOWER = 2
q1_UPPER = 12
q2_LOWER = 2
q2_UPPER = 12
q3_LOWER = 5
q3_UPPER = 20
DS_iter = 1
default_LOWER = 2
default_UPPER = 12
TARGET = 'label'

#Pre_Training
def FMM_preprocess(clusters,input_file_name, output_file_name,nlp_flag):
    vects = defaultdict(list)
    tweet_dict = defaultdict(list)
    JSONfile = read_json(input_file_name)
    #create_folder(output_file_name) #creates the folder for saving LDA models
    tweet_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, rdict) = get_data_dict(JSONfile["dictionary"])
    vects = get_feature_vectors(fdict, JSONfile["data"])
    print("Running FMM in Train mode on {} Tweets on {}.".format(len(vects),id))
    if nlp_flag:
        nlp_flag = 'text'
    else:
        nlp_flag = 'labels'
    bnpy_predict(vects, rdict, clusters,tweet_dict,output_file_name,"fmm",nlp_flag)

def FMM_predict_sampled(folder_name,output_folder,filename):
    train_file = read_json(folder_name+"/"+filename+"_train.json")
    save_to_json(train_file,output_folder+"/"+filename+"_train.json")

    dev_file = read_json(folder_name+"/"+filename+"_dev.json")
    save_to_json(dev_file,output_folder+"/"+filename+"_dev.json")

    test_file = read_json(folder_name+"/"+filename+"_test.json")
    save_to_json(test_file,output_folder+"/"+filename+"_test.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Input folder of trained")
    parser.add_argument("--result_db_name", help="Database with results", default = False)
    parser.add_argument("--result_exp_name", help="Collection of the results", default = False)
    # parser.add_argument("--topics", help="Number of Topics")
    # parser.add_argument("--nlp_flag", help="NLP Data Flag",default=False)
    #parser.add_argument("--upper", help="Upper Limit")
    parser.add_argument("--output_folder", help="Output folder name")
    parser.add_argument("--output_process_id", help="Output Process ID")
    #parser.add_argument("--folder_name", help="Main folder name")
    args = parser.parse_args()
    folder_path = args.input_folder
    results_db = args.result_db_name
    exp_name = args.result_exp_name
    output_folder = args.output_folder
    # selected_cluster = str(retrive_model_from_sampling_db(results_db,exp_name))
    results_log = read_json(results_db)
    selected_cluster = str(results_log["model_selected"])
    folder_path = folder_path.replace("X",selected_cluster)
    FMM_predict_sampled(folder_path,output_folder,args.output_process_id)
    # FMM_preprocess(int(args.topics),args.input_file, args.output_file,args.nlp_flag)

if __name__ == '__main__':
    main()
