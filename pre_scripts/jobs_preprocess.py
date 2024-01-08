# Script for preprocessing the dataset on hate-speech by Leonardelli et al. 
# Download the paper at https://arxiv.org/abs/2109.13563
# Request the dataset at: https://github.com/dhfbk/annotators-agreement-dataset
import pandas as pd
import pdb
from itertools import groupby
from collections import OrderedDict,defaultdict
from collections import Counter
import json
import argparse
import os
import numpy as np
from helper_functions import sentence_embedding,convert_data_pldl_experiments,generate_data_bert,create_folder
#from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, 'utils/')
#from utils import gen_data_plot
import tensorflow as tf
import random
from helper_functions import save_to_json
from sklearn.model_selection import train_test_split
use_original_splits = True


# python jobs_preprocess.py --input_MT_file datasets/jobQ123MT/tweet_assignment_labels.json --input_F8_file datasets/jobQ123CF/f1011589.csv --colTweetID message_id_0 --colTweetText message-0 --colQuestion_mt question1 --colQuestion_f8 which_of_the_following_items_could_best_describe_the_point_of_view_of_jobemployment_topics_in_the_target_tweet --labelDict datasets/jobQ123CF/jobQ1_CF_dict.json --id jobQ1_BOTH --input_split_file datasets/original_splits/jobQ123_shuffle.json --foldername1 datasets/jobQ1_BOTH/processed/modeling_annotator --foldername2 datasets/jobQ1_BOTH/processed/modeling_annotator_NN --foldername3 datasets/jobQ1_BOTH/processed/PLDL
# python jobs_preprocess.py --input_MT_file datasets/jobQ123MT/tweet_assignment_labels.json --input_F8_file datasets/jobQ123CF/f1011589.csv --colTweetID message_id_0 --colTweetText message-0 --colQuestion_mt question2 --colQuestion_f8 which_of_the_following_items_could_best_describe_the_employment_status_of_the_subject_in_the_tweet --labelDict datasets/jobQ123CF/jobQ2_CF_dict.json --id jobQ2_BOTH --input_split_file datasets/original_splits/jobQ123_shuffle.json --foldername1 datasets/jobQ2_BOTH/processed/modeling_annotator --foldername2 datasets/jobQ2_BOTH/processed/modeling_annotator_NN --foldername3 datasets/jobQ2_BOTH/processed/PLDL
# python jobs_preprocess.py --input_MT_file datasets/jobQ123MT/tweet_assignment_labels.json --input_F8_file datasets/jobQ123CF/f1011589.csv --colTweetID message_id_0 --colTweetText message-0 --colQuestion_mt question3 --colQuestion_f8 what_are_the_main_ideas_of_this_tweet_as_it_relates_to_jobswork_choose_all_that_apply --labelDict datasets/jobQ123CF/jobQ3_CF_dict.json --id jobQ3_BOTH --input_split_file datasets/original_splits/jobQ123_shuffle.json --foldername1 datasets/jobQ3_BOTH/processed/modeling_annotator --foldername2 datasets/jobQ3_BOTH/processed/modeling_annotator_NN --foldername3 datasets/jobQ3_BOTH/processed/PLDL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_MT_file", help="Input JSON filename")
    parser.add_argument("--input_F8_file", help="Input messages CSV file")
    parser.add_argument("--colTweetID", help="Tweet ID columm name for CSV", default="message_id-1")
    parser.add_argument("--colTweetText", help="Tweet text column name for CSV", default = "message-1")
    parser.add_argument("--colQuestion_mt", help="Labels column name")
    parser.add_argument("--colQuestion_f8", help="Labels CSV column name")
    parser.add_argument("--labelDict", help="Label dictionary")
    parser.add_argument("--id", help="Identifier")
    parser.add_argument("--input_split_file", help="Input split JSON file",default=None)
    parser.add_argument("--foldername1", help="Main Folder name", default = "datasets/GoEmotions/processed/modeling_annotator")
    parser.add_argument("--foldername2", help="Main Folder name", default = "datasets/GoEmotions/processed/modeling_annotator_nn")
    parser.add_argument("--foldername3", help="Main Folder name", default = "datasets/GoEmotions/processed/pldl")
    args = parser.parse_args()

    input_file = args.input_MT_file
    col_tweet_ID = args.colTweetID
    col_tweet_text = args.colTweetText
    question_mt = args.colQuestion_mt
    question_f8 = args.colQuestion_f8
    original_splits = args.input_split_file
    id = args.id
    foldername1 = args.foldername1
    foldername2 = args.foldername2
    foldername3 = args.foldername3
    create_folder(foldername1)
    create_folder(foldername2)
    create_folder(foldername3)
    
    path = foldername1 + "/" + id + "_combined.json"
    labelDict = args.labelDict
    worker_id_f8 = "_worker_id"
    dframe_data = csv_read(args.input_F8_file,col_tweet_ID,col_tweet_text,question_f8,worker_id_f8)
    dframe_data = dframe_data[[col_tweet_ID,worker_id_f8,question_f8,col_tweet_text]]
    if question_mt=="question3":
        dframe_data = csv_process_q3(dframe_data,question_f8)
    dframe_labels = json_read_annotated(input_file,question_mt)
    data_mt_json = label_grouping_mt(dframe_labels,dframe_data,col_tweet_text,col_tweet_ID,question_mt)
    dframe_mt = pd.DataFrame(data_mt_json)
    mt_cols = dframe_mt.columns.tolist()
    mt_cols = mt_cols[-1:] + mt_cols[:-1]
    dframe_mt = dframe_mt[mt_cols]
    dframe_mt['platform'] = 'mt'
    dframe_data.columns = mt_cols
    dframe_data['platform'] = 'f8'
    dfs = [dframe_mt,dframe_data]
    dfs_combine = pd.concat(dfs)
    dfs_combine.drop_duplicates(inplace=True)

    label_dict = read_json_data_dict(labelDict)
    dfs_combine['label'] = dfs_combine['label'].astype('category')
    dfs_combine['label_vector'] = dfs_combine['label'].cat.codes
    cats = dfs_combine.label.astype('category')
    list_of_cats = dict(enumerate(cats.cat.categories))
    annotators = pd.unique(dfs_combine['worker_id'])
    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index

    annotators_array = np.full(len(annotators_parsed['Aindex']),-1)
    # annotators_parsed = [{x:annotators[x]} for x in range(len(annotators))] #Int indexes 
    dfs_combine = dfs_combine.join(annotators_parsed.set_index('id'), on='worker_id')

    unique_dataitems = pd.unique(dfs_combine['message_id'])
    unique_dataitems_parsed = pd.DataFrame(unique_dataitems)
    unique_dataitems_parsed = unique_dataitems_parsed.rename(columns={0:'id'})
    unique_dataitems_parsed['Mindex'] = unique_dataitems_parsed.index
    dfs_combine = dfs_combine.join(unique_dataitems_parsed.set_index('id'), on='message_id')
    # data_items_parsed = [{x:unique_dataitems[x]} for x in range(len(unique_dataitems))]

    path = foldername1 + "/" + id + "_complete_with_id.json"
    dfs_combine.to_json(path,orient='split')

    # TODO Add in the Splits
    if use_original_splits==True:
        train_items, dev_items, test_items = read_original_split(original_splits)

    dfs_dev = dfs_combine[dfs_combine.message_id.isin(dev_items)]
    path = foldername1 + "/" + id + "_dev.json"
    dfs_dev.to_json(path,orient='split',index=False)
    
    dfs_train = dfs_combine[dfs_combine.message_id.isin(train_items)]
    path = foldername1 + "/" + id + "_train.json"
    dfs_train.to_json(path,orient='split',index=False)
    
    dfs_test = dfs_combine[dfs_combine.message_id.isin(test_items)]
    path = foldername1 + "/" + id + "_test.json"
    dfs_test.to_json(path,orient='split',index=False)

    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index
    # annotators_parsed = [{x:annotators[x]} for x in range(len(annotators))] #Int indexes 

    path = foldername2 + "/" + id + "_complete_with_id.json"
    dfs_combine.to_json(path,orient='split')
    
    dfs_dev = dfs_combine[dfs_combine.message_id.isin(dev_items)]
    path = foldername3 + "/" + id + "_dev.json"

    convert_data_pldl_experiments(dfs_dev,label_dict,'Mindex',path)
    # generate_data_nn(dfs_dev,foldername2,"dev",label_dict,_id)
    
    dfs_train = dfs_combine[dfs_combine.message_id.isin(train_items)]
    path = foldername3 + "/" + id + "_train.json"
    convert_data_pldl_experiments(dfs_train,label_dict,'Mindex',path)
    # generate_data_nn(dfs_train,foldername2,"train",label_dict,_id)

    dfs_test = dfs_combine[dfs_combine.message_id.isin(test_items)]
    path = foldername3 + "/" + id + "_test.json"
    convert_data_pldl_experiments(dfs_test,label_dict,'Mindex',path)

    annotators_parsed['Aindex'] = annotators_parsed.index

    annotators_array = np.full(len(annotators_parsed),-1)
    X_train = pd.unique(dfs_train['message'])
    X_dev = pd.unique(dfs_dev['message'])
    X_test = pd.unique(dfs_test['message'])
    generate_data_bert(dfs_train,foldername2,"train",label_dict,id,X_train,annotators_array)
    generate_data_bert(dfs_test,foldername2,"test",label_dict,id,X_test,annotators_array)
    generate_data_bert(dfs_dev,foldername2,"dev",label_dict,id,X_dev,annotators_array)

def csv_process_q3(dataset,question_col):
    processed_df = pd.DataFrame(columns=dataset.columns)
    for index,row in dataset.iterrows():
        processed_row = row
        label = row[question_col]
        labels_split = label.split('\n')
        for label_split in labels_split:
            processed_row[question_col] = label_split
            processed_df = processed_df.append(processed_row)
    
    return processed_df

def read_json_data_dict(dataDict):
    data_dict = []
    with open(dataDict, 'r') as f:
        data_dict = json.load(f)
    return data_dict["dictionary"]

def json_read_annotated(input_file,question):
    data_dict = []
    with open(input_file, 'r') as f:
        data_dict = json.load(f)
    return data_dict[question]


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


def read_line_combo(dataset_raw,dataset_ids):
    results_set = {}
    count = 0
    ratings = []
    for annotation_row,datarow in zip(dataset_raw,dataset_ids):
        annotation_row = annotation_row['comment']
        ratings = annotation_row['ratings']
        del datarow['ratings']
        result_row = []
        for rating in ratings:
            rating_row = {}
            rating_row.update(datarow)
            rating_row.update(rating)
            results_set[count] = rating_row
            count += 1

    return results_set

def label_grouping_annotators(annotators,dframe_labels,label_dict): #dframe_data,col_tweet_text,col_tweet_ID,col_label):
    results = []
    for worker_id in annotators:
        labels = {}
        data = {}
        labels_for_annotator = dframe_labels.loc[dframe_labels['rater_id'] == worker_id]
        label_counts = labels_for_annotator['label'].value_counts()
        if len(label_counts) == len(label_dict):
            for label_choice in label_dict:
                labels[label_choice] = label_counts[label_choice]
        else:
            pdb.set_trace()
        data = {'worker_id':worker_id,'labels':labels}
        results.append(data)
    return results

def save_to_json(data,outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))
        print ("JSON file saved to "+outputdir)

def convert_labels_hotencoding(data_items,no_classes):
    hotencoded = []

    for index,row in data_items.iterrows():
        labels = np.zeros(no_classes)
        labels[row['label_vector']] = 1
        parsed_row = {}
        parsed_row['item'] = row['Mindex']
        parsed_row['annotator'] = row['Aindex']
        parsed_row['label'] = labels.astype(int)
        hotencoded.append(parsed_row)

    return pd.DataFrame(hotencoded)

def convert_labels_per_group(data_items,no_classes,grouping_category):
    encoded = []
    unique_data_items = pd.unique(data_items[grouping_category])
    for row in unique_data_items:
        encoded_row = {}
        labels = np.zeros(no_classes)
        items = data_items.loc[data_items[grouping_category] == row]
        for index,item in items.iterrows():
            labels[item['label_vector']]+=1
        encoded_row[grouping_category] = row
        encoded_row['label'] = labels.astype(int)
        encoded.append(encoded_row)
    return pd.DataFrame(encoded)    

def label_grouping_mt(dframe_labels,dframe_data,col_tweet_text,col_tweet_ID,col_label):
    results = []
    for message_id, values in dframe_labels.items():
        labels = []
        data = []
        prev_worker = ""
        
        for worker_id,label in values:
            # pdb.set_trace()
            if (prev_worker!=worker_id):
                annotation = {}
                annotation['worker_id'] = worker_id
                if (col_label=='question3'):
                    for label_item in label:
                        #this is to loop through checked -1 in the dataset
                        if (label_item['checked'] == 1):
                            annotation['label'] = label_item['option']
                            labels.append(label_item['option'])
                else:
                    annotation['label'] = label
                    labels.append(label)
                messages = dframe_data.loc[dframe_data[col_tweet_ID] == (message_id)]
                annotation['message']  = messages[col_tweet_text].iloc[0]
                annotation['message_id'] = messages[col_tweet_ID].iloc[0]
                results.append(annotation)
            prev_worker = worker_id
    return results
    
    
def unpivot(dframe,col_tweet_ID,worker_id_mt, col_tweet_text,colLabels):
    df = dframe.melt(id_vars=[col_tweet_ID,worker_id_mt, col_tweet_text], value_vars=colLabels)
    df = df[df["value"]>0]
    df = df.drop(columns = ["value"])
    df = df.rename(columns = {'variable':'label','id':'text_id'})
    cols = ["text_id",worker_id_mt,"label",col_tweet_text]
    df = df[cols]
    
    return df   
    
    
def csv_read(csvLocation,col_tweet_ID,col_tweet_text,col_label,col_worker_id):
    dframe = pd.read_csv(csvLocation, usecols = [col_tweet_ID,col_tweet_text,col_label,col_worker_id] ,dtype={
                col_tweet_ID : str,
                col_worker_id : str,
                col_label : str,
                col_tweet_text : str,
            })
    cols = [col_tweet_ID,col_worker_id,col_label,col_tweet_text]
    dframe = dframe[cols]
    return dframe



if __name__== "__main__":
    main()
