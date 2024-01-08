import pandas as pd
import pdb
from itertools import groupby
from collections import OrderedDict,defaultdict
from collections import Counter
import json
import argparse
import os
import numpy as np

import sys
sys.path.insert(0, 'utils/')
#from utils import gen_data_plot
import tensorflow as tf
from helper_functions import sentence_embedding,convert_data_pldl_experiments,save_to_json,generate_data_bert
# from ldl_utils import read_json
# from split_data import split_items_train_dev_test
# from helper_functions import read_original_split,generate_pd,get_feature_vectors,compile_tweet_dict,get_data_dict

# python3 sbic_preprocess.py --input_MT_file_dev datasets/SBIC/original_splits/SBIC.v2.dev.csv --input_MT_file_train datasets/SBIC/original_splits/SBIC.v2.trn.csv --input_MT_file_test datasets/SBIC/original_splits/SBIC.v2.tst.csv  --colTweetID HITId --colTweetText post --colQuestion_mt intentYN --id sbic_intent --foldername1 datasets/SBIC/processed/intent/modeling_annotator --foldername2 datasets/SBIC/processed/intent/modeling_annotator_nn --foldername3 datasets/SBIC/processed/intent/pldl
# python3 sbic_preprocess.py --input_MT_file_dev datasets/SBIC/original_splits/SBIC.v2.dev.csv --input_MT_file_train datasets/SBIC/original_splits/SBIC.v2.trn.csv --input_MT_file_test datasets/SBIC/original_splits/SBIC.v2.tst.csv  --colTweetID HITId --colTweetText post --colQuestion_mt intentYN --id sbicintent --foldername1 datasets/sbicintent/subsample/modeling_annotator --foldername2 datasets/sbicintent/subsample/modeling_annotator_nn --foldername3 datasets/sbicintent/subsample/pldl

# python sbic_preprocess.py --input_MT_file_dev datasets/SBIC/original_splits/SBIC.v2.dev.csv --input_MT_file_train datasets/SBIC/original_splits/SBIC.v2.trn.csv --input_MT_file_test datasets/SBIC/original_splits/SBIC.v2.tst.csv  --colTweetID HITId --colTweetText post --colQuestion_mt sexYN --id sbic_lewd --foldername1 datasets/SBIC/processed/lewd/modeling_annotator --foldername2 datasets/SBIC/processed/lewd/modeling_annotator_nn --foldername3 datasets/SBIC/processed/lewd/pldl
# python sbic_preprocess.py --input_MT_file_dev datasets/SBIC/original_splits/SBIC.v2.dev.csv --input_MT_file_train datasets/SBIC/original_splits/SBIC.v2.trn.csv --input_MT_file_test datasets/SBIC/original_splits/SBIC.v2.tst.csv  --colTweetID HITId --colTweetText post --colQuestion_mt sexYN --id sbiclewd --foldername1 datasets/sbiclewd/subsample/modeling_annotator --foldername2 datasets/sbiclewd/subsample/modeling_annotator_nn --foldername3 datasets/sbiclewd/subsample/pldl


import random

use_original_splits = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_MT_file_dev", help="Input messages dev CSV file")
    parser.add_argument("--input_MT_file_train", help="Input messages train CSV file")
    parser.add_argument("--input_MT_file_test", help="Input messages test CSV file")
    parser.add_argument("--colTweetID", help="Tweet ID columm name for CSV", default="HITId")
    parser.add_argument("--colTweetText", help="Tweet text column name for CSV", default = "post")
    parser.add_argument("--colQuestion_mt", help="Labels column name")
    parser.add_argument("--id", help="Identifier")
    parser.add_argument("--foldername1", help="Main Folder name", default = "datasets/SBIC/processed/intentfull/modeling_annotator")
    parser.add_argument("--foldername2", help="Main Folder name", default = "datasets/SBIC/processed/intentfull/modeling_annotator_nn")
    parser.add_argument("--foldername3", help="Main PLDL experiments Folder name", default = "datasets/SBIC/processed/intentfull/pldl")
    args = parser.parse_args()

    dev_input_file = args.input_MT_file_dev
    train_input_file = args.input_MT_file_train
    test_input_file = args.input_MT_file_test    
    col_tweet_ID = args.colTweetID
    col_tweet_text = args.colTweetText
    question_mt = args.colQuestion_mt
    foldername1 = args.foldername1
    foldername2 = args.foldername2
    foldername3 = args.foldername3
    id = args.id
    path = foldername1 + "/"+ id + "_combined.json"
    worker_id_mt = "WorkerId"
    
    dev_dframe_data = csv_read(args.input_MT_file_dev,col_tweet_ID,col_tweet_text,question_mt,worker_id_mt)
    dev_dframe_data = dev_dframe_data[[col_tweet_ID,worker_id_mt,question_mt,col_tweet_text]]
    dev_dframe_data['platform'] = 'mt'
    dev_dframe_data['split'] = 'dev'
    
    train_dframe_data = csv_read(args.input_MT_file_train,col_tweet_ID,col_tweet_text,question_mt,worker_id_mt)
    train_dframe_data = train_dframe_data[[col_tweet_ID,worker_id_mt,question_mt,col_tweet_text]]
    train_dframe_data['platform'] = 'mt'
    train_dframe_data['split'] = 'train'
    
    test_dframe_data = csv_read(args.input_MT_file_test,col_tweet_ID,col_tweet_text,question_mt,worker_id_mt)
    test_dframe_data = test_dframe_data[[col_tweet_ID,worker_id_mt,question_mt,col_tweet_text]]
    test_dframe_data['platform'] = 'mt'
    test_dframe_data['split'] = 'test'
    if use_original_splits==False:
        train_items = list(pd.unique(train_dframe_data['HITId']))
        train_items = random.sample(train_items,1000)
        train_dframe_data = train_dframe_data[train_dframe_data.HITId.isin(train_items)]

        test_items = list(pd.unique(test_dframe_data['HITId']))
        test_items = random.sample(test_items,500)
        test_dframe_data = test_dframe_data[test_dframe_data.HITId.isin(test_items)]
        
        dev_items = list(pd.unique(dev_dframe_data['HITId']))
        dev_items = random.sample(dev_items,500)
        dev_dframe_data = dev_dframe_data[dev_dframe_data.HITId.isin(dev_items)]


    dfs = [dev_dframe_data,train_dframe_data,test_dframe_data]
    dfs_combine = pd.concat(dfs)
    dfs_combine.drop_duplicates(inplace=True)
    
    if question_mt == "intentYN":
        conditions =[dfs_combine['intentYN'].eq('1.0'),dfs_combine['intentYN'].eq('0.66'),dfs_combine['intentYN'].eq('0.33'),dfs_combine['intentYN'].eq('0.0')]
        choices = ['Intended', 'Probably Intended', 'Probably Not Intended', 'Not Intended']
        dfs_combine['intentYN'] = np.select(conditions, choices)
        dfs_combine = dfs_combine.rename(columns = {'intentYN':'label'})
    else:
        conditions = [dfs_combine['sexYN'].eq('1.0'),dfs_combine['sexYN'].eq('0.5'),dfs_combine['sexYN'].eq('0.0')]
        choices = ['Lewd', 'Maybe Lewd', 'Not Lewd']
        dfs_combine['sexYN'] = np.select(conditions, choices)
        dfs_combine = dfs_combine.rename(columns = {'sexYN':'label'})
    
    dfs_combine = dfs_combine.rename(columns = {'post':'message'})
    label_dict = {index : choices[index] for index in range(0,len(choices))}
    dfs_combine['label'] = dfs_combine['label'].astype('category')
    dfs_combine['label_vector'] = dfs_combine['label'].cat.codes
    cats = dfs_combine.label.astype('category')
    list_of_cats = dict(enumerate(cats.cat.categories))
    annotators = pd.unique(dfs_combine['WorkerId'])
    

    # path = foldername1 + "/" + id +"_annotations.json"
    # dfs_combine.to_json(path,orient='split')
    
    dfs_dev = dfs_combine[dfs_combine.split =='dev']
    path = foldername1 + "/" + id + "_dev.json"
    dfs_dev.to_json(path,orient='split',index=False)
    
    dfs_train = dfs_combine[dfs_combine.split =='train']
    path = foldername1 + "/" + id + "_train.json"
    dfs_train.to_json(path,orient='split',index=False)
    
    dfs_test = dfs_combine[dfs_combine.split =='test']
    path = foldername1 + "/" + id + "_test.json"
    dfs_test.to_json(path,orient='split',index=False)

    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index
    # annotators_parsed = [{x:annotators[x]} for x in range(len(annotators))] #Int indexes 
    dfs_combine = dfs_combine.join(annotators_parsed.set_index('id'), on='WorkerId')

    unique_dataitems = pd.unique(dfs_combine['HITId'])
    unique_dataitems_parsed = pd.DataFrame(unique_dataitems)
    unique_dataitems_parsed = unique_dataitems_parsed.rename(columns={0:'id'})
    unique_dataitems_parsed['Mindex'] = unique_dataitems_parsed.index
    dfs_combine = dfs_combine.join(unique_dataitems_parsed.set_index('id'), on='HITId')
    # data_items_parsed = [{x:unique_dataitems[x]} for x in range(len(unique_dataitems))]

    path = foldername2 + "/" + id + "_complete_with_id.json"
    dfs_combine.to_json(path,orient='split')
    

    dfs_dev = dfs_combine[dfs_combine.split =='dev']
    path = foldername3 + "/" + id + "_dev.json"
    convert_data_pldl_experiments(dfs_dev,choices,'Mindex',path)
    # generate_data_nn(dfs_dev,foldername2,"dev",label_dict,id)
    
    dfs_train = dfs_combine[dfs_combine.split =='train']
    path = foldername3 + "/" + id + "_train.json"
    convert_data_pldl_experiments(dfs_train,choices,'Mindex',path)
    # generate_data_nn(dfs_train,foldername2,"train",label_dict,id)

    dfs_test = dfs_combine[dfs_combine.split =='test']
    path = foldername3 + "/" + id + "_test.json"
    convert_data_pldl_experiments(dfs_test,choices,'Mindex',path)
    # generate_data_nn(dfs_test,foldername2,"test",label_dict,id)
    ds_df = dfs_combine
    ds_df = ds_df.drop(["HITId","WorkerId"],axis=1)
    #TODO start modified annotator model
    # ds_df = ds_df.rename(columns={"Aindex":"rater_id","Mindex":"text_id"})

    # ds_df = ds_df[['text_id','rater_id','label','message','label_vector']]

    ds_df = ds_df.rename(columns={"Aindex":"worker_id","Mindex":"message_id"})
    ds_df = ds_df[['message_id','worker_id','label','message','label_vector','split']]

    dfs_dev = ds_df[ds_df.split =='dev']
    path = foldername1 + "/" + id + "_dev.json"
    dfs_dev.to_json(path,orient='split',index=False)
    
    dfs_train = ds_df[ds_df.split =='train']
    path = foldername1 + "/" + id + "_train.json"
    dfs_train.to_json(path,orient='split',index=False)
    
    dfs_test = ds_df[ds_df.split =='test']
    path = foldername1 + "/" + id + "_test.json"
    dfs_test.to_json(path,orient='split',index=False)

    path = foldername1 + "/" + id +"_annotations.json"
    ds_df.to_json(path,orient='split')

    path = foldername1 + "/" + id +"_label_cats.json"
    save_to_json(list_of_cats,path)

    #TODO end modified model annotators

    X_train = pd.unique(dfs_train['message'])
    X_dev = pd.unique(dfs_dev['message'])
    X_test = pd.unique(dfs_test['message'])
    annotators_array = np.full(len(annotators_parsed),-1)
    generate_data_bert(dfs_train,foldername2,"train",label_dict,"SI",X_train,annotators_array)
    generate_data_bert(dfs_test,foldername2,"test",label_dict,"SI",X_test,annotators_array)
    generate_data_bert(dfs_dev,foldername2,"dev",label_dict,"SI",X_dev,annotators_array)

    
    
def generate_data_nn(data_items,foldername,split_name,label_dict,id):

    path = foldername + "/" + id + "_"+split_name+".json"
    data_items.to_json(path,orient='split',index=False)
    original_dataset = data_items

    path = foldername + "/" + id + "_"+split_name+"_AIL.csv"
    data_items_parsed = convert_labels_hotencoding(data_items,len(label_dict))
    data_items_parsed.to_csv(path,index=False,header=False)
    data_items = data_items[['Mindex','Aindex','label_vector']]

    data_items_item_dist = convert_labels_per_group(data_items,len(label_dict),'Mindex')
    data_items_item_dist.columns = ["item","label"]
    path = foldername + "/" + id + "_"+split_name+"_IL.csv"
    data_items_item_dist.to_csv(path,index=False,header=False)
    Y = data_items_item_dist['label'].to_numpy()
    Y_final = []
    Yi_values = []

    for row in Y:
        row_values = []
        yi_row = []
        total = sum(row)
        for value in row:
            row_values.append(value)
            yi_row.append(value/total)
        Y_final.append(row_values)
        Yi_values.append(yi_row)

    Y = np.asarray(Y_final)
    path = foldername + "/" + "Y_"+split_name+".npy"
    np.save(path, Y)
    Yi_values = np.asarray(Yi_values)
    path = foldername + "/" + "Yi_"+split_name+".npy"
    np.save(path,Yi_values)


    Ii = data_items_item_dist['item'].to_numpy()
    Ii = np.expand_dims(np.asarray(Ii),axis=1)
    path = foldername + "/" + "Ii_"+split_name+".npy"
    np.save(path,Ii)

    data_items_annotator_dist = convert_labels_per_group(data_items,len(label_dict),'Aindex')
    data_items_annotator_dist.columns = ["annotator","label"]
    path = foldername + "/" + id + "_"+split_name+"_AL.csv"
    data_items_annotator_dist.to_csv(path,index=False,header=False)

    Ai = data_items_annotator_dist['annotator'].to_numpy()
    Ai = np.expand_dims(np.asarray(Ai),axis=1)
    path = foldername + "/" + "Ai_"+split_name+".npy"
    np.save(path,Ai)

    Ya_values = []
    Ya_rows = data_items_annotator_dist['label'].to_numpy()
    for row in Ya_rows:
        ya_row = []
        total = sum(row)
        for value in row:
            ya_row.append(value/total)
        Ya_values.append(ya_row)

    Ya = np.asarray(Ya_values)
    path = foldername + "/" + "Ya_"+split_name+".npy"
    np.save(path,Ya)

    data_items_index = pd.unique(original_dataset['Mindex'])
    data_items_embed,embeddings = sentence_embedding(original_dataset,data_items_index)
    path = foldername + "/" + id + "_"+split_name+"_IE.csv"
    data_items_embed.to_csv(path,index=False)
    X = np.asarray(embeddings)
    path = foldername + "/" + "X_"+split_name+".npy"
    np.save(path,X)

    path = foldername + "/" + "Xi_"+split_name+".npy"
    data_items_embed_Xi = data_items_embed.to_numpy()
    np.save(path,data_items_embed_Xi)
    
    
    

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


if __name__== "__main__":
    main()
