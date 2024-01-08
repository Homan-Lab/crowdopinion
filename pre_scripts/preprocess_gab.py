import pandas as pd
import pdb
from itertools import groupby
from collections import OrderedDict,defaultdict
from collections import Counter
import json
import argparse
import os
import numpy as np
from helper_functions import sentence_embedding,convert_data_pldl_experiments,generate_data_bert,create_folder,move_disco_embed_to_co
#from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, 'utils/')
#from utils import gen_data_plot
# import tensorflow as tf
import random
from helper_functions import save_to_json
from sklearn.model_selection import train_test_split
use_original_splits = False

# from ldl_utils import read_json
# from split_data import split_items_train_dev_test
# from helper_functions import read_original_split,generate_pd,get_feature_vectors,compile_tweet_dict,get_data_dict

# python3 preprocess_gab.py --input_MT_raw_file datasets/gab/GabHateCorpus_annotations.tsv --id gab --foldername1 datasets/gab/processed/modeling_annotator --foldername2 datasets/gab/processed/modeling_annotator_nn --foldername3 datasets/gab/processed/pldl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_MT_raw_file", help="Input messages raw CSV file 1")
    parser.add_argument("--id", help="Identifier",default="PP")
    parser.add_argument("--foldername1", help="Main Folder name", default = "datasets/PP/processed/modeling_annotator")
    parser.add_argument("--foldername2", help="Main Folder name", default = "datasets/PP/processed/modeling_annotator_nn")
    parser.add_argument("--foldername3", help="Main Folder name", default = "datasets/PP/processed/pldl")
    args = parser.parse_args()

    raw_input_file = args.input_MT_raw_file
    col_tweet_ID = "ID"
    col_tweet_text = "Text"
    # colLabels = ["DEM_True", "DEM_REP_True", "DEM_IND_True", "REP_True", "REP_DEM_True", "REP_IND_True", "IND_True", "IND_DEM_True", "IND_REP_True","DEM_False", "DEM_REP_False", "DEM_IND_False", "REP_False", "REP_DEM_False", "REP_IND_False", "IND_False", "IND_DEM_False", "IND_REP_False"]
    colLabels = [0,1] #0 = not hate, 1 = hate. processing bumps this up one 1
    
    _id = args.id
    foldername1 = args.foldername1
    foldername2 = args.foldername2
    foldername3 = args.foldername3
    create_folder(foldername1)
    create_folder(foldername2)
    create_folder(foldername3)
    path = foldername1 + "/"+_id + "_combined.json"
    worker_id_mt = "annotator_id"
    

    dfs_combine = pd.read_csv(raw_input_file, delimiter="\t")
    dfs_combine = dfs_combine[["ID", "Annotator", "Text", "Hate"]]
    dfs_combine = dfs_combine.rename(columns={
        "ID": "comment_id",
        "Annotator": "annotator_id",
        "Text": "message",
        "Hate": "label"
    })
    # dfs_combine['label'] = dfs_combine['label']+1
    dfs_combine.drop_duplicates(inplace=True)

    label_dict = {index : colLabels[index] for index in range(0,len(colLabels))}
    dfs_combine['label'] = dfs_combine['label'].astype('category')
    dfs_combine['label_vector'] = dfs_combine['label'].cat.codes
    cats = dfs_combine.label.astype('category')
    list_of_cats = dict(enumerate(cats.cat.categories))
    annotators = pd.unique(dfs_combine["annotator_id"])
    dfs_combine = dfs_combine.rename(columns = {'text':'message'})
    data_items = pd.unique(dfs_combine['comment_id'])
    path = foldername1 + "/"+_id +"_annotations.json"

    dfs_combine.to_json(path,orient='split')
    train_items,dev_items = train_test_split(data_items,test_size=0.4)
    dev_items,test_items = train_test_split(dev_items,test_size=0.5)


    dfs_dev = dfs_combine[dfs_combine.comment_id.isin(dev_items)]
    path = foldername1 + "/" + _id + "_dev.json"
    dfs_dev.to_json(path,orient='split',index=False)
    
    dfs_train = dfs_combine[dfs_combine.comment_id.isin(train_items)]
    path = foldername1 + "/" + _id + "_train.json"
    dfs_train.to_json(path,orient='split',index=False)
    
    dfs_test = dfs_combine[dfs_combine.comment_id.isin(test_items)]
    path = foldername1 + "/" + _id + "_test.json"
    dfs_test.to_json(path,orient='split',index=False)

    ds_df = dfs_combine
    # ds_df = ds_df.drop([col_tweet_ID,worker_id_mt],axis=1)
    # ds_df = ds_df.rename(columns={"Aindex":worker_id_mt,"Mindex":col_tweet_ID})
    ds_df = ds_df[['comment_id',worker_id_mt,'label','message','label_vector']]

    path = foldername1 + "/"+_id +"_annotations.json"
    ds_df.to_json(path,orient='split')
    # pdb.set_trace()
    # path = foldername2 + "/" + _id + "_complete_with_id.json"
    # dfs_combine.to_json(path,orient='split')
    
    dfs_dev = dfs_combine[dfs_combine.comment_id.isin(dev_items)]
    path = foldername3 + "/" + _id + "_dev.json"
    convert_data_pldl_experiments(dfs_dev,colLabels,'comment_id',path)
    # generate_data_nn(dfs_dev,foldername2,"dev",label_dict,_id)
    
    dfs_train = dfs_combine[dfs_combine.comment_id.isin(train_items)]
    path = foldername3 + "/" + _id + "_train.json"
    convert_data_pldl_experiments(dfs_train,colLabels,'comment_id',path)
    # generate_data_nn(dfs_train,foldername2,"train",label_dict,_id)

    dfs_test = dfs_combine[dfs_combine.comment_id.isin(test_items)]
    path = foldername3 + "/" + _id + "_test.json"
    convert_data_pldl_experiments(dfs_test,colLabels,'comment_id',path)
    # generate_data_nn(dfs_test,foldername2,"test",label_dict,_id)

    annotators_parsed = pd.unique(dfs_combine[worker_id_mt])
    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index

    annotators_array = np.full(len(annotators_parsed),-1)
    X_train = pd.unique(dfs_train['message'])
    X_dev = pd.unique(dfs_dev['message'])
    X_test = pd.unique(dfs_test['message'])
    
    dfs_train = rename_df_embedding(dfs_train)
    dfs_test = rename_df_embedding(dfs_test)
    dfs_dev = rename_df_embedding(dfs_dev)

    generate_data_bert(dfs_train,foldername2,"train",label_dict,_id,X_train,annotators_array)
    generate_data_bert(dfs_test,foldername2,"test",label_dict,_id,X_test,annotators_array)
    generate_data_bert(dfs_dev,foldername2,"dev",label_dict,_id,X_dev,annotators_array)

    move_disco_embed_to_co(foldername2,foldername3)



def rename_df_embedding(df_to_rename):
    df_to_rename['Aindex'] = df_to_rename['annotator_id']
    df_to_rename['Mindex'] = df_to_rename['comment_id']
    return df_to_rename

def label_grouping_annotators(annotators,dframe_labels,label_dict): #dframe_data,col_tweet_text,col_tweet_ID,col_label):
    results = []
    for worker_id in annotators:
        labels = {}
        data = {}
        labels_for_annotator = dframe_labels.loc[dframe_labels['annotator_id'] == worker_id]
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

def generate_data_nn(data_items,foldername,split_name,label_dict,_id):

    path = foldername + "/" + _id + "_"+split_name+".json"
    data_items.to_json(path,orient='split',index=False)
    original_dataset = data_items

    path = foldername + "/" + _id + "_"+split_name+"_AIL.csv"
    data_items_parsed = convert_labels_hotencoding(data_items,len(label_dict))
    data_items_parsed.to_csv(path,index=False,header=False)
    data_items = data_items[['Mindex','Aindex','label_vector']]

    data_items_item_dist = convert_labels_per_group(data_items,len(label_dict),'Mindex')
    data_items_item_dist.columns = ["item","label"]
    path = foldername + "/" + _id + "_"+split_name+"_IL.csv"
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
    path = foldername + "/" + _id + "_"+split_name+"_AL.csv"
    np.set_printoptions(linewidth=100000)
    data_items_annotator_dist.to_csv(path,index=False,header=False)

    path = foldername + "/" + _id + "_"+split_name+"_AL.json"
    data_items_annotator_dist.to_json(path,orient='split',index=False)
    # save_to_json(data_items_annotator_dist,path)

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
    path = foldername + "/" + _id + "_"+split_name+"_IE.csv"
    data_items_embed.to_csv(path,index=False)
    
    X = np.asarray(embeddings)
    path = foldername + "/" + "X_"+split_name+".npy"
    np.save(path,X)

    path = foldername + "/" + "Xi_"+split_name+".npy"
    data_items_embed_Xi = data_items_embed.to_numpy()
    np.save(path,data_items_embed_Xi)
    #gen_data_plot(X, tf.cast(Y,dtype=tf.float32), use_tsne=False,fname="Xi_"+split_name,out_dir=foldername +"/")
    
   
       
def read_splits(dev_input_file, train_input_file, test_input_file):
    data_dev = pd.read_csv(dev_input_file, sep = "\t", header = None)
    dev_items = data_dev[2].tolist()
    data_train = pd.read_csv(train_input_file, sep = "\t", header = None)
    train_items = data_train[2].tolist()
    data_test = pd.read_csv(test_input_file, sep = "\t", header = None)
    test_items = data_test[2].tolist()
    
    return dev_items,train_items,test_items    

    
    
def unpivot(dframe,col_tweet_ID,worker_id_mt, col_tweet_text,colLabels):
    df = dframe.melt(id_vars=[col_tweet_ID,worker_id_mt, col_tweet_text], value_vars=colLabels)
    df = df[df["value"]>0]
    df = df.drop(columns = ["value"])
    df = df.rename(columns = {'variable':'label','id':'text_id'})
    cols = ["text_id",worker_id_mt,"label",col_tweet_text]
    df = df[cols]
    
    return df   
    
    
def csv_read(csvLocation,col_tweet_ID,col_tweet_text,col_label,col_worker_id):
    dframe = pd.read_csv(csvLocation, usecols = [col_tweet_ID,col_tweet_text,col_worker_id]+col_label)
    cols = [col_tweet_ID,col_worker_id]+col_label+[col_tweet_text]
    dframe = dframe[cols]

    return dframe


if __name__== "__main__":
    main()
