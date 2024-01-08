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
from helper_functions import sentence_embedding,convert_data_pldl_experiments,generate_data_bert,create_folder,move_disco_embed_to_co
#from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, 'utils/')
#from utils import gen_data_plot
import tensorflow as tf
import random
from helper_functions import save_to_json
from sklearn.model_selection import train_test_split
# use_original_splits = True


# Sub python preprocess_deepak.py --input_raw_file1 /home/cyril/DataDrive/Experiments/pldl/Datasets/Deepak/toxicity_ratings.json --input_raw_file2 /home/cyril/DataDrive/Experiments/pldl/Datasets/Deepak/classified_data_with_worker_hash.json --id TR --use_original_splits 0 --foldername1 datasets/TR/processed/modeling_annotator --foldername2 datasets/TR/processed/modeling_annotator_NN --foldername3 datasets/TR/processed/PLDL

# Full python preprocess_deepak.py --input_raw_file1 /home/cyril/DataDrive/Experiments/pldl/Datasets/Deepak/toxicity_ratings.json --input_raw_file2 /home/cyril/DataDrive/Experiments/pldl/Datasets/Deepak/121922_classified_data_with_worker_hash.json --id TR --use_original_splits 1 --foldername1 datasets/TRFull/processed/modeling_annotator --foldername2 datasets/TRFull/processed/modeling_annotator_NN --foldername3 datasets/TRFull/processed/PLDL

def deepak_process_line(file_name):
    data_read = []
    with open(file_name) as fh:
        for line in fh:
            data_read.append(json.loads(line))
    return data_read

def read_line_combo(dataset_raw,dataset_ids):
    """
    For each row in the dataset, we take the ratings and the data row, and then for each rating, we
    create a new row with the data row and the rating
    
    :param dataset_raw: the raw data from the json file
    :param dataset_ids: the list of dictionaries that contain the data for each line
    :return: A dictionary of dictionaries.
    """
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_raw_file1", help="Input messages raw annotator file w/o IDS", default = 'toxicity_ratings.json' )
    parser.add_argument("--input_raw_file2", help="Input messages raw annotator file w IDS", default='classified_data_with_worker_hash.json')
    parser.add_argument("--id", help="Identifier")
    parser.add_argument("--use_original_splits", help="Subsample or original dataset",default=1)
    parser.add_argument("--foldername1", help="Main Folder name", default = "datasets/GoEmotions/processed/modeling_annotator")
    parser.add_argument("--foldername2", help="Main Folder name", default = "datasets/GoEmotions/processed/modeling_annotator_nn")
    parser.add_argument("--foldername3", help="Main Folder name", default = "datasets/GoEmotions/processed/pldl")
    args = parser.parse_args()

    raw_input_file1 = args.input_raw_file1
    raw_input_file2 = args.input_raw_file2


    _id = args.id
    use_original_splits = int(args.use_original_splits)
    foldername1 = args.foldername1
    foldername2 = args.foldername2
    foldername3 = args.foldername3
    create_folder(foldername1)
    create_folder(foldername2)
    create_folder(foldername3)
    path = foldername1 + "/"+_id + "_combined.json"
    datase_ids = deepak_process_line(raw_input_file1)
    dataset = deepak_process_line(raw_input_file2)
    processed = read_line_combo(dataset,datase_ids)
    dfs_combine = pd.DataFrame(processed).T
    dfs_combine.drop_duplicates(inplace=True)
    colLabels = pd.unique(dfs_combine['toxic_score'])
    colLabels.sort()
    dfs_combine = dfs_combine[['comment','comment_id','toxic_score','worker_id']]
    dfs_combine = dfs_combine.rename({'comment':'message','comment_id': 'text_id', 'toxic_score':'label','worker_id':'rater_id'}, axis=1) 

    label_dict = {index : colLabels[index] for index in range(0,len(colLabels))}
    dfs_combine['label'] = dfs_combine['label'].astype('category')
    dfs_combine['label_vector'] = dfs_combine['label'].cat.codes
    cats = dfs_combine.label.astype('category')
    list_of_cats = dict(enumerate(cats.cat.categories))
    annotators = pd.unique(dfs_combine['rater_id'])
    # dfs_combine = dfs_combine.rename(columns = {'text':'message'})
    path = foldername1 + "/"+_id +"_annotations.json"
    dfs_combine.to_json(path,orient='split')
    
    # dev_items,train_items,test_items = read_splits(dev_input_file, train_input_file, test_input_file)
    item_list = pd.unique(dfs_combine['text_id'])
    item_list = [x for x in item_list]
    pdb.set_trace()
    if use_original_splits!=1:
        print("***down sampling***")
        item_list = random.sample(item_list,2000)

    train_items, test_items = train_test_split(item_list, test_size=0.5, random_state=42)
    test_items, dev_items = train_test_split(test_items, test_size=0.5, random_state=42)

    dfs_dev = dfs_combine[dfs_combine.text_id.isin(dev_items)]
    path = foldername1 + "/" + _id + "_dev.json"
    dfs_dev.to_json(path,orient='split',index=False)
    
    dfs_train = dfs_combine[dfs_combine.text_id.isin(train_items)]
    path = foldername1 + "/" + _id + "_train.json"
    dfs_train.to_json(path,orient='split',index=False)
    
    dfs_test = dfs_combine[dfs_combine.text_id.isin(test_items)]
    path = foldername1 + "/" + _id + "_test.json"
    dfs_test.to_json(path,orient='split',index=False)

    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index
    # annotators_parsed = [{x:annotators[x]} for x in range(len(annotators))] #Int indexes 

    dfs_combine = dfs_combine.join(annotators_parsed.set_index('id'), on='rater_id')

    unique_dataitems = pd.unique(dfs_combine['text_id'])
    unique_dataitems_parsed = pd.DataFrame(unique_dataitems)
    unique_dataitems_parsed = unique_dataitems_parsed.rename(columns={0:'id'})
    unique_dataitems_parsed['Mindex'] = unique_dataitems_parsed.index
    dfs_combine = dfs_combine.join(unique_dataitems_parsed.set_index('id'), on='text_id')
    # data_items_parsed = [{x:unique_dataitems[x]} for x in range(len(unique_dataitems))]

    path = foldername2 + "/" + _id + "_complete_with_id.json"
    dfs_combine.to_json(path,orient='split')
    
    dfs_dev = dfs_combine[dfs_combine.text_id.isin(dev_items)]
    path = foldername3 + "/" + _id + "_dev.json"
    convert_data_pldl_experiments(dfs_dev,colLabels,'Mindex',path)
    # generate_data_nn(dfs_dev,foldername2,"dev",label_dict,_id)
    
    dfs_train = dfs_combine[dfs_combine.text_id.isin(train_items)]
    path = foldername3 + "/" + _id + "_train.json"
    convert_data_pldl_experiments(dfs_train,colLabels,'Mindex',path)
    # generate_data_nn(dfs_train,foldername2,"train",label_dict,_id)

    dfs_test = dfs_combine[dfs_combine.text_id.isin(test_items)]
    path = foldername3 + "/" + _id + "_test.json"
    convert_data_pldl_experiments(dfs_test,colLabels,'Mindex',path)
    # generate_data_nn(dfs_test,foldername2,"test",label_dict,_id)

    # train_items_nn = train_items.rename({'message_id': 'Mindex', 'worker_id': 'Aindex', 'label':'label_vector'}, axis=1) 
    # test_items_nn = test_items.rename({'message_id': 'Mindex', 'worker_id': 'Aindex', 'label':'label_vector'}, axis=1) 
    # dev_items_nn = dev_items.rename({'message_id': 'Mindex', 'worker_id': 'Aindex', 'label':'label_vector'}, axis=1) 
    # annotators_parsed = pd.unique(dfs_combine['Aindex'])
    # annotators_parsed = pd.DataFrame(annotators)
    # annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index

    annotators_array = np.full(len(annotators_parsed),-1)
    X_train = pd.unique(dfs_train['message'])
    X_dev = pd.unique(dfs_dev['message'])
    X_test = pd.unique(dfs_test['message'])
    generate_data_bert(dfs_train,foldername2,"train",label_dict,"TR",X_train,annotators_array)
    generate_data_bert(dfs_test,foldername2,"test",label_dict,"TR",X_test,annotators_array)
    generate_data_bert(dfs_dev,foldername2,"dev",label_dict,"TR",X_dev,annotators_array)
    move_disco_embed_to_co(foldername2,foldername3)


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

# def generate_data_nn(data_items,foldername,split_name,label_dict,_id):

#     path = foldername + "/" + _id + "_"+split_name+".json"
#     data_items.to_json(path,orient='split',index=False)
#     original_dataset = data_items

#     path = foldername + "/" + _id + "_"+split_name+"_AIL.csv"
#     data_items_parsed = convert_labels_hotencoding(data_items,len(label_dict))
#     data_items_parsed.to_csv(path,index=False,header=False)
#     data_items = data_items[['Mindex','Aindex','label_vector']]

#     data_items_item_dist = convert_labels_per_group(data_items,len(label_dict),'Mindex')
#     data_items_item_dist.columns = ["item","label"]
#     path = foldername + "/" + _id + "_"+split_name+"_IL.csv"
#     data_items_item_dist.to_csv(path,index=False,header=False)
#     Y = data_items_item_dist['label'].to_numpy()
#     Y_final = []
#     Yi_values = []

#     for row in Y:
#         row_values = []
#         yi_row = []
#         total = sum(row)
#         for value in row:
#             row_values.append(value)
#             yi_row.append(value/total)
#         Y_final.append(row_values)
#         Yi_values.append(yi_row)

#     Y = np.asarray(Y_final)
#     path = foldername + "/" + "Y_"+split_name+".npy"
#     np.save(path, Y)
#     Yi_values = np.asarray(Yi_values)
#     path = foldername + "/" + "Yi_"+split_name+".npy"
#     np.save(path,Yi_values)


#     Ii = data_items_item_dist['item'].to_numpy()
#     Ii = np.expand_dims(np.asarray(Ii),axis=1)
#     path = foldername + "/" + "Ii_"+split_name+".npy"
#     np.save(path,Ii)

#     data_items_annotator_dist = convert_labels_per_group(data_items,len(label_dict),'Aindex')
#     data_items_annotator_dist.columns = ["annotator","label"]
#     path = foldername + "/" + _id + "_"+split_name+"_AL.csv"
#     np.set_printoptions(linewidth=100000)
#     data_items_annotator_dist.to_csv(path,index=False,header=False)

#     path = foldername + "/" + _id + "_"+split_name+"_AL.json"
#     data_items_annotator_dist.to_json(path,orient='split',index=False)
#     # save_to_json(data_items_annotator_dist,path)

#     Ai = data_items_annotator_dist['annotator'].to_numpy()
#     Ai = np.expand_dims(np.asarray(Ai),axis=1)
#     path = foldername + "/" + "Ai_"+split_name+".npy"
#     np.save(path,Ai)

#     Ya_values = []
#     Ya_rows = data_items_annotator_dist['label'].to_numpy()
#     for row in Ya_rows:
#         ya_row = []
#         total = sum(row)
#         for value in row:
#             ya_row.append(value/total)
#         Ya_values.append(ya_row)

#     Ya = np.asarray(Ya_values)
#     path = foldername + "/" + "Ya_"+split_name+".npy"
#     np.save(path,Ya)

#     data_items_index = pd.unique(original_dataset['Mindex'])
#     data_items_embed,embeddings = sentence_embedding(original_dataset,data_items_index)
#     path = foldername + "/" + _id + "_"+split_name+"_IE.csv"
#     data_items_embed.to_csv(path,index=False)
    
#     X = np.asarray(embeddings)
#     path = foldername + "/" + "X_"+split_name+".npy"
#     np.save(path,X)

#     path = foldername + "/" + "Xi_"+split_name+".npy"
#     data_items_embed_Xi = data_items_embed.to_numpy()
#     np.save(path,data_items_embed_Xi)
#     #gen_data_plot(X, tf.cast(Y,dtype=tf.float32), use_tsne=False,fname="Xi_"+split_name,out_dir=foldername +"/")
    
   
       
# def read_splits(dev_input_file, train_input_file, test_input_file):
#     data_dev = pd.read_csv(dev_input_file, sep = "\t", header = None)
#     dev_items = data_dev[2].tolist()
#     data_train = pd.read_csv(train_input_file, sep = "\t", header = None)
#     train_items = data_train[2].tolist()
#     data_test = pd.read_csv(test_input_file, sep = "\t", header = None)
#     test_items = data_test[2].tolist()
    
#     return dev_items,train_items,test_items    

    
    
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
