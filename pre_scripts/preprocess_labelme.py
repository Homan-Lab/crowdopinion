import pandas as pd
import numpy as np
import pdb
from itertools import groupby
from collections import OrderedDict,defaultdict
from collections import Counter
from sklearn.model_selection import train_test_split
import json
import argparse
import os
from sklearn.decomposition import PCA
from helper_functions import convert_data_pldl_experiments
import random
N_CLASSES = 8

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_x", help="Input X filename")
    # parser.add_argument("--input_ground_truth", help="Ground truth")
    # parser.add_argument("--input_mt_labels", help="Labels from MTruk")

    # parser.add_argument("--id", help="Identifier")
    # parser.add_argument("--foldername", help="Main Folder name")
    # args = parser.parse_args()
    folder = '/home/cyril/DataDrive/Experiments/pldl/modeling_annotators_NN/datasets/labelMe_OriginalJan4'
    input_x = np.load("/home/cyril/DataDrive/Experiments/pldl/Datasets/LabelMe/data_train_vgg16.npy")
    input_gt = np.load("/home/cyril/DataDrive/Experiments/pldl/Datasets/LabelMe/labels_train.npy")
    answers = np.load("/home/cyril/DataDrive/Experiments/pldl/Datasets/LabelMe/answers.npy")
    # pldl_path = "/mnt/DataDrive/Experiments/pldl/experimental_code/data/labelMe/original/"
    pldl_path = "/mnt/DataDrive/Experiments/pldl/experimental_code/data/labelMeFullJan4/"
    N_ANNOT = answers.shape[1]
    create_folder(folder)
    create_folder(pldl_path)
    input_x = input_x.reshape(10000,4*4*512)

    filter_n_annotators = False
    min_annotators = 3
    use_original_splits = True
    processed = []
    rows_only = []
    df = pd.DataFrame(columns=['message_id','worker_id','label'])
    features = []
    for i in range(len(answers)):
        labels = {}
        row = {}
        feature_row = {}
        feature_row['message_id'] = i
        feature_row['message'] = input_x[i].tolist()

        # pdb.set_trace()
        if filter_n_annotators:
            label_counter = Counter(answers[i])
            if (label_counter[-1]>(N_ANNOT-min_annotators)):
                continue
        for r in range(N_ANNOT):
            if answers[i,r] != -1:
                single_label_row = {}
                labels[r] = answers[i,r]
                single_label_row['message_id'] = i
                single_label_row['worker_id'] = r
                single_label_row['label'] = answers[i,r]
                rows_only.append(single_label_row)
                df = df.append({'message_id':i,'worker_id':r,'label':answers[i,r]},ignore_index=True)
        row['label']= labels
        row['message_id'] = i
        features.append(feature_row)
        # feature_row['message_id'] = i
        # feature_row['message'] = input_x[i].tolist()
        processed.append(row)
        # features.append(feature_row)

    item_list = pd.unique(df['message_id'])
    item_list = [x for x in item_list]

    if use_original_splits==False:
        item_list = random.sample(item_list,2000)
        features = filter_out_sampled(item_list,features)
   
    label_dict = [x for x in range(8)]
    # rows_only = pd.DataFrame(rows_only)
    data_items_item_dist = convert_labels_per_group(df,len(label_dict),'message_id')
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
    path = folder + "/" + "Y_whole.npy"
    np.save(path, Y)

    X_train, X_test, y_train, y_test = train_test_split(features, item_list, test_size=0.5, random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    rows_only = pd.DataFrame(rows_only)
    label_dict = [x for x in range(8)]
    train_items = rows_only[rows_only.message_id.isin(y_train)]
    test_items = rows_only[rows_only.message_id.isin(y_test)]
    dev_items = rows_only[rows_only.message_id.isin(y_dev)]
    # pdb.set_trace()

    annotators = pd.unique(df['worker_id'])
    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index
    annotators_array = np.full(len(annotators_parsed['Aindex']),-1)

    pdb.set_trace()
    # generate_data_nn(rows_only,folder,"train",label_dict,"labelMe",features,annotators_array)

    generate_data_nn(train_items,folder,"train",label_dict,"labelMe",X_train,annotators_array)
    generate_data_nn(test_items,folder,"test",label_dict,"labelMe",X_test,annotators_array)
    generate_data_nn(dev_items,folder,"dev",label_dict,"labelMe",X_dev,annotators_array)

    X_rows = pd.DataFrame(X_train)
    labels = train_items.join(X_rows.set_index('message_id'),on='message_id')
    path = pldl_path + "labelMe_train.json"
    convert_data_pldl_experiments(labels,label_dict,'message_id',path)
    
    X_rows = pd.DataFrame(X_dev)
    labels = dev_items.join(X_rows.set_index('message_id'),on='message_id')
    path = pldl_path + "labelMe_dev.json"
    convert_data_pldl_experiments(labels,label_dict,'message_id',path)
    
    X_rows = pd.DataFrame(X_test)
    labels = test_items.join(X_rows.set_index('message_id'),on='message_id')
    path = pldl_path + "labelMe_test.json"
    convert_data_pldl_experiments(labels,label_dict,'message_id',path)

def filter_out_sampled(sampled_set,features):
    processed_rows = []
    for sampled_row in sampled_set:
        row = features[sampled_row]
        processed_rows.append(row)
    return processed_rows


def generate_data_nn(data_items,foldername,split_name,label_dict,id,features,annotators_array):
    data_items = data_items[['message_id','worker_id','label']]
    path = foldername + "/" + id + "_"+split_name+".json"
    data_items.to_json(path,orient='split',index=False)
    original_dataset = data_items

    path = foldername + "/" + id + "_"+split_name+"_AIL.csv"
    data_items_parsed = convert_labels_hotencoding(data_items,len(label_dict))
    data_items_parsed.to_csv(path,index=False,header=False)
    data_items = data_items[['message_id','worker_id','label']]

    data_items_item_dist = convert_labels_per_group(data_items,len(label_dict),'message_id')
    data_items_item_dist = data_items_item_dist[['message_id','label']]
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

    data_items_annotator_dist = convert_labels_per_group(data_items,len(label_dict),'worker_id')
    data_items_annotator_dist = data_items_annotator_dist[['worker_id','label']]
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
    
    embeddings = [] #{}
    embeddings_only = []
    for feature_row in features:
        row = {}
        # embeddings[feature_row['message_id']] = feature_row['message']
        embeddings_only.append(feature_row['message'])
        row['item'] = feature_row['message_id']
        row['embedding'] = feature_row['message']
        embeddings.append(row)
    path = foldername + "/" + "Xi_"+split_name+".npy"
    data_items_embed_Xi = pd.DataFrame(embeddings)
    data_items_embed_Xi = data_items_embed_Xi.to_numpy() #np.asarray(embeddings)
    np.save(path,data_items_embed_Xi,fix_imports=True)

    X = np.asarray(embeddings_only)
    path = foldername + "/" + "X_"+split_name+".npy"
    np.save(path,X,fix_imports=True)

    crowd_layer = generate_annotator_label_crowdlayer(annotators_array,data_items)
    path = foldername + "/" + "YAI_"+split_name+".npy"
    np.save(path,crowd_layer)

def generate_annotator_label_crowdlayer(annotator_array,data_items):

    parsed_data = []
    unique_message_ids = np.unique(data_items['message_id'])
    # parsed_data = {message_id:annotator_array for message_id in unique_message_ids}
    for message_id in unique_message_ids:
        rows = data_items.loc[data_items['message_id'] == message_id]
        annotator_choices = np.zeros(len(annotator_array)) - 1
        # pdb.set_trace()
        for index,row in rows.iterrows():
            annotator_choices[row['worker_id']] = row['label']
        annotator_choices = annotator_choices.astype(int)
        parsed_data.append(annotator_choices)
    return parsed_data

def convert_labels_hotencoding(data_items,no_classes):
    hotencoded = []

    for index,row in data_items.iterrows():
        labels = np.zeros(no_classes)
        labels[row['label']] = 1
        parsed_row = {}
        parsed_row['item'] = row['message_id']
        parsed_row['annotator'] = row['worker_id']
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
            labels[item['label']]+=1
        encoded_row[grouping_category] = row
        encoded_row['label'] = labels.astype(int)
        encoded.append(encoded_row)
    return pd.DataFrame(encoded)

def create_folder(folderpath):
    # Check whether the specified path exists or not
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

if __name__== "__main__":
    main()
