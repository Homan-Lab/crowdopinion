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
from helper_functions import convert_data_pldl_experiments,generate_data_bert,move_disco_embed_to_co,create_folder
import random
N_CLASSES = 8

def main():
    folder = "./MR/DisCo"
    DATA_PATH = "./MR/RAW/MovieReviews/"
    input_x = read_texts(DATA_PATH+"texts_train.txt")
    answers = pd.read_csv(DATA_PATH+"answers.txt", header=None, delimiter=" ")
    answers = answers.iloc[:, :-1]
    print("AMT answers matrix shape: %s" % str(answers.shape))
    N_ANNOT = answers.shape[1]
    print("Num. annotators: %d" % N_ANNOT)

    co_path = "./MR/co/"
    N_ANNOT = answers.shape[1]

    create_folder(co_path)
    create_folder(folder)

    filter_n_annotators = False
    min_annotators = 3
    use_original_splits = True
    processed = []
    rows_only = []
    df = pd.DataFrame(columns=['message_id','worker_id','label'])
    # list_to_df = []
    features = []
    for i in range(len(answers)):
        labels = {}
        row = {}
        feature_row = {}
        feature_row['message_id'] = i
        feature_row['message'] = input_x[i]
        features.append(feature_row)
        answers_row = answers.iloc[[i]].values
        answers_row = answers_row[0]
        # if filter_n_annotators:
        #     label_counter = Counter(answers_row)
        #     if (label_counter[-1]>(N_ANNOT-min_annotators)):
        #         continue

        for r in range(N_ANNOT):
            label_value = int(answers_row[r]*10)
            if answers_row[r] != -1 and label_value != -1 and label_value != 11:
                single_label_row = {}
                labels[r] = label_value
                single_label_row['message_id'] = i
                single_label_row['worker_id'] = r
                single_label_row['label'] = label_value
                single_label_row['message'] = input_x[i]
                rows_only.append(single_label_row)
                # list_to_df.append()
                # df = df.append({'message_id':i,'worker_id':r,'label':label_value,'message':input_x[i]},ignore_index=True)
        row['label']= labels
        row['message_id'] = i
        # feature_row['message_id'] = i
        # feature_row['message'] = input_x[i].tolist()
        processed.append(row)
        # features.append(feature_row)
    df = pd.DataFrame.from_records(rows_only)
    df.message_id = df.message_id.astype(int)
    df.worker_id = df.worker_id.astype(int)
    item_list = pd.unique(df['message_id'])
    item_list = [x for x in item_list]
    
    if use_original_splits==False:
        item_list = random.sample(item_list,2000)
        features = filter_out_sampled(item_list,features)

    label_dict = [x for x in range(len(pd.unique(df['label'])))]
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
    
    label_dict = pd.unique(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(features, item_list, test_size=0.5, random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    rows_only = pd.DataFrame(rows_only)
    label_dict = [x for x in range(len(label_dict))]
    train_items = rows_only[rows_only.message_id.isin(y_train)]
    test_items = rows_only[rows_only.message_id.isin(y_test)]
    dev_items = rows_only[rows_only.message_id.isin(y_dev)]

    annotators = pd.unique(df['worker_id'])
    annotators_parsed = pd.DataFrame(annotators)
    annotators_parsed = annotators_parsed.rename(columns={0:'id'})
    annotators_parsed['Aindex'] = annotators_parsed.index

    annotators_array = np.full(len(annotators_parsed['Aindex']),-1)

    train_items_nn = train_items.rename({'message_id': 'Mindex', 'worker_id': 'Aindex', 'label':'label_vector'}, axis=1) 
    test_items_nn = test_items.rename({'message_id': 'Mindex', 'worker_id': 'Aindex', 'label':'label_vector'}, axis=1) 
    dev_items_nn = dev_items.rename({'message_id': 'Mindex', 'worker_id': 'Aindex', 'label':'label_vector'}, axis=1) 

    generate_data_bert(train_items_nn,folder,"train",label_dict,"MR",X_train,annotators_array)
    generate_data_bert(test_items_nn,folder,"test",label_dict,"MR",X_test,annotators_array)
    generate_data_bert(dev_items_nn,folder,"dev",label_dict,"MR",X_dev,annotators_array)

    labels = train_items #.join(X_rows.set_index('message_id'),on='message_id')
    path = co_path + "MR_train.json"
    convert_data_pldl_experiments(labels,label_dict,'message_id',path)

    labels = dev_items #.join(X_rows.set_index('message_id'),on='message_id')
    path = co_path + "MR_dev.json"
    convert_data_pldl_experiments(labels,label_dict,'message_id',path)
    
    labels = test_items #.join(X_rows.set_index('message_id'),on='message_id')
    path = co_path + "MR_test.json"
    convert_data_pldl_experiments(labels,label_dict,'message_id',path)

    move_disco_embed_to_co(folder,co_path)


def read_texts(filename):
    f = open(filename)
    data = [line.strip() for line in f]
    f.close()
    return data

def filter_out_sampled(sampled_set,features):
    processed_rows = []
    for sampled_row in sampled_set:
        row = features[sampled_row]
        processed_rows.append(row)
    return processed_rows


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

if __name__== "__main__":
    main()
