from collections import defaultdict,Counter
import pandas as pd
import json
import numpy as np
import pdb
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def move_disco_embed_to_co(disco_path,co_path):
    create_folder(co_path+"bert_embeddings")
    splits = ["train","dev","test"]
    for split in splits:
        vects_path = disco_path+"/X_"+split+".npy"
        vects = np.load(vects_path,allow_pickle=True)
        np.save(co_path+"bert_embeddings/X_"+split+".npy",vects)
        del vects
        print("Saved X_"+split+".npy")
        
def convert_data_pldl_experiments(dframe,label_dict,grouping_category,file_path):
    parsed_results = {}
    parsed_results['data'] = label_grouping_general(dframe,label_dict,grouping_category)
    parsed_results['dictionary'] = [str(x) for x in label_dict]
    save_to_json(parsed_results,file_path)


def save_to_json(data,outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))

def label_grouping_general(dframe,label_dict,grouping_category):
    result = []
    unique_data_items = pd.unique(dframe[grouping_category])
    for row in unique_data_items:
        row_value = {}
        labels = {}
        row_value['message_id'] = int(row)
        items = dframe.loc[dframe[grouping_category] == row]
        items_counter_str = items.astype({"label": str}) #convert to string to match counters
        label_counter = Counter(items_counter_str['label'])
        row_value['message'] = items.head(1)['message'].values[0]
        label_sum = sum(label_counter.values())
        
        if label_sum == 0:
            pdb.set_trace()
        for label in label_dict:            
            labels[label] = label_counter[str(label)]
        row_value['labels'] = labels

        result.append(row_value)
    return result

def sentence_embedding(data_items,data_index):
    # embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2') #overall best score 384
    embedder = SentenceTransformer('paraphrase-mpnet-base-v2') #overall best score for clustering 768
    # embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') #best score with Twitter
    # embedder = SentenceTransformer('all-mpnet-base-v2')
    # embedder = SentenceTransformer('all-MiniLM-L12-v2')
    encoded = []
    embeddings = []
    for index in tqdm(data_index):
        encoded_row = {}
        encoded_row['item'] = index
        row = data_items[data_items.Mindex==index].iloc[0]
        row = row['message']
        embedding = embedder.encode(row,normalize_embeddings=True)
        embedding  = [x for x in embedding]
        encoded_row['embedding'] = embedding
        embeddings.append(embedding)
        encoded.append(encoded_row)
    return pd.DataFrame(encoded),embeddings


        
def generate_data_bert(data_items,foldername,split_name,label_dict,id,features,annotators_array):
    
    print("********** Processing Split: ",split_name," **********")

    np.set_printoptions(linewidth=100000)
    data_items_features = data_items
    path = foldername + "/" + id + "_"+split_name+".json"
    data_items.to_json(path,orient='split',index=False)
    original_dataset = data_items

    path = foldername + "/" + id + "_"+split_name+"_AIL.csv"
    data_items_parsed = convert_labels_hotencoding(data_items,len(label_dict))
    data_items_parsed.to_csv(path,index=False,header=False)
    
    # parse with message columns
    data_items_parsed = convert_labels_hotencoding_text(data_items,len(label_dict))
    path = foldername + "/" + id + "_"+split_name+"_AIL_data.csv"
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


    data_items_embed,embeddings = sentence_embedding(data_items_features,data_items_index)
    path = foldername + "/" + id + "_"+split_name+"_IE.csv"
    data_items_embed.to_csv(path,index=False)
    X = np.asarray(embeddings)
    path = foldername + "/" + "X_"+split_name+".npy"
    np.save(path,X)

    path = foldername + "/" + "Xi_"+split_name+".npy"
    data_items_embed_Xi = data_items_embed.to_numpy()
    np.save(path,data_items_embed_Xi)
    
    crowd_layer = generate_annotator_label_crowdlayer(annotators_array,data_items)
    path = foldername + "/" + "YAI_"+split_name+".npy"
    np.save(path,crowd_layer)


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

def generate_annotator_label_crowdlayer(annotator_array,data_items):

    parsed_data = []
    unique_message_ids = np.unique(data_items['Mindex'])
    # parsed_data = {message_id:annotator_array for message_id in unique_message_ids}
    
    for message_id in unique_message_ids:
        rows = data_items.loc[data_items['Mindex'] == message_id]
        annotator_choices = np.zeros(len(annotator_array)) - 1
        for index,row in rows.iterrows():
            annotator_choices[row['Aindex']] = row['label_vector']
        annotator_choices = annotator_choices.astype(int)
        parsed_data.append(annotator_choices)
    return parsed_data

def convert_labels_hotencoding_text(data_items,no_classes):
    hotencoded = []

    for index,row in data_items.iterrows():
        labels = np.zeros(no_classes)
        labels[row['label_vector']] = 1
        parsed_row = {}
        parsed_row['item'] = row['Mindex']
        parsed_row['annotator'] = row['Aindex']
        parsed_row['label'] = labels.astype(int)
        parsed_row['message'] = row['message']
        hotencoded.append(parsed_row)

    return pd.DataFrame(hotencoded)

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

def create_folder(folderpath):
    # Check whether the specified path exists or not
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)