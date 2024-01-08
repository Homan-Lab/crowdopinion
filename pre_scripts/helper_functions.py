#!/usr/bin/env python

from collections import defaultdict,OrderedDict
import json
import pdb
import os
import numpy as np
import pandas as pd
from numba import cuda, jit, prange, vectorize, guvectorize, njit,types
from numba.typed import List,Dict
import math
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import Counter

def move_disco_embed_to_co(disco_path,co_path):
    create_folder(co_path+"bert_embeddings")
    splits = ["train","dev","test"]
    for split in splits:
        vects_path = disco_path+"/X_"+split+".npy"
        vects = np.load(vects_path,allow_pickle=True)
        np.save(co_path+"bert_embeddings/X_"+split+".npy",vects)
        del vects
        print("Saved X_"+split+".npy")

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

def read_json_data_dict(dataDict):
    data_dict = []
    with open(dataDict, 'r') as f:
        data_dict = json.load(f)
    return data_dict["dictionary"]


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


def convert_data_pldl_experiments(dframe,label_dict,grouping_category,file_path):
    parsed_results = {}
    parsed_results['data'] = label_grouping_general(dframe,label_dict,grouping_category)
    parsed_results['dictionary'] = [str(x) for x in label_dict]
    save_to_json(parsed_results,file_path)

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

def get_data_dict (l):
    enuml = enumerate(l)
    fdict = defaultdict(list)
    rdict = defaultdict(list)
    fdict = {k:v for v, k in enuml}
    rdict = {k:v for v, k in fdict.items()}
    return (fdict, rdict)

def read_json(fname):
    with open(fname) as f:
        data = json.load(f)
    return data

def assign_probability_distribution_to_value(data_to_assign,probability_distributions):
    result = []
    for data_item in data_to_assign:
        data_item = probability_distributions[0][data_item]
        result.append(data_item)
    result = np.array(result)

    return result

def convert_to_label_distributions(filename,label_dict,column_name):
    answer_counters = defaultdict(list)
    (fdict, label_dict) = get_data_dict(label_dict)
    worker_counters = get_feature_vectors_only(fdict, filename,column_name)
    return worker_counters

def convert_to_label_counts(filename,label_dict,column_name):
    (fdict, label_dict) = get_data_dict(label_dict)
    worker_counters = get_feature_vectors_only_empirical(fdict, filename,column_name)
    return worker_counters

def vectorize(fdict, labels):
    vect = defaultdict(list)
    vect = [0] * len(fdict)
    for name,number in labels.items():
        vect[fdict[name]] = number
    return vect
    

def get_feature_vectors_only(fdict, data,column_name):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        total_labels = float(sum(vect))
        vect[:] = [x /total_labels for x in vect]
        output[item[column_name]] = vect
    return output

def get_feature_vectors_only_empirical(fdict, data,column_name):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        vect[:] = [x for x in vect]
        output[item[column_name]] = vect
    return output

def compile_tweet_dict(json_list):
    result = {int(x["message_id"]): x["message"] for x in json_list}
    return result

def assign_class(data_to_assign):
    result = []
    for data_item in data_to_assign:
        data_item = data_item.astype(float)
        index = np.where(data_item==1)
        index = index[0][0]
        result.append(index)
    result = np.array(result)

    return result

def convert_list_to_numba(list_of_elements):
    list_result = List()
    for element in list_of_elements:
        list_result.append(element)
    return list_result

@jit(nopython=True)
def convert_dict_to_numba_dataitem(dict_of_elements):
    # inner_dict_type = types.DictType(types.unicode_type, types.int64)
    inner_dict_type = types.unicode_type, types.int64
    
    dict_result = Dict.empty(
        key_type=types.int64,
        value_type=inner_dict_type,
    )

    for element in dict_of_elements:
        element_values = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.int64,
        )
        for value in dict_of_elements[element]:
            element_values[value] = dict_of_elements[element][value]
        dict_result[element] = element_values
    return dict_result    

# Annotators, the data type for keys change
@jit(nopython=True)
def convert_dict_to_numba_annotator(dict_of_elements):
    # inner_dict_type = types.DictType(types.unicode_type, types.int64)
    inner_dict_type = types.unicode_type, types.int64
    
    dict_result = Dict.empty(
        key_type=types.unicode_type,
        value_type=inner_dict_type,
    )
    for element in dict_of_elements:
        element_values = Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )
        
        for value in dict_of_elements[element]:
            element_values[value] = dict_of_elements[element][value]
        dict_result[element] = element_values
    return dict_result    

@jit(nopython=True)
def convert_dict_to_numba_str_int(dict_of_elements):

    dict_result = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )
    for element in dict_of_elements:
        dict_result[element] = dict_of_elements[element]
    return dict_result

@jit(nopython=True)
def convert_dict_to_numba_int_str(dict_of_elements):

    dict_result = Dict.empty(
        key_type=types.int64,
        value_type=types.unicode_type,
    )
    for element in dict_of_elements:
        dict_result[element] = dict_of_elements[element]
    return dict_result

@jit(nopython=True)
def convert_dict_to_numba_int_int(dict_of_elements):

    dict_result = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for element in dict_of_elements:
        dict_result[element] = dict_of_elements[element]
    return dict_result

# @jit(nopython=True)
def construct_thetawm(W,Z,P,unique_dataitems,annotators,theta,Y):

    theta_wm = List()
    
    for data_item_id in unique_dataitems:

        data_item_index = unique_dataitems[data_item_id]
        annotations = Y[data_item_id] 
        
        for annotator_id in annotations:

            annotator_index = annotators[annotator_id]

            W_value = W[data_item_index]
            Z_value = Z[annotator_index]
            label = annotations[annotator_id]
            theta_wm.append(theta[W_value][Z_value][label])

    return theta_wm

# @njit
# @jit(nopython=True)
def construct_psi_omega_prods(W,Z,P,unique_dataitems,annotators,psi,omega):
    unique_dataitems = List(unique_dataitems)

    wm = np.zeros((self.M, self.N, self.P))
    
    for data_item in unique_dataitems:

        annotations = Y[data_item] 
        data_item_index = unique_dataitems.index(data_item)

        for annotation in annotations:
            
            annotator_id = annotation
            annotator_index = annotators.index(annotator_id)

            W_value = W[data_item_index]
            Z_value = Z[annotator_index]
            label = annotations[annotation]
            wm[W_value,Z_value,label] += psi[k] * omega[l]
            

    return wm

# @jit(nopython=True)
def update_thetawm_dataitem(W,Z,P,theta_wm,unique_dataitems,annotators,theta,Y,updated_data_item_index):
    data_item = unique_dataitems[updated_data_item_index]
    annotations = Y[data_item]

    for annotator_id in annotations:
        annotator_index = annotators[annotator_id] #annotators.index(annotator_id)
        index_to_update = updated_data_item_index + annotator_index
        W_value = W[updated_data_item_index]
        Z_value = Z[annotator_index]
        label = annotations[annotator_id]
        theta_wm[index_to_update] = theta[W_value][Z_value][label]

    return theta_wm

# @njit
def update_thetawm_annotatorid(W,Z,P,theta_wm,unique_dataitems,annotators,theta,Y,updated_annotator_index):
    
    annotator_id = annotators[updated_annotator_index]
    annotations = Y[annotator_id] 
    for annotation in annotations:
        data_item_index = unique_dataitems[annotation]#.index(annotation)
        index_to_update = data_item_index + updated_annotator_index
        W_value = W[data_item_index]
        Z_value = Z[updated_annotator_index]
        label = annotations[annotation]
        theta_wm[index_to_update] = theta[W_value][Z_value][label]
    return theta_wm

def assign_probability_distribution_to_value(data_to_assign,probability_distributions):
    result = []
    for data_item in data_to_assign:
        data_item = probability_distributions[0][data_item]
        result.append(data_item)
    result = np.array(result)

    return result

def predict_empirical(unique_dataitems,W,Y_dict,Z,theta,omega,P):
    # Generated through empirical approach
    item_class_distributions = {}
    item_classes = np.unique(W) #np.unique(self.W)
    predictions = {}
    for item_class in item_classes:
        item_indexes = np.where(W==item_class)
        item_indexes = item_indexes[0]
        item_class_distribution = np.zeros(P)

        for item_index in item_indexes:
            data_item = unique_dataitems[item_index]
            annotators_for_item = Y_dict[data_item]
            label_distribution_items = np.zeros(P)
            weights = 0
            for annotator in annotators_for_item:
                annotator_index = annotator.index(annotator)
                annotator_class = Z[annotator_index]
                weights += omega[0][annotator_class]
                label_distribution_item = (theta[item_class][annotator_class])*omega[0][annotator_class]
                label_distribution_items = label_distribution_items + label_distribution_item
            label_distribution_items = label_distribution_items/weights
            item_class_distribution = item_class_distribution + label_distribution_items
    return item_class_distributions

        # Empirical Prediction Method
        # W = self.W
        # Z = self.Z 
        # unique_dataitems = pd.unique(annotated_dataset['message_id'])
        # M = len(unique_dataitems)
        # annotators = pd.unique(annotated_dataset['worker_id'])
        # annotators = convert_list_to_numba(annotators)
        # N = len(annotators)
        
        # Y = annotated_dataset[["worker_id","message_id","label_vector"]]
        # Y.columns = ['N','M','P']
        # cols = ['M','N','P']
        # Y = Y[cols]

        # Y_dict = (Y.groupby('M')
        #     .apply(lambda x: dict(zip(x['N'],x['P'])))
        #     .to_dict())
        # Y_annotator_dict = (Y.groupby('N')
        #     .apply(lambda x: dict(zip(x['M'],x['P'])))
        #     .to_dict())
        # Y_dict = convert_dict_to_numba_dataitem(Y_dict)
        
        # item_classes = np.unique(self.W)
        
        # item_class_distributions = {}
        # predictions = {}

        # for item_class in item_classes:
        #     item_indexes = np.where(self.W==item_class)
        #     item_indexes = item_indexes[0]
        #     item_class_distribution = np.zeros(self.P)

        #     for item_index in item_indexes:
        #         data_item = unique_dataitems[item_index]
        #         annotators_for_item = Y_dict[data_item]
        #         label_distribution_items = np.zeros(self.P)
        #         weights = 0
        #         for annotator in annotators_for_item:
        #             annotator_index = annotator.index(annotator)
        #             annotator_class = self.Z[annotator_index]
        #             weights += self.omega[0][annotator_class]
        #             label_distribution_item = (self.theta[item_class][annotator_class])*self.omega[0][annotator_class]
        #             label_distribution_items = label_distribution_items + label_distribution_item
        #         label_distribution_items = label_distribution_items/weights
        #         item_class_distribution = item_class_distribution + label_distribution_items
            
        #     item_class_distributions[item_class] = item_class_distribution/len(item_indexes)

        # return self.W,item_class_distributions


def map_probability_to_label(choices,prediction):
    result = {}
    for x,y in zip(choices,prediction):
    	result[x] = y
    return result

def normalize(X):
    s = sum(X)
    value = [x/s for x in X]
    return value

def KLdivergence(P, Q):
    # from Q to P
    # https://datascience.stackexchange.com/a/26318/30372
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001

    P = P + epsilon
    Q = Q + epsilon
    #pdb.set_trace()
    return np.sum(P * np.log(P/Q))


def find_kl(train_vectors,train_preds):
    KLsum = []

    for train_vector,train_pred in zip(train_vectors,train_preds):
        train_pred = np.asarray(train_pred)
        train_vector = np.asarray(train_vector)
        KL = KLdivergence(train_vector, train_pred)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)
    return np.mean(KLsum)


def generate_topics_dict(topics_dist):
    topic_dict = {}
    for id,topic in zip(range(len(topics_dist)),topics_dist):
        topic_id = '"0%s"' % str(1)
        text = str(round(topic[0],3))+"*"+topic_id
        for i in range(len(topic)-1):
            topic_id = '"0%s"' % str(i+2)
            text+=" + "+str(round(topic[i+1],3))+"*"+topic_id
        topic_dict.update({id:text})
    return topic_dict

def generate_label_distributions(annotations):
    Y_dict = (annotations.groupby('message_id')
        .apply(lambda x: dict(zip(x['worker_id'],x['label_vector'])))
        .to_dict())
    return Y_dict

def generate_pd_data(result):
    total = float(sum(result))
    result = [x/total for x in result]

    return result

def save_to_json(data,outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))


def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise

            # delete old key
            del d[key]
    return d

def prepare_topics_dict_export(topics_dict):
    # topics_dict = stringify_keys(topics_dict)
    parsed_topics = {}
    for topic in topics_dict.keys():  
        parsed_topics[str(topic)] = topics_dict[topic].tolist()
    return parsed_topics


def create_folder(folderpath):
    # Check whether the specified path exists or not
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
