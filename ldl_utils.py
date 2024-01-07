#Utilties script of the LDL pipeline
import json
import pdb
from collections import defaultdict

def get_data_dict (l):
    enuml = enumerate(l)
    fdict = defaultdict(list)
    rdict = defaultdict(list)
    fdict = {k:v for v, k in enuml}
    rdict = {k:v for v, k in fdict.items()}
    return (fdict, rdict)

def get_feature_vectors(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:

        vect = vectorize(fdict, item["labels"])
        item["message_id"] = int(item["message_id"])
        output[item["message_id"]] = vect
    return output

def compile_tweet_dict(json_list):
    result = {int(x["message_id"]): x["message"] for x in json_list}
    return result

def vectorize(fdict, labels):
    vect = defaultdict(list)
    vect = [0] * len(fdict)
    for name,number in labels.items():
        vect[fdict[name]] = number
    return vect

def compile_tweet_ft_dict(json_lst):
    result = defaultdict(list)
    result = {int(k):v for k,v in json_lst.items()}
    return result

def convert_tweetids(fname):
    json_lst = fname
    result = {int(k) for element in json_lst for k in element.items()}
    return result

def read_json(fname):
    datastore = defaultdict(list)
    if fname:
        with open(fname, 'r') as f:
            datastore = json.load(f)
    return datastore

def save_label_dict(dict,id,foldername):
    with open(foldername+"/"+id+"_dict.json", 'w') as outfile:
        outfile.write(json.dumps(dict, indent=4))

def load_label_dict(id,foldername):
    label_dict = defaultdict(list)
    with open(foldername+"/"+id+"_dict.json", 'r') as f:
        label_dict = json.load(f)
    return label_dict

def save_label_vects(vects,id,foldername):
    with open(foldername+"/"+id+"_vects.json", 'w') as outfile:
        outfile.write(json.dumps(vects, indent=4))

def load_label_vects(id,foldername):
    label_dict = defaultdict(list)
    with open(foldername+"/"+id+"_vects.json", 'r') as f:
        label_dict = json.load(f)
    label_dict = compile_tweet_ft_dict(label_dict)
    pdb.set_trace()
    return label_dict

#OLD
# def get_feature_vectors(fdict, data):
#     #output = {}
#     output = defaultdict(list)
#     for item in data:
#         vect = vectorize(fdict, item["labels"])
#         item["tweet_id"] = int(item["tweet_id"])
#         output[item["tweet_id"]] = vect
#     return output
#
# def compile_tweet_dict(fname):
#     json_lst = read_json(fname)
#     result = {int(k) : v  for element in json_lst for k, v in element.items()}
#     return result
