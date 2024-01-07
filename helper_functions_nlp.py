#!/usr/bin/env python
import gensim
import re
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import CountVectorizer
from helper_functions import generate_pd,relu
import pdb
import os
import joblib
pretrained_emb = "data/lexicons/glove.twitter.27B/glove.twitter.27B.100d.txt"
#doc2vec parameters
vector_size = 300 #Originally 300, changed for 50
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processe

def text_hybrid_labels(text_vectors,labels,weight):

    weight = float(weight)
    if weight>1:
        weight=weight/100
    print ("Hybrid Training with "+str(weight))
    hybrid_results = []
    for text_vect,label_vect in zip(text_vectors,labels):
        try:
            label_vect = np.array(labels[label_vect])
        except:
            label_vect = np.array(label_vect)
        label_vect = label_vect*weight
        text_vect = np.array(text_vect,dtype=float)
        text_vect = text_vect*(1-weight)
        #text_vect.extend(label_vect)
        hybrid_result = np.concatenate((text_vect, label_vect), axis=None)
        hybrid_results.append(hybrid_result)
    return hybrid_results

def data_prep_bnpy_glove(choice_counts, choices):
    '''
    Structure data in Bag of words format
    :param choice_counts: dictionary object {message_id : list_of_answer_counts}
    :param choices: possible answer choices
    :return:
    '''

    vocab_list = choices

    word_ids_per_doc = [x for x in range(len(vocab_list))]
    nWords = len(word_ids_per_doc)
    word_id = []
    word_count = []
    doc_range = [0]
    i = 0

    # create a list of word ids and non zero word counts for each document
    for doc_id in choice_counts:
        ans_counts = np.array(doc_id)
        ans_counts = ans_counts*ans_counts
        # find words with count > 0
        nz_word_ids = np.flatnonzero(ans_counts)
        nz_word_counts = ans_counts.ravel()[nz_word_ids]
        # print(ans_counts, nz_word_ids, nz_word_counts)
        # array([2, 1, 0, 4]), array([0, 1, 3]), array([2, 1, 4])

        word_id.extend(nz_word_ids.tolist())
        word_count.extend(nz_word_counts.tolist())

        nWords_in_doc = len(nz_word_ids)
        i += nWords_in_doc
        doc_range.append(i)

    bow_info = {
        'word_id' : np.array(word_id),
        'word_count' : np.array(word_count),
        'doc_range' : np.array(doc_range),
        'vocab_size' : np.array(nWords),
        'vocabList' : np.array(choices),
        'logFunc' : False
    }

    return bow_info
       
def build_glove_embed(cleaned_messages):
    train_vecs = list(prep_tokens_for_doc2vec(cleaned_messages))
    vec_model = gensim.models.Doc2Vec(train_vecs, vector_size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, epochs=train_epoch)
    return vec_model

def glove_embed_vects(tokens_only,vec_model):
    train_embeds = []
    train_embed_gensim = []
    for train_vect in tokens_only:
        vect = vec_model.infer_vector(train_vect).tolist()
        train_embeds.append(vec_model.infer_vector(train_vect).tolist())
        train_embed_gensim.append(embed_to_vect(vect,len(vect)))
    return train_embeds,train_embed_gensim
    
def prep_tokens_for_doc2vec(fname, tokens_only=False):
  len_of_array = len(fname)  
  for line,i in zip(fname,range(0,len_of_array)):
      if tokens_only:
          yield gensim.utils.simple_preprocess(line)
      else:
          # For training data, add tags
          yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def embed_to_vect(answer_counters, choices):
    labels = []
    for item in zip(answer_counters, range(choices)):
        labels.append(str(item[0]))
    return labels#''.join(labels).split()

def lemmatize_stemming(text):
    # return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess_stem_clean(text):
    result = []
    text = remove_url(text)
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result

def remove_url(text):
    text = re.sub(r"http\S+", "", text) #remove URLs from the text
    return text

# def clean_text_for_sklean(dataset):
#     documents = []
#     indexs = []
#     cleaned_documents = []
#     for message_id in dataset:
#         message = dataset[int(message_id)]
#         documents.append(message)
#         indexs.append(message_id)
#         answer_tokens = preprocess_stem_clean(message)
#         answer_str = ' '.join(answer_tokens)
#         cleaned_documents.append(answer_str)
#     return documents,indexs,cleaned_documents

def transform_bert_for_lda(vectors):
    # Vector points are squared since LDA does need values to be positive
    result_vectors = [vector**2 for vector in vectors]
    return result_vectors
    
def clean_text_for_sklean(dataset):
    documents = []
    indexs = []
    cleaned_documents = []
    answer_tokens_list = []
    for message_id in dataset:
        message = dataset[int(message_id)]
        documents.append(message)
        indexs.append(message_id)
        answer_tokens = preprocess_stem_clean(message)
        answer_tokens_list.append(answer_tokens)
        answer_str = ' '.join(answer_tokens)
        cleaned_documents.append(answer_str)
    return documents,indexs,cleaned_documents,answer_tokens_list
    
def build_bag_of_words(dataset):
    vectorizer = CountVectorizer(max_features=1000)
    bow_model = vectorizer.fit_transform(dataset)
    return bow_model,vectorizer
   
def data_in_cluster_sklearn(cluster_predicitions,no_clusters,data_index,answer_counters):
    labels_of_clusters = {}
    data_index = [str(x) for x in data_index]
    answer_counters = {str(x):answer_counters[x] for x in answer_counters}
    for cluster_id,data_i in zip(cluster_predicitions,data_index):
        labels = answer_counters[str(data_i)]
        try:
            labels_of_clusters[cluster_id] = [i+j for i,j in zip (labels_of_clusters[cluster_id],labels)]
        except:
            labels_of_clusters[cluster_id] = labels
    cluster_information = {}
    for cluster in range(no_clusters):
        try:
            cluster_information[str(cluster)] = generate_pd(labels_of_clusters[cluster]).tolist()
        except:
            cluster_information[str(cluster)] = np.zeros(5).tolist()

    return cluster_information

def save_trained_model_joblib_sklearn_nlp(MODEL_LOG_DIR, model, output_name, i):
    # http://scikit-learn.org/stable/modules/model_persistence.html
    # i in range(LOWER, UPPER)
    # j in range(ITERATIONS)
    model_dir = MODEL_LOG_DIR + '/CL' + str(i)

    if not os.path.exists(MODEL_LOG_DIR):
        os.makedirs(MODEL_LOG_DIR)

    joblib.dump(model, model_dir + '.pkl')
    # model.close()

def hybrid_flag(input_from_bash):
    try:
        flag = eval(input_from_bash)
    except:
        flag = float(input_from_bash)
    return flag

def transform_for_bnpy(vectors):
    result_vectors = [relu(vector) for vector in vectors] #3 for 50 window size
    return result_vectors