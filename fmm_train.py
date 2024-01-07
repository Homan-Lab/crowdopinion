#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import argparse
import bnpy
import sys
import pdb
import numpy as np
from numpy import argmax, dot
from collections import defaultdict,OrderedDict
from cf_csv_preprocess import save_to_json
from ldl_utils import get_data_dict, get_feature_vectors, vectorize,read_json,compile_tweet_dict,save_label_dict,load_label_dict,save_label_vects,load_label_vects
from helper_functions import data_prep_bnpy,save_bnpy_model,load_bnpy_model,build_prob_distribution,map_probability_to_label,generate_topics_dict,create_folder
from FMM_utils import get_assignments,language_prep_bnpy
import pickle
import argparse
import pandas as pd
from helper_functions_nlp import clean_text_for_sklean,build_bag_of_words,data_in_cluster_sklearn,save_trained_model_joblib_sklearn_nlp,prep_tokens_for_doc2vec,embed_to_vect,build_glove_embed,glove_embed_vects,text_hybrid_labels,hybrid_flag,data_prep_bnpy_glove,transform_bert_for_lda
from helper_functions import bnpy_find_kl,iteration_selection_bnpy,find_item_distribution_clusters_sklearn,get_ids_only,write_model_logs_to_json,relu
from tqdm import tqdm
import shutil

ITERATIONS = 10
q1_LOWER = 2
q1_UPPER = 12
q2_LOWER = 2
q2_UPPER = 12
q3_LOWER = 5
q3_UPPER = 20
DS_iter = 1
default_LOWER = 2
default_UPPER = 12
FMM_DPMM_Gamma = 0.5
TARGET = 'label'
iterations = 10


#Pre_Training
def FMM_preprocess(input_file_name, output_file_name):
    vects = defaultdict(list)
    tweet_dict = defaultdict(list)
    JSONfile = read_json(input_file_name)
    create_folder(output_file_name) #creates the folder for saving LDA models
    tweet_dict = compile_tweet_dict(JSONfile["data"])
    JSONfile["dictionary"] = [str(x) for x in JSONfile["dictionary"]]
    (fdict, rdict) = get_data_dict(JSONfile["dictionary"])
    vects = get_feature_vectors(fdict, JSONfile["data"])

    print("Running FMM in Train mode on {} Data Items on {}.".format(len(vects),id))
    return vects, rdict,tweet_dict
    
def transform_for_bnpy(vectors):
    result_vectors = [relu(vector) for vector in vectors] #3 for 50 window size
    return result_vectors

def bnpy_train(tweetid_answer_counters, choices, lower,upper, message_dict, path_to_save, process_id, target,dev_vects,dev_tweet_dict,test_vects,test_tweet_dict,hybrid,train_vectors,dev_vectors,test_vectors):
    '''
    Train bnpy multinomial mixture model
    :param split_prep: type of data split to use for this experiment(shuffle/dense)
    :param tweetid_answer_counters: dictionary of the form {tweet_id: [ct_ans1, ct_ans2, ct_ans3 ...]}
    :param choices: possible answers choices
    :param ITERATIONS: number of iterations from which the best model will be chosen
    :param LOWER: start value for number of clusters with which the model will be trained
    :param UPPER: end value for number of clusters with which the model will be trained
    :param output_name: the name of output directory
    :param target: This parameter is used to select embedding method based on input. It supports 1) bow or glove, which triggers data to be processed and embedded. 
    2) embeddings, which expects a .npy file of existing vectors in the format of IxN (I = vectors, N = number of items) or 3) label, which experiments on labels only.
    :return: None
    '''
    algName = 'VB'
    train_message_ids = get_ids_only(tweetid_answer_counters)

    if target == 'vectors':
        # train_vectors = transform_bert_for_lda(train_vectors)
        # dev_vectors = transform_bert_for_lda(dev_vectors)
        # test_vectors = transform_bert_for_lda(test_vectors)
        trn_pd = pd.DataFrame(train_vectors)
        trn_pd.columns = trn_pd.columns.astype(str)
        trn_dataset = bnpy.data.XData.from_dataframe(trn_pd)
        trn_vectors_values = trn_dataset.column_names
        bow_info_train = data_prep_bnpy_glove(train_vectors, trn_vectors_values)
        bow_info_dev = data_prep_bnpy_glove(dev_vectors, trn_vectors_values)
        bow_info_test = data_prep_bnpy_glove(test_vectors, trn_vectors_values)
        # algName = 'EM'
    # if "facebook" in path_to_save:
    #     algName = 'VB'
          
    if target == 'bow' or target =='glove':
        train_messages,train_message_ids,train_cleaned_messages,train_tokens = clean_text_for_sklean(message_dict)
        dev_messages,dev_message_ids,dev_cleaned_messages,dev_tokens = clean_text_for_sklean(dev_tweet_dict)
        test_messages,test_message_ids,test_cleaned_messages,test_tokens = clean_text_for_sklean(test_tweet_dict)
        # train_vectors,sklearn_bow_model = build_bag_of_words(train_cleaned_messages)
        # dev_vectors = sklearn_bow_model.transform(dev_cleaned_messages)
        # pdb.set_trace()
        bow_info_train = language_prep_bnpy(train_cleaned_messages,train_cleaned_messages)
        bow_info_dev = language_prep_bnpy(train_cleaned_messages,dev_cleaned_messages)
        bow_info_test = language_prep_bnpy(train_cleaned_messages,test_cleaned_messages)

    if target =='glove':
        vec_model = build_glove_embed(train_cleaned_messages)
        train_vectors,_ = glove_embed_vects(train_tokens,vec_model)
        train_vectors = [transform_for_bnpy(train_vector) for train_vector in train_vectors]
        trn_vectors = pd.DataFrame(train_vectors)
        trn_vectors.columns = trn_vectors.columns.astype(str)
        trn_dataset = bnpy.data.XData.from_dataframe(trn_vectors)
        trn_vectors_values = trn_dataset.column_names

        dev_cleaned_messages = list(prep_tokens_for_doc2vec(dev_cleaned_messages,tokens_only=True))
        dev_vectors,_ = glove_embed_vects(dev_cleaned_messages,vec_model)
        dev_vectors = [transform_for_bnpy(dev_vector) for dev_vector in dev_vectors]

        test_cleaned_messages = list(prep_tokens_for_doc2vec(test_cleaned_messages,tokens_only=True))
        test_vectors,_ = glove_embed_vects(test_cleaned_messages,vec_model)
        test_vectors = [transform_for_bnpy(test_vector) for test_vector in test_vectors]

        bow_info_train = data_prep_bnpy_glove(train_vectors, trn_vectors_values)
        bow_info_dev = data_prep_bnpy_glove(dev_vectors, trn_vectors_values)
        bow_info_test = data_prep_bnpy_glove(test_vectors, trn_vectors_values)

    if target == 'label' or hybrid==100:
        ### convert data to bag of words format ###
        bow_info_train = data_prep_bnpy(tweetid_answer_counters, choices.values())
        bow_info_dev = data_prep_bnpy(dev_vects, choices.values())
        bow_info_test = data_prep_bnpy(test_vects, choices.values())
    
    if hybrid and hybrid<100:
        
        train_vectors = text_hybrid_labels(train_vectors,tweetid_answer_counters,float(hybrid))
        column_names = [str(x) for x in range(len(train_vectors[0]))]

        dev_vectors = text_hybrid_labels(dev_vectors,dev_vects,float(hybrid))
        test_vectors = text_hybrid_labels(test_vectors,test_vects,float(hybrid))

        bow_info_train = data_prep_bnpy_glove(train_vectors, column_names)
        bow_info_dev = data_prep_bnpy_glove(dev_vectors, column_names)
        bow_info_test = data_prep_bnpy_glove(test_vectors, column_names)
    else:
        
        bow_info_train = data_prep_bnpy(tweetid_answer_counters, choices.values())
        bow_info_dev = data_prep_bnpy(dev_vects, choices.values())
        bow_info_test = data_prep_bnpy(test_vects, choices.values())

    ### create a bnpy DataObj ###
    trn_dataset = bnpy.data.BagOfWordsData(**bow_info_train)
    dev_dataset = bnpy.data.BagOfWordsData(**bow_info_dev)
    test_dataset = bnpy.data.BagOfWordsData(**bow_info_test)
    
    results_log_dict = {}
    ### train and save the Mixture Model ###
    for n_clusters in tqdm(range(lower,upper),desc="Training FMM Model"):
        trained_model = None
        info_dict = None
        kl = []
        results = {}
        # get the best model out of nTask runs
        # https://bnpy.readthedocs.io/en/latest/examples/01_asterisk_K8/plot-01-demo=init_methods-model=mix+gauss.html?highlight=initname#initname-bregmankmeans
        for i in range(iterations):
            
            trained_model, info_dict = bnpy.run(trn_dataset, 'FiniteMixtureModel', 'Mult', algName,
                                nLap=1000, convergeThr=0.0001, nTask=ITERATIONS,
                                K=n_clusters, initname='randexamples',
                                gamma0=FMM_DPMM_Gamma, lam=0.1, doWriteStdOut=False, logFunc=None, doSaveToDisk=False)
            # changing initname to randexamples fixes div error default is bregmankmeans
            info_dict['Centroids'] = np.multiply(info_dict['SS'].WordCounts.transpose(), np.reciprocal(info_dict['SS'].SumWordCounts)).transpose()
            info_dict['curr_loss'] = -1 * trained_model.calc_evidence(trn_dataset)
            # pdb.set_trace()
            LP = trained_model.calc_local_params(trn_dataset)
            preds = LP['resp']
            predictions,cluster_assignments = get_assignments(tweetid_answer_counters,preds)
            train_answer_counters = tweetid_answer_counters
            train_predict = predictions
            #save_bnpy_model(model_dir, trained_model, info_dict)
            kl.append(bnpy_find_kl(train_answer_counters,train_predict))
            results[i] = find_item_distribution_clusters_sklearn(cluster_assignments)
            
            #Generating data to write
            bnpy_write_predicitions(tweetid_answer_counters,predictions,cluster_assignments,choices,info_dict,message_dict,path_to_save+"/CL"+str(n_clusters)+"/temp"+str(i)+"/"+process_id+"_train.json")
            LP = trained_model.calc_local_params(dev_dataset)
            preds = LP['resp']
            predictions,cluster_assignments = get_assignments(dev_vects,preds)
            bnpy_write_predicitions(dev_vects,predictions,cluster_assignments,choices,info_dict,dev_tweet_dict,path_to_save+"/CL"+str(n_clusters)+"/temp"+str(i)+"/"+process_id+"_dev.json")

            LP = trained_model.calc_local_params(test_dataset)
            preds = LP['resp']
            predictions,cluster_assignments = get_assignments(test_vects,preds)
            bnpy_write_predicitions(test_vects,predictions,cluster_assignments,choices,info_dict,test_tweet_dict,path_to_save+"/CL"+str(n_clusters)+"/temp"+str(i)+"/"+process_id+"_test.json")
        results_log_dict[n_clusters],train_pred,dev_pred,test_pred = iteration_selection_bnpy(kl,results,path_to_save + "/CL"+str(n_clusters)+"/temp",n_clusters,process_id)
        shutil.rmtree(path_to_save + "/CL"+str(n_clusters))

        write_model_logs_to_json(path_to_save+"/CL"+str(n_clusters),train_pred,process_id+"_train")
        write_model_logs_to_json(path_to_save+"/CL"+str(n_clusters),dev_pred,process_id+"_dev")
        write_model_logs_to_json(path_to_save+"/CL"+str(n_clusters),test_pred,process_id+"_test")
    results_log_dict["exp_name"] = process_id
    write_model_logs_to_json(path_to_save,results_log_dict,"cluster_log")
    print("Completed FMM Training")

def bnpy_write_predicitions(tweetid_answer_counters,predictions,cluster_assignments,choices,info_dict,message_dict,path_to_save):
    predictions_to_write = []
    data_to_write = {}
    for data_item,prediction,cluster_assignment in zip(tweetid_answer_counters,predictions,cluster_assignments):
        labels = map_probability_to_label(choices,prediction)
        predictions_to_write.append(OrderedDict([("message_id", data_item),("message", message_dict[int(data_item)]),("cluster",cluster_assignment+1),("labels", labels)]))
    #print ("Training completed and saved to "+model_dir)
    data_to_write["data"] = predictions_to_write
    data_to_write["dictionary"] = choices.values()
    data_to_write['topics_dict'] = generate_topics_dict(info_dict['Centroids'])
    save_to_json(data_to_write,path_to_save)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Input training file JSON name")
    parser.add_argument("--dev_file", help="Input dev file JSON name")
    parser.add_argument("--test_file", help="Input test file JSON name")
    parser.add_argument("--input_train_file_vectors", help="Input training file vectors .npy name",default=False)
    parser.add_argument("--dev_file_vectors", help="Input dev file vectors .npy name",default=False)
    parser.add_argument("--test_file_vectors", help="Input test file vectors .npy name",default=False)
    parser.add_argument("--nlp_flag", help="NLP Data Flag",default=False)
    parser.add_argument("--lower", help="Lower Limit")
    parser.add_argument("--upper", help="Upper Limit")
    parser.add_argument("--output_file", help="Output file name")
    parser.add_argument("--folder_name", help="Main folder name",default=False)
    parser.add_argument("--hybrid", help="Hybrid model of text + labels",default=False)
    args = parser.parse_args()
    nlp_flag = args.nlp_flag
    
    train_embed = args.input_train_file_vectors
    dev_embed = args.dev_file_vectors
    test_embed = args.test_file_vectors

    hybrid = hybrid_flag(args.hybrid)
    if nlp_flag!="bow" and nlp_flag!="glove" and nlp_flag!="vectors":
        nlp_flag = 'label'

    if nlp_flag=="vectors":
        train_embed = np.load(train_embed,allow_pickle=True)
        dev_embed = np.load(dev_embed,allow_pickle=True)
        test_embed = np.load(test_embed,allow_pickle=True)
    train_vects, rdict,tweet_dict = FMM_preprocess(args.input_file, args.folder_name)
    dev_vects, rdict,dev_tweet_dict = FMM_preprocess(args.dev_file, args.folder_name)
    test_vects, rdict,test_tweet_dict = FMM_preprocess(args.test_file, args.folder_name)
    bnpy_train(train_vects, rdict, int(args.lower),int(args.upper),tweet_dict,args.folder_name,args.output_file,nlp_flag,dev_vects,dev_tweet_dict,test_vects,test_tweet_dict,hybrid,train_embed,dev_embed,test_embed)

if __name__ == '__main__':
    main()


# #https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
# #for running the pipeline through SSH
# import os
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# import argparse
# import sys
# import pdb
# from FMM_utils import bnpy_train_model
# from ldl_utils import get_data_dict, get_feature_vectors, vectorize,read_json,compile_tweet_dict,save_label_dict,load_label_dict,save_label_vects,load_label_vects
# from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
# from helper_functions import create_folder
# import pickle
# import argparse
# #python FMM_train.py --input data/jobQ123_BOTH/processed/jobQ1_BOTH/split/jobQ1_BOTH_train.json --clusters 5 --output data/jobQ1_BOTH/jobQ1_BOTH_split_fmm.pkl
# #Constants for LDA Data


# #Pre_Training
# def FMM_preprocess(upper,input_file_name, output_file_name):
#     vects = defaultdict(list)
#     tweet_dict = defaultdict(list)
#     JSONfile = read_json(input_file_name)
#     #create_folder(output_file_name) #creates the folder for saving LDA models
#     tweet_dict = compile_tweet_dict(JSONfile["data"])
#     (fdict, rdict) = get_data_dict(JSONfile["dictionary"])
#     vects = get_feature_vectors(fdict, JSONfile["data"])
#     print("Running FMM in Train mode on {} Tweets on {}.".format(len(vects),id))
#     lower=1
#     bnpy_train_model(vects, rdict,lower,upper,tweet_dict,output_file_name,output_file_name,"fmm","label")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_file", help="Input training file JSON name")
#     parser.add_argument("--clusters", help="Lower Limit")
#     #parser.add_argument("--upper", help="Upper Limit")
#     parser.add_argument("--output_file", help="Output file name")
#     #parser.add_argument("--folder_name", help="Main folder name")
#     args = parser.parse_args()

#     FMM_preprocess(int(args.clusters),args.input_file, args.output_file)

# if __name__ == '__main__':
#     main()
