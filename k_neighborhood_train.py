#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	mpl.use('Agg')
import csv
import random
import math
import operator
from tqdm.auto import trange
import numpy as np
import pdb
from itertools import takewhile
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict,OrderedDict
import argparse
import sys
import pdb
from mongo_utils import retrive_model_from_sampling_db
from cf_csv_preprocess import save_to_json
from model_evaluation import KL_PMI_empirical2pred
from ldl_utils import get_data_dict, vectorize,read_json
from tqdm import tqdm
from helper_functions import get_feature_vectors_NBP,compile_tweet_dict,create_folder,KLdivergence,read_labeled_data_NBP,generate_pd,read_json_log
from helper_functions_LSTM_TF import build_labels_dict,plot_KN_history
from helper_functions_nlp import clean_text_for_sklean,build_bag_of_words,data_in_cluster_sklearn,save_trained_model_joblib_sklearn_nlp,build_glove_embed,glove_embed_vects,hybrid_flag,text_hybrid_labels

#http://houseofjeff.com/2015/05/10/using-probability-distribution-and-cosine-similarity-to-automatically-detect-data-contents/
def cosine_sim(ivec, tvec):
    # dot product
    dot = 0.0
    for i, t in zip(ivec, tvec):
        dot += (i*t)

    # vector length
    ilen = math.sqrt( sum( [i*i for i in ivec] ) )
    tlen = math.sqrt( sum( [t*t for t in tvec] ) )

    return dot/(ilen*tlen)

def euclideanDistance(instance1, instance2):
	distance = 0
	length = len(instance1)
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def flat_average(neighborhoods):
	n = len(neighborhoods)
	array_len = len(neighborhoods[0][0])
	result2 = []
	for i in range(n):
		result2.append(neighborhoods[i][0][0:array_len])
	result = np.array(result2)
	flat_average_result = result.mean(axis=0)
	return flat_average_result

def get_neighbors(trainingSet, testInstance, k,measure):
	distances = []
	length = len(testInstance)-1
	#for x in range(len(trainingSet)):
	for train_item_id in trainingSet:
		#dist = euclideanDistance(testInstance, trainingSet[x], length)
		test_item = generate_pd(np.asarray(testInstance, dtype=np.float))
		train_item = generate_pd(np.asarray(trainingSet[train_item_id], dtype=np.float))
		#dist = KLdivergence(test_item, train_item)
		#dist = distance.chebyshev(test_item, train_item)
		dist = map_to_learning_measure(test_item, train_item,measure)
		if dist<=k:
			distances.append((trainingSet[train_item_id], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(len(distances)):
		neighbors.append(distances[x])
	return neighbors

def get_neighbors_bow(training_set, training_answer_counters, testInstance, k,measure):
	distances = []
	length = len(testInstance)-1
	#for x in range(len(trainingSet)):
	for train_item,train_item_id in zip(training_set,training_answer_counters):
		#dist = euclideanDistance(testInstance, trainingSet[x], length)
		test_item = generate_pd(np.asarray(testInstance, dtype=np.float))
		train_item = generate_pd(np.asarray(train_item, dtype=np.float))
		#dist = KLdivergence(test_item, train_item)
		#dist = distance.chebyshev(test_item, train_item)
		dist = map_to_learning_measure(test_item, train_item,measure)
		if dist<=k:
			distances.append((training_answer_counters[train_item_id], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(len(distances)):
		neighbors.append(distances[x])
	return neighbors

# def generate_pd(result):
# 	total = float(np.sum(result))
# 	result = result/total
# 	return result

def generate_pd_data(result):
	total = float(sum(result[0]))
	result = result/total
	return result

def map_probability_to_label(choices,prediction):
	result = {}
	for x,y in zip(choices.values(),prediction):
		result[x] = y
	return result

def map_to_learning_measure(item1,item2,measure):
	if "KL" in measure:
		#print "Neighborhoods using KL divergence"
		return KLdivergence(item1, item2)
	elif "CH" in measure:
		#print "Neighborhoods using Chebyshev divergence"
		return distance.chebyshev(item1, item2)
	elif "EU" in measure:
		#print "Neighborhoods using Euclidean divergence"
		return euclideanDistance(item1, item2)
	elif "CA" in measure:
		#print "Neighborhoods using Euclidean divergence"
		return distance.canberra(item1, item2)
	elif "CS" in measure:
		#print "Neighborhoods using Cosine similarity"
		return cosine_sim(item1, item2)
	else:
		print ("Measure not found")

def train_NBP(train_answer_counters,dev_answer_counters,min_epsilon,max_epsilon,label_dict,dev_message_dict,measure,folder_name,output_name):
	predictions_to_write = []
	results = []
	results_KL = []
	epsilon_values = []
	results_to_write = {}

	for epsilon in np.arange(min_epsilon,max_epsilon,0.1):
		for dev_item in dev_answer_counters:
			target_instance = generate_pd(dev_answer_counters[dev_item])
			distance = get_neighbors(train_answer_counters,target_instance,epsilon,measure)

			if not distance:
				prediction = target_instance
				# prediction = np.zeros(len(target_instance)) this would make the process hard for the algorithm, only neighbors. If no neighbors, its all 0 for the LDL
			else:
				prediction = generate_pd(flat_average(distance))

			predictions_to_write.append(prediction)

		print ("Episilon Value "+str(epsilon))
		print ("Measured using "+measure)
		dev_labels = generate_pd_data(build_labels_dict(dev_answer_counters))
		KL,MIS,Nmis = KL_PMI_empirical2pred(dev_labels,predictions_to_write)
		results_KL.append(KL)
		epsilon_values.append(epsilon)
		results.append(OrderedDict([("epsilon", epsilon),("KL-divergence", KL)]))

	results_to_write["results"] = results
	results_to_write["measure"] = measure
	save_to_json(results_to_write,folder_name+"/"+output_name+"_NBP_results.json")
	plot_KN_history(epsilon_values,results_KL,measure,folder_name,output_name)

	model_epsilon = model_selection(epsilon_values,results_KL)

	return model_epsilon

def neighborhood_predict(train_answer_counters,dev_answer_counters,epsilon,label_dict,dev_message_dict,measure):
	predictions_to_write = []
	data_to_write = {}
	n_neighbors = []
	for dev_item in tqdm(dev_answer_counters,desc="NBP Training"):
		test_instance = generate_pd(dev_answer_counters[dev_item])
		distance = get_neighbors(train_answer_counters,test_instance,epsilon,measure)
		n_neighbors.append(len(distance))
		if not distance:
			prediction = test_instance
			# prediction = np.zeros(len(test_instance)) #this would make the process hard for the algorithm, only neighbors. If no neighbors, its all 0 for the LDL
		else:
			prediction = generate_pd(flat_average(distance))
		#pdb.set_trace()
		labels = map_probability_to_label(label_dict,prediction)
		predictions_to_write.append(OrderedDict([("message_id", dev_item),("message", dev_message_dict[int(dev_item)]),("labels", labels)]))
	data_to_write["data"] = predictions_to_write
	data_to_write["dictionary"] = label_dict.values()

	# data_to_write["topics_dict"] = np.mean(n_neighbors) #To maintain consistency with the clustering methods. This is used for sampling
	data_to_write["topics_dict"] = np.median(n_neighbors) #To maintain consistency with the clustering methods. This is used for sampling
	# print ("NSize"+str(epsilon)+"median"+str(np.median(n_neighbors)))
	return data_to_write

def neighborhood_predict_nlp(train_answer_counters,dev_answer_counters,epsilon,label_dict,dev_message_dict,train_vectors,dev_vectors,measure):
	predictions_to_write = []
	data_to_write = {}
	n_neighbors = []
	for dev_message_id,dev_item in tqdm(zip(dev_answer_counters,dev_vectors),total=len(dev_vectors),desc="NBP Training"):
		# if glove:
		# 	test_instance = dev_item
		# else:
		test_instance = generate_pd(dev_item)
		# distance = get_neighbors(train_vectors,train_answer_counters,test_instance,epsilon,measure)
		distance = get_neighbors_bow(train_vectors,train_answer_counters,test_instance,epsilon,measure)
		n_neighbors.append(len(distance))
		if not distance:
			prediction = test_instance
			# prediction = np.zeros(len(test_instance)) #this would make the process hard for the algorithm, only neighbors. If no neighbors, its all 0 for the LDL
		else:
			prediction = generate_pd(flat_average(distance))
		labels = map_probability_to_label(label_dict,prediction)
		predictions_to_write.append(OrderedDict([("message_id", dev_message_id),("message", dev_message_dict[int(dev_message_id)]),("labels", labels)]))
	data_to_write["data"] = predictions_to_write
	data_to_write["dictionary"] = label_dict.values()
	# data_to_write["topics_dict"] = np.mean(n_neighbors) #To maintain consistency with the clustering methods. This is used for sampling
	data_to_write["topics_dict"] = np.median(n_neighbors) #To maintain consistency with the clustering methods. This is used for sampling
	# print ("NSize"+str(epsilon)+"median"+str(np.median(n_neighbors)))
	return data_to_write

def model_selection(epsilon_values,results):
    min_KL_value = min(results)
    min_epsilon_value = epsilon_values[results.index(min_KL_value)]
    print ("Selected epsilon: "+str(min_epsilon_value)+" KL-divergence: "+str(min_KL_value))
    return min_epsilon_value

def preprocess_data(input_train_file_name,input_dev_file_name,input_test_file_name,folder_name):
    create_folder(folder_name)

    train_answer_counters,train_message_dict,label_dict = read_labeled_data_NBP(input_train_file_name)
    dev_answer_counters,dev_message_dict,label_dict = read_labeled_data_NBP(input_dev_file_name)
    test_answer_counters,test_message_dict,label_dict = read_labeled_data_NBP(input_test_file_name)

    return train_answer_counters,dev_answer_counters,test_answer_counters,label_dict,train_message_dict,dev_message_dict,test_message_dict

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_file", help="Input training file JSON name")
	parser.add_argument("--dev_file", help="Input dev file JSON name")
	parser.add_argument("--test_file", help="Input test file JSON name")
	parser.add_argument("--train_file_vects", help="Input training file vects name",default=False)
	parser.add_argument("--dev_file_vects", help="Input dev file vects name",default=False)
	parser.add_argument("--test_file_vects", help="Input test file vects name",default=False)
	parser.add_argument("--min_epsilon", help="Min Epsilon value",default = 0.0)
	parser.add_argument("--max_epsilon", help="Max Epsilon value",default=False)
	parser.add_argument("--output_file", help="Output file name")
	parser.add_argument("--folder_name", help="Main folder name")
	parser.add_argument("--measure", help="Information theoritic measure")
	parser.add_argument("--NBP_debug", help="Debug NBP", default = False)
	parser.add_argument("--nlp_data", help="NLP Data",default = False)
	parser.add_argument("--results_db", help="Results DB", default = False)
	parser.add_argument("--results_col", help="Col name",default = False)
	parser.add_argument("--glove", help="Doc2Vec with Glove",default = False)
	parser.add_argument("--hybrid", help="Hybrid model of text + labels",default=False)
	parser.add_argument("--json_log_file", help="JSON Log file as fall back for DB", default=False)
	args = parser.parse_args()
	measure = args.measure
	nlp_flag = args.nlp_data
	glove = args.glove
	hybrid = hybrid_flag(args.hybrid)

	train_vects = args.train_file_vects
	dev_vects = args.dev_file_vects
	test_vects = args.test_file_vects
	if train_vects:
		train_vectors = np.load(train_vects,allow_pickle=True)
	if dev_vects:
		dev_vectors = np.load(dev_vects,allow_pickle=True)
	if test_vects:
		test_vectors = np.load(test_vects,allow_pickle=True)
	if "all" in args.measure:
		measure = ["KL","CH","EU","CA","CS"]

	train_answer_counters,dev_answer_counters,test_answer_counters,label_dict,train_message_dict,dev_message_dict,test_message_dict = preprocess_data(args.train_file,args.dev_file,args.test_file,args.folder_name)
	#Training
	if args.NBP_debug:
		model_epsilon_value = train_NBP(train_answer_counters,dev_answer_counters,float(args.min_epsilon),float(args.max_epsilon),label_dict,dev_message_dict,measure,args.folder_name,args.output_file)
	else:
		if (args.max_epsilon):
			model_epsilon_value = float(args.max_epsilon)
		else:
			#results_db = args.results_db
			#col = args.results_col
			#model_epsilon_value = float(retrive_model_from_sampling_db(results_db,col))
			json_log_file = args.json_log_file
			model_epsilon_value = read_json_log(json_log_file)
			print ("NBP Trainning on "+str(model_epsilon_value))
	
	#Predictions
	if (nlp_flag):
		train_messages,train_message_ids,train_cleaned_messages,train_tokens = clean_text_for_sklean(train_message_dict)
		dev_messages,dev_message_ids,dev_cleaned_messages,dev_tokens = clean_text_for_sklean(dev_message_dict)
		test_messages,test_message_ids,test_cleaned_messages,test_tokens = clean_text_for_sklean(test_message_dict)
		if glove:
			glove_model = build_glove_embed(train_cleaned_messages)
			train_vectors,_ = glove_embed_vects(train_tokens,glove_model)
			dev_vectors,_ = glove_embed_vects(dev_tokens,glove_model)
			test_vectors,_ = glove_embed_vects(test_tokens,glove_model)

		# else:
		# 	train_vectors,sklearn_bow_model = build_bag_of_words(train_cleaned_messages)
		# 	train_vectors = train_vectors.toarray()
		# 	dev_vectors = sklearn_bow_model.transform(dev_cleaned_messages).toarray()
		# 	test_vectors = sklearn_bow_model.transform(test_cleaned_messages).toarray()

		if hybrid:
			train_vectors = text_hybrid_labels(train_vectors,train_answer_counters,float(hybrid))
			dev_vectors = text_hybrid_labels(dev_vectors,dev_answer_counters,float(hybrid))
			test_vectors = text_hybrid_labels(test_vectors,test_answer_counters,float(hybrid))
		
		train_predictions = neighborhood_predict_nlp(train_answer_counters,train_answer_counters,model_epsilon_value,label_dict,train_message_dict,train_vectors,train_vectors,measure)
		save_to_json(train_predictions,args.folder_name+"/"+args.output_file+"_predict_train.json")

		dev_predictions = neighborhood_predict_nlp(train_answer_counters,dev_answer_counters,model_epsilon_value,label_dict,dev_message_dict,train_vectors,dev_vectors,measure)
		save_to_json(dev_predictions,args.folder_name+"/"+args.output_file+"_predict_dev.json")
	
		test_predictions = neighborhood_predict_nlp(train_answer_counters,test_answer_counters,model_epsilon_value,label_dict,test_message_dict,train_vectors,test_vectors,measure)
		save_to_json(test_predictions,args.folder_name+"/"+args.output_file+"_predict_test.json")

	else:
		train_predictions = neighborhood_predict(train_answer_counters,train_answer_counters,model_epsilon_value,label_dict,train_message_dict,measure)
		save_to_json(train_predictions,args.folder_name+"/"+args.output_file+"_predict_train.json")

		dev_predictions = neighborhood_predict(train_answer_counters,dev_answer_counters,model_epsilon_value,label_dict,dev_message_dict,measure)
		save_to_json(dev_predictions,args.folder_name+"/"+args.output_file+"_predict_dev.json")

		test_predictions = neighborhood_predict(train_answer_counters,test_answer_counters,model_epsilon_value,label_dict,test_message_dict,measure)
		save_to_json(test_predictions,args.folder_name+"/"+args.output_file+"_predict_test.json")



if __name__ == '__main__':
	main()
