import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import bnpy
import pdb
import numpy as np
from numpy import argmax, dot
from collections import defaultdict,OrderedDict
from helper_functions import data_prep_bnpy,save_bnpy_model,load_bnpy_model,build_prob_distribution,map_probability_to_label,generate_topics_dict,save_to_json_foldercheck
from helper_functions_nlp import clean_text_for_sklean,build_bag_of_words,data_in_cluster_sklearn,save_trained_model_joblib_sklearn_nlp,prep_tokens_for_doc2vec,embed_to_vect,build_glove_embed,glove_embed_vects,text_hybrid_labels,hybrid_flag


FMM_DPMM_Gamma = 0.5
ITERATIONS = 5

def bnpy_predict(tweetid_answer_counters, choices, n_clusters, message_dict, path_to_save, model_name, target):
    '''
    Train bnpy multinomial mixture model
    :param split_prep: type of data split to use for this experiment(shuffle/dense)
    :param tweetid_answer_counters: dictionary of the form {tweet_id: [ct_ans1, ct_ans2, ct_ans3 ...]}
    :param choices: possible answers choices
    :param ITERATIONS: number of iterations from which the best model will be chosen
    :param LOWER: start value for number of clusters with which the model will be trained
    :param UPPER: end value for number of clusters with which the model will be trained
    :param output_name: the name of output directory
    :return: None
    '''

    # Read data splits from file, NOT generate each time
    # with open(SPLIT_LOG_DIR + output_name + "_" + split_prep + ".json") as fp:
    #     results_dict = json.load(fp)
    # train_items = results_dict['train_set']
    # dev_items = results_dict['dev_set']

    # train_answer_counters = {}
    # for k in train_items:
    #     train_answer_counters[k] = tweetid_answer_counters[k]
    # dev_answer_counters = {}
    # for k in dev_items:
    #     dev_answer_counters[k] = tweetid_answer_counters[k]


    if target == 'label':
        ### convert data to bag of words format ###
        bow_info_train = data_prep_bnpy(tweetid_answer_counters, choices.values())
    #     bow_info_dev = data_prep_bnpy(dev_answer_counters, choices)
    elif target == 'text':
        bow_info_train = language_prep_bnpy(message_dict)
    #     bow_info_dev = language_prep_bnpy(message_dict, dev_items)

    ### create a bnpy DataObj ###

    trn_dataset = bnpy.data.BagOfWordsData(**bow_info_train)
    #dev_dataset = bnpy.data.BagOfWordsData(**bow_info_dev)

    ### train and save the Mixture Model ###
    if model_name == "fmm":
        trained_model = None
        info_dict = None
        # get the best model out of nTask runs
        # https://bnpy.readthedocs.io/en/latest/examples/01_asterisk_K8/plot-01-demo=init_methods-model=mix+gauss.html?highlight=initname#initname-bregmankmeans
        
        trained_model, info_dict = bnpy.run(trn_dataset, 'FiniteMixtureModel', 'Mult', 'VB',
                            nLap=1000, convergeThr=0.0001, nTask=ITERATIONS,
                            K=n_clusters, initname='bregmankmeans',
                            gamma0=FMM_DPMM_Gamma, lam=0.1, doWriteStdOut=False, logFunc=None, doSaveToDisk=False)
        info_dict['Centroids'] = np.multiply(info_dict['SS'].WordCounts.transpose(), np.reciprocal(info_dict['SS'].SumWordCounts)).transpose()
        info_dict['curr_loss'] = -1 * trained_model.calc_evidence(trn_dataset)


        ### Store trained model ###
        #model_dir = output_folder + "/" + str(i) + "/"
        #model_dir = folder + '/gamma0=' + str(FMM_DPMM_Gamma) + '/' + target + '/CL' + str(i) + '/'
        #model_dir = self.MM_LOG_DIR + 'gamma0=' + str(FMM_DPMM_Gamma) + '/' + output_name + "_" + split_prep + "_" + target + '/CL' + str(i) + '/'
        ### save the best model ###

        LP = trained_model.calc_local_params(trn_dataset)
        preds = LP['resp']
        predictions,cluster_assignments = get_assignments(tweetid_answer_counters,preds)
        #save_bnpy_model(model_dir, trained_model, info_dict)


    #Generating data to write
    predictions_to_write = []
    data_to_write = {}

    for data_item,prediction,cluster_assignment in zip(tweetid_answer_counters,predictions,cluster_assignments):
        labels = map_probability_to_label(choices,prediction)
        predictions_to_write.append(OrderedDict([("message_id", data_item),("message", message_dict[int(data_item)]),("cluster",cluster_assignment+1),("labels", labels)]))
    #print ("Training completed and saved to "+model_dir)
    data_to_write["data"] = predictions_to_write
    data_to_write["dictionary"] = choices.values()
    data_to_write['topics_dict'] = generate_topics_dict(info_dict['Centroids'])
    save_to_json_foldercheck(data_to_write,path_to_save)

    #models_dir = folder + '/gamma0=' + str(FMM_DPMM_Gamma) + '/' + target
    #bnpy_model_selection(trn_dataset,models_dir,tweetid_answer_counters,choices.values(),message_dict,output_name,target)
    #print('\n===== Trained model * ' + str(output_name) + ' * stored in directory ' + str(self.MM_LOG_DIR) + '=====\n')



def bnpy_train_model_old(tweetid_answer_counters, choices, LOWER, UPPER, message_dict, folder,output_name, model_name, target):
    '''
    Train bnpy multinomial mixture model
    :param split_prep: type of data split to use for this experiment(shuffle/dense)
    :param tweetid_answer_counters: dictionary of the form {tweet_id: [ct_ans1, ct_ans2, ct_ans3 ...]}
    :param choices: possible answers choices
    :param ITERATIONS: number of iterations from which the best model will be chosen
    :param LOWER: start value for number of clusters with which the model will be trained
    :param UPPER: end value for number of clusters with which the model will be trained
    :param output_name: the name of output directory
    :return: None
    '''

    # Read data splits from file, NOT generate each time
    # with open(SPLIT_LOG_DIR + output_name + "_" + split_prep + ".json") as fp:
    #     results_dict = json.load(fp)
    # train_items = results_dict['train_set']
    # dev_items = results_dict['dev_set']

    # train_answer_counters = {}
    # for k in train_items:
    #     train_answer_counters[k] = tweetid_answer_counters[k]
    # dev_answer_counters = {}
    # for k in dev_items:
    #     dev_answer_counters[k] = tweetid_answer_counters[k]


    if target == 'label':
        ### convert data to bag of words format ###
        bow_info_train = data_prep_bnpy(tweetid_answer_counters, choices.values())
    #     bow_info_dev = data_prep_bnpy(dev_answer_counters, choices)
    # elif target == 'text':
    #     bow_info_train = language_prep_bnpy(message_dict, train_items)
    #     bow_info_dev = language_prep_bnpy(message_dict, dev_items)

    ### create a bnpy DataObj ###

    trn_dataset = bnpy.data.BagOfWordsData(**bow_info_train)
    #dev_dataset = bnpy.data.BagOfWordsData(**bow_info_dev)

    ### train and save the Mixture Model ###
    if model_name == "fmm":
        for i in range(LOWER, UPPER):
            trained_model = None
            info_dict = None
            import pdb; pdb.set_trace()
            # get the best model out of nTask runs
            # https://bnpy.readthedocs.io/en/latest/examples/01_asterisk_K8/plot-01-demo=init_methods-model=mix+gauss.html?highlight=initname#initname-bregmankmeans
            trained_model, info_dict = bnpy.run(trn_dataset, 'FiniteMixtureModel', 'Mult', 'VB',
                                nLap=1000, convergeThr=0.0001, nTask=ITERATIONS,
                                K=i, initname='bregmankmeans',
                                gamma0=FMM_DPMM_Gamma, lam=0.1, doWriteStdOut=False, logFunc=None, doSaveToDisk=False)
            #trained_model, info_dict = bnpy.run(trn_dataset, 'FiniteMixtureModel', 'Mult', 'VB',nLap=1000, convergeThr=0.0001, nTask=ITERATIONS,K=i, initname='bregmankmeans',gamma0=FMM_DPMM_Gamma, lam=0.1, doWriteStdOut=False, logFunc=None, doSaveToDisk=False)
            info_dict['Centroids'] = np.multiply(info_dict['SS'].WordCounts.transpose(), np.reciprocal(info_dict['SS'].SumWordCounts)).transpose()
            info_dict['curr_loss'] = -1 * trained_model.calc_evidence(trn_dataset)


            ### Store trained model ###
            #model_dir = output_folder + "/" + str(i) + "/"
            model_dir = folder + '/gamma0=' + str(FMM_DPMM_Gamma) + '/' + target + '/CL' + str(i) + '/'
            #model_dir = self.MM_LOG_DIR + 'gamma0=' + str(FMM_DPMM_Gamma) + '/' + output_name + "_" + split_prep + "_" + target + '/CL' + str(i) + '/'
            ### save the best model ###

            predictions = trained_model.calc_local_params(trn_dataset)
            preds = LP['resp']
            predictions = get_assignments(tweetid_answer_counters,preds)
            pdb.set_trace()
            #save_bnpy_model(model_dir, trained_model, info_dict)

    # elif model_name == "dpmm":
    #     # https://bnpy.readthedocs.io/en/latest/examples/06_we8there/run-02-demo=mix_vb+proposals-model=dp_mix+mult.html?highlight=DPMixtureModel#train-with-birth-and-merge-proposals
    #     merge_kwargs = dict(
    #         m_startLap=5,
    #         m_pair_ranking_procedure='elbo',
    #         m_pair_ranking_direction='descending',
    #         m_pair_ranking_do_exclude_by_thr=1,
    #         m_pair_ranking_exclusion_thr=-0.0005,
    #         )
    #
    #     trained_model, info_dict = bnpy.run(
    #         trn_dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    #         doWriteStdOut=False, nTask=ITERATIONS,
    #         nLap=1000, convergeThr=0.0001, nBatch=1,
    #         K=2, initname='bregmankmeans+lam1+iter1',
    #         gamma0=FMM_DPMM_Gamma, lam=0.1,
    #         moves='birth,merge,shuffle',
    #         b_startLap=2, b_Kfresh=5, b_stopLap=10,
    #         **merge_kwargs)

        # gives us cluster likelihoods for each document
        # not needed here, but good to know
        #LP = dict()
        #LP = trained_model.obsModel.calc_local_params(trn_dataset, LP)
        #LP = trained_model.allocModel.calc_local_params(trn_dataset, LP)

        #LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

        # Summary step
        #SS = hmodel.get_global_suff_stats(Data, LP)

        # ELBO calculation
        #info_dict['curr_loss'] = -1 * trained_model.calc_evidence(trn_dataset)

        info_dict['Centroids'] = np.multiply(info_dict['SS'].WordCounts.transpose(), np.reciprocal(info_dict['SS'].SumWordCounts)).transpose()

        i = len(info_dict['Centroids'])

        ### Stroke trained model ###
        model_dir = self.MM_LOG_DIR + 'gamma0=' + str(FMM_DPMM_Gamma) + '/' + output_name + "_" + split_prep + "_" + target + '/CL' + str(i) + '/'
        ### save the best model ###
        save_bnpy_model(model_dir, trained_model, info_dict)
    pdb.set_trace()
    print ("Training completed and saved to "+model_dir)
    models_dir = folder + '/gamma0=' + str(FMM_DPMM_Gamma) + '/' + target
    bnpy_model_selection(trn_dataset,models_dir,tweetid_answer_counters,choices.values(),message_dict,output_name,target)
    #print('\n===== Trained model * ' + str(output_name) + ' * stored in directory ' + str(self.MM_LOG_DIR) + '=====\n')

def bnpy_model_selection(trn_dataset,model_location,tweetid_answer_counters, choices, message_dict, output_name, target):
    '''
    Load a trained bnpy HModel and find cluster assignments for test set
    :param tweetid_answer_counters: dictionary of the form {tweet_id: [ct_ans1, ct_ans2, ct_ans3 ...]}
    :param choices: Possible answers
    :param dataset_name: file name containing tweetids of test samples
    :param data_dir: directory containing the trained models and dataset files
    :param model_name: type of model to test('fmm' or 'dpmm')
    :return: dictionary containing results of test
    '''
    # print(output_name)

    # # Read data splits from file, NOT generate each time
    # with open(SPLIT_LOG_DIR + output_name + "_" + split_prep + ".json") as fp:
    #     results_dict = json.load(fp)
    # dev_items = results_dict['dev_set']
    # test_items = results_dict['test_set']

    # # Keep the order of items (using for loop)
    # dev_answer_counters = {}
    # for k in dev_items:
    #     dev_answer_counters[k] = tweetid_answer_counters[k]
    # dev_vectors = get_ans_vectors(dev_answer_counters)
    # test_answer_counters = {}
    # for k in test_items:
    #     test_answer_counters[k] = tweetid_answer_counters[k]
    # test_vectors = get_ans_vectors(test_answer_counters)

    # if target == 'label':
    #     ### convert data to bag of words format ###
    #     bow_info_dev = data_prep_bnpy(dev_answer_counters, choices)
    #     bow_info_tst = data_prep_bnpy(test_answer_counters, choices)
    # elif target == 'text':
    #     bow_info_dev = language_prep_bnpy(message_dict, dev_items)
    #     bow_info_tst = language_prep_bnpy(message_dict, test_items)

    # ### create a bnpy DataObj ###
    # dev_dataset = bnpy.data.BagOfWordsData(**bow_info_dev)
    # tst_dataset = bnpy.data.BagOfWordsData(**bow_info_tst)

    ### model location ###
    #model_location = self.MM_LOG_DIR + 'gamma0=' + str(FMM_DPMM_Gamma) + '/' + output_name + "_" + split_prep + "_" + target
    tst_dataset = trn_dataset
    test_vectors = tweetid_answer_counters

    print(model_location)
    model_dict = load_bnpy_model(model_location)

    test_results = dict()
    cluster_centroid_samples = {}

    clusters = list(model_dict.keys())
    clusters.sort()

    likelies = []
    ks, ms = [], []
    pdb.set_trace()
    for k in clusters:

        hmodel, info_dict = model_dict[k]
        # Compute likelihood
        ll = -info_dict['loss']
        likelies.append(ll)

        ### calculate the local parameters (includes class probabilities) of this trained model on test data ###
        # https://bnpy.readthedocs.io/en/latest/allocmodel/mix/index.html#accessing-learned-cluster-assignments
        LP = hmodel.calc_local_params(tst_dataset)
        test_pred_vectors = LP['resp']  # cluster probabilities for each sample

        # # https://bnpy.readthedocs.io/en/latest/allocmodel/mix/index.html#accessing-learned-cluster-probabilities
        # pi0 = hmodel.allocModel.get_active_comp_probs()  # overall component probabilities

        # cluster centroids
        pdb.set_trace()
        centroids = np.asarray(info_dict['Centroids'])

        ents = list(tests(test_vectors, test_pred_vectors))
        ent = sum([x / math.log(len(choices)) - y / math.log(float(k)) for x, y, z, w in ents])
        test_ent = ent / float(len(test_items))
        maxy = sum([w * math.log(float(k)) - z * math.log(len(choices)) for x, y, z, w in ents])

        cluster_assignments, dist_by_cluster, assignments_per_cluster = self.get_assignments(test_vectors, test_pred_vectors)
        cross = self.get_perplexity(test_vectors, cluster_assignments, dist_by_cluster, assignments_per_cluster)

        # for item, true, pred in zip(test_items, test_vectors, test_pred_vectors):
        #     print(item, answers2pct(true), pred)

        # Find optimal model with k directly (see k in manuscript)
        centroid_samples, word_dist_by_doc, word_dist_by_cluster = find_samples_around_centroids(message_dict, test_items, test_vectors, test_pred_vectors, centroids, k, output_name)
        # print(word_dist_by_cluster, np.sum(word_dist_by_cluster, axis=1), len(np.sum(word_dist_by_cluster, axis=1)))

        new_ents = list(tests(word_dist_by_doc, test_pred_vectors))
        new_ent = sum([x/math.log(word_dist_by_doc.shape[1])-y/math.log(float(k)) for x,y,z,w in new_ents])
        new_test_ent = new_ent / float(len(test_items))
        cross1 = self.get_perplexity(word_dist_by_doc, cluster_assignments, np.transpose(word_dist_by_cluster), assignments_per_cluster)

        test_results[k] = {"max_diff": maxy, "ent": test_ent, "ll": ll, "cross": cross, "word_ent": new_test_ent, "word_cross": cross1}

    #write_model_logs_to_json(self.MM_LOG_DIR + 'gamma0=' + str(FMM_DPMM_Gamma) + '/', test_results, output_name + "_" + split_prep + "_" + target + "_onKLtest")
    print('\n===== Test results for * ' + str(output_name) + ' * stored in directory ' + str(self.MM_LOG_DIR) + ' =====\n')


def bnpy_best_result(self, split_prep, tweetid_answer_counters, choices, message_dict, output_name, model_name, target):
    '''
    ADAPTED FROM "bnpy_test"
    Load a trained bnpy HModel and find cluster assignments for test set
    :param tweetid_answer_counters: dictionary of the form {tweet_id: [ct_ans1, ct_ans2, ct_ans3 ...]}
    :param choices: Possible answers
    :param dataset_name: file name containing tweetids of test samples
    :param data_dir: directory containing the trained models and dataset files
    :param model_name: type of model to test('fmm' or 'dpmm')
    :return: dictionary containing results of test
    '''
    print(output_name)

    # Read data splits from file, NOT generate each time
    with open(SPLIT_LOG_DIR + output_name + "_" + split_prep + ".json") as fp:
        results_dict = json.load(fp)
    dev_items = results_dict['dev_set']
    test_items = results_dict['test_set']

    # Keep the order of items (using for loop)
    dev_answer_counters = {}
    for k in dev_items:
        dev_answer_counters[k] = tweetid_answer_counters[k]
    dev_vectors = get_ans_vectors(dev_answer_counters)
    test_answer_counters = {}
    for k in test_items:
        test_answer_counters[k] = tweetid_answer_counters[k]
    test_vectors = get_ans_vectors(test_answer_counters)

    if target == 'label':
        ### convert data to bag of words format ###
        bow_info_dev = data_prep_bnpy(dev_answer_counters, choices)
        bow_info_tst = data_prep_bnpy(test_answer_counters, choices)
    elif target == 'text':
        bow_info_dev = language_prep_bnpy(message_dict, dev_items)
        bow_info_tst = language_prep_bnpy(message_dict, test_items)

    ### create a bnpy DataObj ###
    dev_dataset = bnpy.data.BagOfWordsData(**bow_info_dev)
    tst_dataset = bnpy.data.BagOfWordsData(**bow_info_tst)

    ### model location ###
    model_dict = dict()
    for gamma in [0.25, 0.5, 1, 25, 50, 75]:
        model_location = self.MM_LOG_DIR[:-1] + '/gamma0=' + str(gamma) + '/' + output_name + "_" + split_prep + "_" + target
        for (dirpath, dirnames, filenames) in os.walk(model_location):
            for di in dirnames:
                nClusters = int(di.strip("CL"))
                pickle_fpath = os.path.join(model_location, di) + '/best_model.pklz'
                with gzip.open(pickle_fpath, 'rb') as model:
                    m = pickle.load(model)
                model_dict[str(gamma) + '__' + str(nClusters)] = (m['best_model'], m['info'])  # tuple

    cluster_centroid_samples = {}

    ks, ms = [], []
    clusters = list(model_dict.keys())
    clusters.sort()
    for k in clusters:
        hmodel, info_dict = model_dict[k]
        # Dev for selection
        LP = hmodel.calc_local_params(dev_dataset)
        dev_pred_vectors = LP['resp']  # cluster probabilities for each sample
        cluster_assignments, dist_by_cluster, assignments_per_cluster = self.get_assignments(dev_vectors, dev_pred_vectors)
        cross = self.get_perplexity(dev_vectors, cluster_assignments, dist_by_cluster, assignments_per_cluster)
        # print(k, cross)
        ks.append(k)
        ms.append(cross)
    print('BEST', ks[ms.index(min(ms))], min(ms))

    bestgamma = ks[ms.index(min(ms))].split('__')[0]
    bestk = ks[ms.index(min(ms))].split('__')[1]

    test_output = self.MM_LOG_DIR[:-1] + '/gamma0=' + str(bestgamma) + '/' + output_name + "_" + split_prep + "_" + target + "_onKLtest" + ".json"
    print(test_output)
    with open(test_output) as fp:
        test_results = json.load(fp)
    best_test = test_results[bestk]
    print(bestgamma, bestk, "%.2f" % best_test["cross"])
    print('\n')


def get_assignments(test_vectors, test_pred_vectors):
    for test_vect in test_vectors:
        first_id = test_vect
        break
    dist_by_cluster = [[0.0] * len(test_vectors[first_id]) for i in test_pred_vectors[0]]
    #assignments_per_cluster = [0.0] * len(test_pred_vectors[0])
    cluster_assignments = [argmax(tpv) for tpv in test_pred_vectors]
    predictions = []
    for i,test_i in zip(range(len(test_pred_vectors)),test_vectors):
        dist_by_cluster[cluster_assignments[i]] = [j + k for j,k in zip(dist_by_cluster[cluster_assignments[i]], test_vectors[test_i])]
        #assignments_per_cluster[cluster_assignments[i]] += 1
        predictions.append(dist_by_cluster[cluster_assignments[i]])

    return build_prob_distribution(predictions),cluster_assignments
    #return cluster_assignments, dist_by_cluster, assignments_per_cluster

def language_prep_bnpy(message_dict,dict_to_encode):

    vocab_list = []
    for message in message_dict:
        # Naive tokenization
        tokens = message.split()
        # Advanced tokenization 
        # tokens = get_normalized_tokens(message, set())
        for token in tokens:
            if token not in vocab_list:
                vocab_list.append(token)

    word_ids_per_doc = [x for x in range(len(vocab_list))]
    nWords = len(word_ids_per_doc)

    word_id = []
    word_count = []
    doc_range = [0]
    i = 0

    # create a list of word ids and non zero word counts for each document
    # for index, msg_id in enumerate(subitems):
    #     message = message_dict[msg_id]
    #     # Naive tokenization
    for message in dict_to_encode:
        tokens = message.split()
        # Advanced tokenization 
        # tokens = get_normalized_tokens(message, set())

        ans_counts = [0] * nWords
        for token in tokens:
            try:
                ans_counts[vocab_list.index(token)] += 1
            except:
                continue
        ans_counts = np.array(ans_counts)

        # find words with count > 0
        nz_word_ids = np.flatnonzero(ans_counts)
        nz_word_counts = ans_counts.ravel()[nz_word_ids]

        word_id.extend(nz_word_ids.tolist())
        word_count.extend(nz_word_counts.tolist())

        nWords_in_doc = len(nz_word_ids)
        if nWords_in_doc != 0:
            i += nWords_in_doc
            doc_range.append(i)
        # else:
        #     # print(index, tokens)
        #     print(ans_counts, nz_word_ids, nz_word_counts, nWords_in_doc, i)

    bow_info = {
        'word_id' : np.array(word_id),
        'word_count' : np.array(word_count),
        'doc_range' : np.array(doc_range),
        'vocab_size' : np.array(nWords),
        'vocabList' : vocab_list,
        'logFunc' : False
    }

    return bow_info