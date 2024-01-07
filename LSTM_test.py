import argparse
import sys
import pdb
from ldl_utils import get_data_dict, get_feature_vectors, vectorize,read_json,compile_tweet_dict
from LSTM_utils import LSTM_testing
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work

ITERATIONS = 100
q1_LOWER = 2
q1_UPPER = 12
q2_LOWER = 2
q2_UPPER = 12
q3_LOWER = 5
q3_UPPER = 20
DS_iter = 1
target = 'label'
task = 'lda'

def LSTM_Processing(foldername,input,source):
    vects = defaultdict(list)
    tweet_dict = defaultdict(list)
    f = foldername + input
    j = read_json(f)
    f = foldername + "/Tweets.json"
    tweet_message_dict = compile_tweet_dict(f)
    (fdict, rdict) = get_data_dict(j["dictionary"])
    tweetid_answer_counters = get_feature_vectors(fdict, j["data"])
    LSTM_testing(tweetid_answer_counters, rdict, source, tweet_message_dict, task, target)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Input Trainning File JSON Name")
    parser.add_argument("--dev", help="Dev Set Location")
    parser.add_argument("--model", help="Output Model PKL Location")
    parser.add_argument("--id", help="Identifier")
    args = parser.parse_args()
    input = args.input_file
    dev = args.dev
    model = args.model
    id = args.id
    foldername = "tools/pre_processing/csv_import/jobQ123CF/"
    LSTM_Processing(foldername,input,id)

if __name__ == '__main__':
    main()
