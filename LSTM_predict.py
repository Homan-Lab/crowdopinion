import argparse
import sys
import pdb
from Seq2Seq_predict import Seq2Seq_predict
from ldl_utils import get_data_dict, vectorize,read_json
from helper_functions_LSTM_TF import get_feature_vectors,compile_tweet_dict,build_text_labels
from LSTM_utils import keras_feature_prep
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
from helper_functions import create_folder
from keras.models import load_model


def LSTM_predict(test_messages,test_answer_counters,label_dict,model,output,batchsize):
    predict_text = []
    predict_labels = []
    predict_text,predict_labels = build_text_labels(test_messages,test_answer_counters)
    test_features, test_word_index = keras_feature_prep(predict_text)
    LSTM_model = Seq2Seq_predict(num_decoder_tokens=predict_labels.shape[1], word_index=test_word_index,model_path=model)
    pdb.set_trace()
    #LSTM_model = pickle.load(open(model, "rb", -1)
    predict = LSTM_model.predict(test_features,batch_size=batchsize)
    predict = predict[:, 0, 1:-1]
    pdb.set_trace()

def LSTM_processing(input_train_model,input_dev_file_name, output_file_name,batch_size):

    dev_answer_counters = defaultdict(list)
    JSONfile = read_json(input_dev_file_name)
    predict_message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, choices) = get_data_dict(JSONfile["dictionary"])
    predict_answer_counters = get_feature_vectors(fdict, JSONfile["data"])
    LSTM_predict(predict_message_dict,predict_answer_counters,choices,input_train_model,output_file_name,batch_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_model", help="Input training LSTM model")
    parser.add_argument("--input_dev_file", help="Input dev file JSON name")
    parser.add_argument("--batch_size", help="Batch Size")
    parser.add_argument("--output_file", help="Output file name")
    args = parser.parse_args()

    LSTM_processing(args.input_train_model,args.input_dev_file, args.output_file,int(args.batch_size))

if __name__ == '__main__':
    main()
