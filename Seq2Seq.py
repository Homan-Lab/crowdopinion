from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop,Adam
# from keras.optimizers import Adam

from keras.models import Model
from keras.layers import Dense, Embedding, Input

from keras.layers import Activation
from keras.layers import LSTM
from keras.callbacks import TerminateOnNaN
import numpy as np
import keras
from matplotlib import pyplot
import pdb

# python2 old keras libraries
# from keras.optimizers import Adam
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical, plot_model
# from keras.models import Model, Sequential
# from keras.layers import Dense, Embedding, Input
# from keras.layers import Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout
# from keras.layers import BatchNormalization, Activation
# from keras.layers import LSTM, GRU, Bidirectional
# from keras.layers import GlobalAveragePooling1D
# from keras.preprocessing import sequence
# from keras.models import load_model
# from keras.callbacks import EarlyStopping, TerminateOnNaN
# import numpy as np
# import keras
# from matplotlib import pyplot
# import pdb
# end

class Seq2Seq:
    def __init__(self, num_decoder_tokens = 11, max_words = 1000, max_len = 150, latent_dim = 11, word_index=None):
    # Define an input sequence and process it.

        self.num_decoder_tokens = num_decoder_tokens#+2
        encoder_inputs = Input(shape=(None,))
        #if word_index:
        #    encoder_embedding_layer = Embedding(max_words, 50, input_length=max_len, weights=[word_index], trainable=True)(encoder_inputs)
        #else:
        encoder_embedding_layer = Embedding(max(word_index.values())+1, 50, input_length=max_len)(encoder_inputs)
        encoder_lstm_layer, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding_layer)
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm_layer = LSTM(latent_dim, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
        #layer = Dense(11)(decoder_lstm_layer)
        #layer = Activation('relu')(layer)
        #layer = Dropout(0.5)(layer)
        decoder_outputs = Dense(self.num_decoder_tokens)(decoder_lstm_layer)
        decoder_outputs = Activation('softmax')(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`


        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # Compile & run training
        #model.summary()
        #model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])


        # Note that `decoder_target_data` needs to be one-hot encoded,
        # rather than sequences of integers like `decoder_input_data`!


        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(
             decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = Dense(self.num_decoder_tokens, activation='softmax')(decoder_outputs)
        self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)


    def fit(self, X, Y, batch_size=50, epochs=20, validation_data=None):

        self.model.compile(loss='kullback_leibler_divergence', optimizer=Adam(), metrics=['accuracy']) #, metrics=['accuracy'])
        #pdb.set_trace()
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[TerminateOnNaN(),keras.callbacks.TensorBoard(log_dir='./dllogs')],validation_data=validation_data) #EarlyStopping(monitor='val_loss', min_delta=0.0001)])

        # #pyplot.plot(history.history['val_loss'])
        # pyplot.plot(history.history['loss'])
        # pyplot.title('model train vs validation loss')
        # pyplot.ylabel('loss')
        # pyplot.xlabel('epoch')
        # pyplot.legend(['train', 'validation'], loc='upper right')
        # pyplot.savefig("validation_loss_seq2seq.pdf")
        # pyplot.clf()

        return history

    def predict(self, X, batch_size=32):
        states_value = self.encoder_model.predict(X)
        target_seq = np.zeros((states_value[1].shape[0], 1, self.num_decoder_tokens))
        target_seq[:,:,0] = 1
        #for state_value in states_value:
        # Generate empty target sequence of length 1.
        #target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        #    target_seq[0,0,0] = 1.0
        #pdb.set_trace()
        output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value, batch_size=batch_size)
        #    output_vals.append(output_tokens)
        return output_tokens

    def evaluate(self, X, Y, batch_size=32):
        return self.model.evaluate(X, Y, batch_size=batch_size)

    def save(self, name):
        self.model.save(name)
