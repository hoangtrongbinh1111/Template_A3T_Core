from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dropout, Dense, GRU, Bidirectional, SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import tqdm
import numpy as np
from pathlib import Path
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
EMBEDDING_SIZE = 100
SEQUENCE_LENGTH = 100
TEST_SIZE = 0.5
FILTERS = 70
BATCH_SIZE = 100
EPOCHS = 5
OUTPUT_FOLDER = "save"
URL_PATH = './data/data.csv'
lstm_units = 1024
SEQUENCE_LENGTH = 100
# converts the utf-8 into tokinized characters

# Load data


def get_embedding_vectors(tokenizer, dim=EMBEDDING_SIZE):
    embedding_index = {}
    with open(f"embed/glove.6B.{dim}d.txt", 'r', encoding='utf8', errors='ignore') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_train_data(train_data_dir, val_size, embed_size=EMBEDDING_SIZE):
    """load urls from the specified directory"""
    if Path(train_data_dir).is_file() and Path(train_data_dir).suffix == '.csv':
        data = pd.read_csv(train_data_dir, sep=',').sample(
            n=1000, random_state=4)
        urls, temp = list(data['url']), list(data['Label'])
        labels = []
        for l in temp:
            if l == 1:
                labels.append(l)
            else:
                labels.append(0)
        temp_trainX, temp_valX, temp_trainY, temp_valY = train_test_split(
            urls, labels, test_size=val_size, random_state=4)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(urls)
        trainX = tokenizer.texts_to_sequences(temp_trainX)
        trainX = np.array(trainX)
        trainY = np.array(temp_trainY)
        trainX = pad_sequences(trainX, maxlen=SEQUENCE_LENGTH)
        trainY = tf.keras.utils.to_categorical(trainY)
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        valX = tokenizer.texts_to_sequences(temp_valX)
        valX = np.array(valX)
        valY = np.array(temp_valY)
        valX = pad_sequences(valX, maxlen=SEQUENCE_LENGTH)
        valY = tf.keras.utils.to_categorical(valY)
        valX, valY = shuffle(valX, valY, random_state=0)

        embedding_matrix = get_embedding_vectors(tokenizer, embed_size)
        return trainX, valX, trainY, valY, tokenizer, embedding_matrix
    return


def get_model(tokenizer, embedding_matrix, rnn_cell, num_neurons, num_layers, embed_size=EMBEDDING_SIZE):  # builds the lstm model
    model = Sequential()
    layer = Embedding(len(tokenizer.word_index)+1, 
                EMBEDDING_SIZE,
                trainable=False,
                input_shape=(SEQUENCE_LENGTH,))
        # layer.set_weights([embedding_matrix])
    model.add(layer)
    for i in range (num_layers-1):
        if rnn_cell == 'gru':
            model.add (GRU(num_neurons, recurrent_dropout=0.3, return_sequences= True))
        elif rnn_cell == 'lstm':
            model.add(LSTM(num_neurons, recurrent_dropout=0.3, return_sequences= True))
        else:
            # model.add(Bidirectional(LSTM(rnn_units, recurrent_dropout=0.3, return_sequences= True)))
            model.add(SimpleRNN(num_neurons, recurrent_dropout = 0.3, return_sequences = True))

    ### last rnn layer
    if rnn_cell == 'gru':
        model.add (GRU(num_neurons, recurrent_dropout=0.3))
    elif rnn_cell == 'lstm':
        model.add(LSTM(num_neurons, recurrent_dropout=0.3))
    else:
        # model.add(Bidirectional(LSTM(rnn_units, recurrent_dropout=0.3), merge_mode = "concat"))
        model.add (SimpleRNN (num_neurons, recurrent_dropout = 0.3))

    model.add (Dense(64))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation="softmax"))
    
    return model


def get_test_data(train_data_dir, tokenizer):
    """load urls from the specified directory"""
    if Path(train_data_dir).is_file() and Path(train_data_dir).suffix == '.csv':
        # .sample(n = 10000, random_state = 2)
        data = pd.read_csv(train_data_dir, sep=',')
        urls, temp = list(data['url']), list(data['Label'])
        labels = []
        for l in temp:
            if l == 1:
                labels.append(l)
            else:
                labels.append(0)
        trainX = tokenizer.texts_to_sequences(urls)
        trainX = pad_sequences(trainX, maxlen=SEQUENCE_LENGTH)
        trainX = np.array(trainX)
        trainY = np.array(labels)
        
        trainY = tf.keras.utils.to_categorical(trainY)
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        return trainX, trainY
    return

def get_test_review_data(train_data_dir):
    """load urls from the specified directory"""
    if Path(train_data_dir).is_file() and Path(train_data_dir).suffix == '.csv':
        # .sample(n = 10000, random_state = 2)
        data = pd.read_csv(train_data_dir, sep=',')
        return len(data)
    return

def get_sample_data(url_sample, tokenizer):
    """load urls from the specified directory"""
    if url_sample != '' and url_sample != None:
        sample = tokenizer.texts_to_sequences(url_sample)
        sample = pad_sequences(sample, maxlen=SEQUENCE_LENGTH)
        sample = np.array(sample)
        # sample = shuffle(sample, random_state=0)
    return sample
