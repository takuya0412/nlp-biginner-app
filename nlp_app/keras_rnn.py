from os.path import join

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import Conv1D, Dense, Embedding, Flatten, MaxPooling1D, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

from tokenizer import tokenizer


def tokens_to_sequence(we_model, tokens):
    sequence = []
    for token in tokens:
        try:
            sequence.append(we_model.wv.vocab[token].index + 1)
        except KeyError:
            pass
    return sequence


def get_keras_embedding(keyed_vectors, *args, **kwargs):
    weights = keyed_vectors.vectors
    word_num = weights.shape[0]
    embedding_dim = weights.shape[1]
    zero_word_vector = np.zeros((1, weights.shape[1]))
    weights_with_zero = np.vstack((zero_word_vector, weights))
    return Embedding(input_dim=word_num + 1,
                     output_dim=embedding_dim,
                     weights=[weights_with_zero],
                     *args, **kwargs)


if __name__ == "__main__":
    BASE_DIR = '../sample_code/assets/dialogue_agent_data/'
    we_model = Word2Vec.load(
        '../../../word_embeddings_model/latest-ja-word2vec-gensim-model/word2vec.gensim.model')

    training_data = pd.read_csv(join(BASE_DIR, 'training_data.csv'))
    training_texts = training_data['text']
    tokenized_training_texts = [tokenizer(text) for text in training_texts]
    training_sequence = [tokens_to_sequence(we_model, tokens) for tokens in tokenized_training_texts]

    MAX_SEQUENCE_LENGTH = 20

    x_train = pad_sequences(training_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    y_train = np.asarray(training_data['label'])
    n_classes = max(y_train) + 1

    model = Sequential()

    model.add(get_keras_embedding(we_model.wv,
                                  input_shape=(MAX_SEQUENCE_LENGTH, ),
                                  mask_zero=True,
                                  trainable=False))
    model.add(LSTM(units=256))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, to_categorical(y_train), epochs=50)

    test_data = pd.read_csv(join(BASE_DIR, 'test_data.csv'))

    test_texts = test_data['text']
    tokenized_test_texts = [tokenizer(text) for text in test_texts]
    test_sequences = [tokens_to_sequence(we_model, tokens)for tokens in tokenized_test_texts]
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = np.asarray(test_data['label'])

    y_pred = np.argmax(model.predict(x_test), axis=1)

    print(accuracy_score(y_test, y_pred))
