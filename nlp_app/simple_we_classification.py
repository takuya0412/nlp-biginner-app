from os.path import join
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from tokenizer import tokenizer

model = Word2Vec.load('../../../word_embeddings_model/latest-ja-word2vec-gensim-model/word2vec.gensim.model')


def calc_text_feature(text):
    tokens = tokenizer(text)
    word_vectors = np.empty((0, model.wv.vector_size))

    for token in tokens:
        try:
            word_vector = model[token]
            word_vectors = np.vstack((word_vectors, word_vector))

        except KeyError:
            pass

    if word_vectors.shape[0] == 0:
        return np.zeros(model.wv.vector_size)
    return np.sum(word_vectors, axis=0)


BASE_DIR = '../sample_code/assets/dialogue_agent_data/'
training_data = pd.read_csv(join(BASE_DIR, 'training_data.csv'))
test_data = pd.read_csv(join(BASE_DIR,'test_data.csv'))

X_train = np.array([calc_text_feature(text) for text in training_data['text']])
y_train = np.array(training_data['label'])

X_test = np.array([calc_text_feature(text) for text in test_data['text']])
y_test = np.array(test_data['label'])

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
