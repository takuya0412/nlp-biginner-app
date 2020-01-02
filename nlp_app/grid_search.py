from os.path import join

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from tokenizer import tokenizer

BASE_DIR = '../sample_code/assets/dialogue_agent_data/'

training_data = pd.read_csv(join(BASE_DIR, 'training_data.csv'))
train_texts = training_data['text']
train_labels = training_data['label']

vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)

parameters = {
    'n_estimators': [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
    'max_features': ('sqrt', 'log2', None)
}
classifier = RandomForestClassifier()
gridsearch = GridSearchCV(classifier, parameters)

gridsearch.fit(train_vectors, train_labels)

print('Best params are: {}'.format(gridsearch.best_params_))

test_data = pd.read_csv(join(BASE_DIR, 'test_data.csv'))
test_texts = test_data['text']
test_labels = test_data['label']

test_vectors = vectorizer.transform(test_texts)
predictions = gridsearch.predict(test_vectors)
