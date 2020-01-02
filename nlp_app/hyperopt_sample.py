from os.path import join

import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tokenizer import tokenizer

BASE_DIR = '../sample_code/assets/dialogue_agent_data'

training_data = pd.read_csv(join(BASE_DIR, 'training_data.csv'))
train_text = training_data['text']
train_labels = training_data['label']

vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_text)

tr_labels, val_labels, tr_vectors, val_vectors = \
    train_test_split(train_labels, train_vectors, random_state=42)

def objective(args):
    classifier = RandomForestClassifier(n_estimators=int(args['n_estimators']),
                                        max_features=args['max_features'])
    classifier.fit(tr_vectors, tr_labels)
    val_predictions = classifier.predict(val_vectors)
    accuracy = accuracy_score(val_predictions, val_labels)
    return -accuracy

max_features_choices = ('sqrt', 'log2', None)
space = {
    'n_estimators': hp.quniform('n_estimators', 10, 500, 10),
    'max_features': hp.choice('max_features', max_features_choices),
}

best = fmin(objective, space=space, algo=tpe.suggest, max_evals=30)
print(best['n_estimators'])
print(best['max_features'])

best_classifier = RandomForestClassifier(
    n_estimators=int(best['n_estimators']),
    max_features=max_features_choices[best['max_features']]
)
best_classifier.fit(train_vectors, train_labels)

test_data = pd.read_csv(join(BASE_DIR, 'test_data.csv'))
test_texts = test_data['text']
test_labels = test_data['label']

test_vectors = vectorizer.transform(test_texts)
predictions = best_classifier.predict(test_vectors)
print(accuracy_score(test_labels, predictions))
