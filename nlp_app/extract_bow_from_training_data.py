from os.path import dirname, join, normpath
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenizer


BASE_DIR = normpath(dirname(__file__))
csv_path = join(BASE_DIR, "../sample_code/assets/dialogue_agent_data/training_data.csv")
training_data = pd.read_csv(csv_path)
training_texts = training_data['text']


vectorizer = CountVectorizer(tokenizer=tokenizer)
vectorizer.fit(training_texts)
bow = vectorizer.transform(training_texts)
