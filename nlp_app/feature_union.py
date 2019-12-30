import scipy
from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenizer


texts = [
    '私は私のことが好きなあなたが好きです',
    '私はラーメンが好きです。',
    '富士山は日本一高い山です',
]

word_bow_vectorizer = CountVectorizer(tokenizer=tokenizer)
char_bigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))

word_bow_vectorizer.fit(texts)
char_bigram_vectorizer.fit(texts)

word_bow = word_bow_vectorizer.transform(texts)
char_bigram = char_bigram_vectorizer.transform(texts)

feat = scipy.sparse.hstack((word_bow, char_bigram))
