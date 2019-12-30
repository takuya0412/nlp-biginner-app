from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizer import tokenizer


texts = [
    '東京から大阪に行く',
    '大阪から東京に行く'
]

vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(2,2))
vectorizer.fit(texts)
tfidf_ngram = vectorizer.transform(texts)
