from sklearn.feature_extraction.text import CountVectorizer

texts = [
    '東京から大阪に行く',
    '大阪から東京に行く'
]

vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
vectorizer.fit(texts)
n_gram = vectorizer.transform(texts)
