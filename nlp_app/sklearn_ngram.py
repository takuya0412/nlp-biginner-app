from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenizer

texts = [
    '東京から大阪に行く',
    '大阪から東京に行く'
]

vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))
vectorizer.fit(texts)
bi_gram = vectorizer.transform(texts)
