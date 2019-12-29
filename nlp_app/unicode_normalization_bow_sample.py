import unicodedata

from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenizer


def normalize_and_tokenize(text):
    normalized = unicodedata.normalize('NFKC', text)
    return tokenizer(normalized)


texts = [
    'ディスプレイを買った',
    'ディスプレイを買った'
]

vectorizer = CountVectorizer(tokenizer=normalize_and_tokenize)
vectorizer.fit(texts)

print('bow:')
print(vectorizer.transform(texts).toarray())

print('vocabulary:')
print(vectorizer.vocabulary_)
