from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenizer

texts = [
    "私は京都が好きです。",
    "私はラーメンが好きです。",
    "富士山は日本一高い山です"
]

vectrizer = CountVectorizer(tokenizer=tokenizer)
vectrizer.fit(texts)
bow = vectrizer.transform(texts)

print("bow:{}".format(bow))
