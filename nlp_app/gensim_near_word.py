import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')

tokyo = model['tokyo']

print(model.wv.similar_by_vector(tokyo))
