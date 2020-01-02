import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')

tokyo = model['tokyo']
japan = model['japan']
france = model['france']

v = tokyo - japan + france

print('tokyo - japan + france = {}'.format(
    model.wv.similar_by_vector(v, topn=1)
))
