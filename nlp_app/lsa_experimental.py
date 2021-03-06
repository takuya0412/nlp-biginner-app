import unicodedata
import neologdn
import pandas as pd
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from draw_barcharts import draw_barcharts, plt

tagger = MeCab.Tagger()


def tokenize(text):
    text = unicodedata.normalize('NFKC', text)
    text = neologdn.normalize(text)
    text = text.lower()

    node = tagger.parseToNode(text)

    result = []
    while node:
        features = node.feature.split(',')

        if features[0] != 'BOS/EOS':
            if features[0] not in ['助詞', '助動詞']:
                token = features[6] if features[6] != '*' else node.surface
                result.append(token)

        node = node.next
    return result

texts = [
    '車は速く走る',
    'バイクは速く走る',
    '自転車はゆっくり走る',
    '三輪車はゆっくり走る',
    'プログラミングは楽しい',
    'Pythonは楽しい'
]

vectorizer = CountVectorizer(tokenizer=tokenize)
vectorizer.fit(texts)
bow = vectorizer.transform(texts)

bow_table = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())
print('Shape: {}'.format(bow.shape))
print(bow_table)

draw_barcharts(bow.toarray(), vectorizer.get_feature_names(), texts)
plt.show()

svd = TruncatedSVD(n_components=4, random_state=42)

svd.fit(bow)

decomposed_features = svd.transform(bow)

print("shape: {}".format(decomposed_features.shape))
print(decomposed_features)

draw_barcharts(decomposed_features, range(svd.n_components), texts)
plt.show()

draw_barcharts(svd.components_, vectorizer.get_feature_names(), range(svd.n_components))
plt.show()

print("重要度:{}".format(svd.singular_values_))

