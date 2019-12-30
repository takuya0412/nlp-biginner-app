import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

rx_periods = re.compile(r'[.。. ]+')

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [
            {
                'length': len(text),
                'num_sentences': len([sent for sent in rx_periods.split(text)
                                    if len(sent) > 0])
            }
            for text in texts
        ]

combined = FeatureUnion([
    ('stats', Pipeline([
        ('stats', TextStats()),
        ('vect', DictVectorizer()),
    ])),
    ('char_bigram', CountVectorizer(analyzer='char', ngram_range=(2, 2))),
])

texts = [
    'こんにちは。こんばんは。',
    '焼肉が食べたい'
]

combined.fit(texts)
feat = combined.transform(texts)

