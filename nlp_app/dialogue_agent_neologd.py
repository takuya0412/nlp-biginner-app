from os.path import dirname, join, normpath
import unicodedata
import neologdn

import MeCab
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

MECAB_DIC_DIR = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd'

class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger('-d {}'.format(MECAB_DIC_DIR))

    def _tokenizer(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = neologdn.normalize(text)
        text = text.lower()

        node = self.tagger.parseToNode(text)

        result = []
        while node:
            features = node.feature.split(',')

            if features[0] != 'BOS/EOS':
                if features[0] not in ['助詞', '助動詞']:
                    token = features[6] if features[6] != '*' else node.surface
                    result.append(token)

            node = node.next
        return result

    def train(self, texts, labels):
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=self._tokenizer)),
            ('classifier', SVC()),
        ])

        pipeline.fit(texts, labels)

        self.pipeline = pipeline

    def predict(self, texts):
        return self.pipeline.predict(texts)


if __name__ == "__main__":
    BASE_DIR = normpath(dirname(__file__))

    training_data = pd.read_csv(join(BASE_DIR, '../sample_code/assets/dialogue_agent_data/training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'])

    with open(join(BASE_DIR, '../sample_code/assets/dialogue_agent_data/replies.csv')) as f:
        replies = f.read().split('\n')

    input_text = '名前を教えてよ'

    predictions = dialogue_agent.predict([input_text])
    predicted_class_id = predictions[0]

    print(replies[predicted_class_id])
