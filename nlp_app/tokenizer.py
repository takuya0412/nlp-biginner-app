import MeCab

tagger = MeCab.Tagger()


def tokenizer(text):
    node = tagger.parseToNode(text)
    tokens = []

    while node:
        if node.surface:
            tokens.append(node.surface)
        node = node.next
    return tokens


