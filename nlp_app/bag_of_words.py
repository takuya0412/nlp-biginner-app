from tokenizer import tokenizer


def calc_bow(tokenized_texts):
    vocabulary = {}
    for tokenized_text in tokenized_texts:
        for token in tokenized_text:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    n_vocab = len(vocabulary)

    bow = [[0] * n_vocab for i in range(len(tokenized_texts))]

    for i, tokenized_text in enumerate(tokenized_texts):
        for token in tokenized_text:
            index = vocabulary[token]
            bow[i][index] += 1
    return vocabulary, bow


if __name__ == "__main__":
    texts = []
    statement = input(">>")
    while statement:
        texts.append(statement)
        statement = input(">>")

    wakati_text = [tokenizer(text) for text in texts]
    print("tokenized_text:{}".format(wakati_text))
    vocabulary, bow = calc_bow(tokenized_texts=wakati_text)
    print("vocabulary:\n {}".format(vocabulary))
    print("bow: \n{}".format(bow))
