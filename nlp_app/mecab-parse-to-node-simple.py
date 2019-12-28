import MeCab

tagger = MeCab.Tagger()

node = tagger.parseToNode('私はあなたであなたは私です。')

while node.next:
    print(node.next.surface)
    node = node.next
