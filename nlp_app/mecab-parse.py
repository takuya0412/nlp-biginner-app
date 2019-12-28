import MeCab

tagger = MeCab.Tagger()
print(tagger.parse('今夜は月が綺麗ですね、いえ、あなたのことではありませんよ!'))
