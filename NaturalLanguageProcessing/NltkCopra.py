from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg
from nltk.corpus import wordnet

synonyms = []
antonyms = []

for syns in wordnet.synsets("good"):
	for l in syns.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())

print(synonyms)
print(antonyms)

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('plane.n.01')
print(w1.wup_similarity(w2))



























