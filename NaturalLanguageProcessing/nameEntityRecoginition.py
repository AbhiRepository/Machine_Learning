import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

traintext = state_union.raw("2005-GWBush.txt")
sampletext = state_union.raw("2006-GWBush.txt")

ourTokenizer = PunktSentenceTokenizer(traintext)
tokenizeText = ourTokenizer.tokenize(sampletext)

for i in tokenizeText:
	words = nltk.word_tokenize(i)
	tagged = nltk.pos_tag(words)
	nameEnt = nltk.ne_chunk(tagged, binary=True)
	nameEnt.draw()



