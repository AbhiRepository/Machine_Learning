import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

ourTokenizer = PunktSentenceTokenizer(train_text)
tokenizedText = ourTokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenizedText[:1]:
			words = nltk.word_tokenize(i)
			#tagged = nltk.pos_tag(words)
			#print(tagged)
			print(words)

	except Exception as e:
		print(str(e))

process_content()










