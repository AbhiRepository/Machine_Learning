from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
exampleText = 'It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once.'
exampleText = word_tokenize(exampleText)
print(exampleText)


for word in exampleText:
	print(ps.stem(word))
