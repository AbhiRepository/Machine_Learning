from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

example_text = 'hi my name is Mr. Ahishek, what are you doing here. I wanna study hard. I am good at it.'
example_text = word_tokenize(example_text)

for word in example_text:
	if word in set(stopwords.words('english')):
		print(word)


