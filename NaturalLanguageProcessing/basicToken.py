import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

example = ['PRESIDENT', 'GEORGE', 'W.', 'BUSH', "'S", 'ADDRESS', 'BEFORE', 'A', 'JOINT', 'SESSION', 'OF', 'THE', 'CONGRESS', 'ON', 'THE', 'STATE', 'OF', 'THE', 'UNION', 'January', '31', ',', '2006', 'THE', 'PRESIDENT', ':', 'Thank', 'you', 'all', '.']
tagged = nltk.pos_tag(example)

print(tagged)










