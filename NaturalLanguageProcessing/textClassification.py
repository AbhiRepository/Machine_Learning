from nltk.corpus import movie_reviews
import nltk
import random
import pickle

document = []
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		document.append((movie_reviews.words(fileid),category))

random.shuffle(document)

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]
# print(word_features)

# document = ["words","category","words","category",..]
# word_features = [top 3000 words]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = []
for (rev,cat) in document:
	featuresets.append((find_features(rev),cat))

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("classifier accuracy percent:",(nltk.classify.accuracy(classifier,testing_set)))
print(classifier.show_most_informative_features(15))

# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()

















