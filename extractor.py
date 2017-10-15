import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

CORPUS = [
['the sky is good', 'pos'],
['sky is good and sky is beautiful','pos'],
['worst bad negative','neg'],
['the beautiful sky is so good','pos'],
['like and true for those good and beautiful', 'pos'],
['so bad negative worst','neg'],
['im so bad', 'neg']

]

doc = ['good']

def feature_extraction():
    corpus = []

    for rowCorpus in CORPUS:
        sentiment = rowCorpus[1]
        sentence = rowCorpus[0]
        featureVector = get_feature_vector(sentence)
        corpus.append((featureVector, sentiment))
    return corpus


def get_feature_vector(sentence):
    ngram_range = (1, 1)
    corpus = []
    corpus.append(sentence)

    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)

    featureVector = vectorizer.get_feature_names()
    return featureVector


def get_words_in_corpus(corpus):
    all_words = []
    for (text, sentiment) in corpus:
        all_words.extend(text)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = list(wordlist.keys())

    return word_features

def extract_features(sentence):
    corpora = set(sentence)
    features = {}
    for word in word_features:
        features[word] = (word in corpora)
    return features

def corpus_classfication(corpus, training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # print(classifier.classify(extract_features(['worst', 'sentence'])))
    return classifier.classify(extract_features(corpus))

corpus = feature_extraction()
word_features = get_word_features(get_words_in_corpus(corpus))
training_set = nltk.classify.apply_features(extract_features, corpus)
print(corpus_classfication(corpus, training_set))

# print("accuracy: ", nltk.classify.accuracy(classifier, testing_se) * 100)