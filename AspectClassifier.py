import nltk
import warnings
from collections import Counter
from io import StringIO
from nltk.corpus import wordnet
import Normalize
import Summarizer

# import gensim
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import numpy
# import scipy
# import pandas
# import sklearn


def get_nouns(tokens_get):
    nouns_send = [token for token, pos in nltk.pos_tag(tokens_get) if pos.startswith('N')]
    return nouns_send


def get_aspect(nouns_get):
    counts = Counter(nouns_get)
    return counts.most_common(1)


def get_classifier(aspect_get):
    search_item = StringIO()
    search_item.write(aspect_get)
    search_item.write('.n.01')
    definition = [wordnet.synset(search_item.getvalue()).definition()]
    definition_nouns = Normalize.normalize_text(definition, tokenize_sentence=False)
    classifier_send = definition_nouns['tokens']
    classifier_send.append(aspect_get)
    return classifier_send


def intersect_lists(from_aspect, from_corpus):
    intersection = [itm for itm in from_aspect if itm in from_corpus]
    return intersection


corpus = [("hotel is luxurious, eloquent, beautiful designed and everyone makes you feel like royalty.", 'pos'),
          ("The rooms are nicely and spacious.", "pos"),
          ("The service is great, it should set the standard for all of the rest hotel.", "pos"),
          ("The bathroom stinks.", "neg"),
          ("The food is awful.", "neg")
          ]

sent_array = [corpus[i][0] for i in range(len(corpus))]
corpus = Normalize.normalize_text(sent_array, tokenize_sentence=False)
# print(corpus['tokens'])
nouns = get_nouns(corpus["tokens"])
# print(nouns)
aspect = get_aspect(nouns)
# print(aspect)
classifier = get_classifier(aspect[0][0])
# print(classifier)
classifier = intersect_lists(classifier, nouns)
# print(classifier)
