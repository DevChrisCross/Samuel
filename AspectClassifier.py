import Normalize
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import numpy
import scipy
import pandas
import sklearn
import nltk
from collections import Counter
from io import StringIO
from nltk.corpus import wordnet
import summarizer_v2


def get_nouns(corpus):
    nouns = [token for token, pos in nltk.pos_tag(nltk.word_tokenize(corpus)) if pos.startswith('N')]
    return nouns


def get_aspect(nouns):
    counts = Counter(nouns)
    return counts.most_common(1)


def get_classifier(aspect):
    search_item = StringIO()
    search_item.write(aspect)
    search_item.write('.n.01')
    definition = wordnet.synset(search_item.getvalue()).definition()
    definition_nouns = get_nouns(definition)
    definition_nouns.append(aspect)
    return definition_nouns


def intersect_lists(from_aspect, from_corpus):
    intersection = [itm for itm in from_aspect if itm in from_corpus]
    return intersection


corpus = [("hotel is luxurious, eloquent, beautifull designed and everyone makes you feel like Royalty", 'pos'),
          ("The rooms are nicely and spacious", "pos"),
          ("The service is great, it should set the standard for all of the rest hotel", "pos"),
          ("The bathroom stinks","neg"),
          ("The food is awful","neg")
          ]

sent_array = [corpus[i][0] for i in range(len(corpus))]
# corpus = Normalize.normalize_corpus(sent_array)
corpus = summarizer_v2.normalize_text(sent_array, tokenize_sentence=False)
# print(corpus["normalized"])
nouns = get_nouns("".join(corpus["raw"]))
aspect = get_aspect(nouns)
classifier = get_classifier(aspect[0][0])
print(classifier)
classifier = intersect_lists(classifier, nouns)

print(classifier)