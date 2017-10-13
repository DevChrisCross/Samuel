import nltk
# import spacy
import re
import string
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import numpy
import scipy
import pandas
import sklearn
import AspectClassifier

#####################################



# nouns = get_nouns(corpus)
# aspect = get_aspect(nouns)
# classifier = get_classifier(aspect[0][0])
# classifier = intersect_lists(classifier, nouns)

    # print(corpus)
    # print(tokens)
    # print(classifier)
    # print(sentence_n)
    # # print(nltk.corpus.stopwords.words('english'))
    # return normalized_corpus


# documentString = u"Let's have some fun! You should personally try it. " \
#                  u"'Twas night before christmas. " \
#                  u"Daren't do it around 9 o'clock. " \
#                  u"Dog is the man's bestfriend. Ain't me."

documentString = u"Extraordinary hotel. " \
                 u"This hotel has good services, bad dogs, super qualified staffs. " \
                 u"hotel hotel hotel."

kingsman = u"Another witty and fun, action film by Matthew Vaughn. " \
           u"Taron Egerton and Colin Firth come together once more to recreate their fantastic spy duo. " \
           u"Though the movie lacks in parts of what it promised in the trailers and details of what many fans may have hoped. " \
           u"It begrudgingly makes up for though its eye-popping visuals, " \
           u"great new characters, and a clear believable storyline of Harry Hart's revival. Kingsman: " \
           u"The Golden Circle is a near pitch perfect sequel to relive the entertainment of Kingsman: The Secret Service."

sample = u"<p class='alert alert-info'>The food is <span> great </span> but the service is not good.</p>"

book_review = u"I'm going to keep this brief since there isn't much to say that hasn't already been said. *clears throat* " \
              u"I think the reason I waited so long to read this series is because I just couldn't imagine myself enjoying " \
              u"reading about an eleven-year-old boy and his adventures at a school of wizardry. I thought it would be too " \
              u"juvenile for my taste. I was wrong, of course." \
              u"I can honestly say that I loved every minute of this. It's a spectacular little romp with funny, courageous, " \
              u"and endearing characters that you can't help but love." \
              u"It has talking chess pieces, singing hats, a giant three-headed dog named Fluffy, a hilarious giant with a " \
              u"dragon fetish, a master wizard that's just a little bit crazy, mail carrier owls, goblins running a bank, " \
              u"unicorns, centaurs(!), trolls . . . and probably much more that I'm forgetting. " \
              u"And then there's the lead characters: Hermione, the young scholar who starts out prim and up-tight but soon " \
              u"becomes a true friend; Ron, the boy who has little money but who has an abundance of family and loyalty to his " \
              u"friends to make up for it; and then there's Harry, the boy who starts out sleeping in a closet and ends up being " \
              u"a hero. Harry is kind to those that deserve it, fearless when it counts the most, and wonderfully intelligent." \
              u"What's not to love? "

normalize_corpus(kingsman)

# corpus, tokens, sentence_n = Normalize.normalize_corpus(sample)
# print(corpus)
# print(tokens)
# print(sentence_n)

