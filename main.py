import nltk
# import spacy
import re
import string
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim
import numpy
import scipy
import pandas
import sklearn
from Normalize import Normalize

# nlp = spacy.load("en")
# vectorCount = nlp(documentString)

sentence_delimeter = u"@s"

def clean_tags(corpus):
    tags = []
    tags_index = []
    open_tag = -1
    close_tag = -1

    for i in range(0, len(corpus)):
        if corpus[i] == "<":
            open_tag = i

        if corpus[i] == ">":
            close_tag = i + 1

        if open_tag > -1 and close_tag > -1:
            tags_index.append((open_tag, close_tag))
            open_tag = -1
            close_tag = -1

    for index in tags_index:
        tags.append(corpus[index[0]:index[1]])

    for tag in tags:
        corpus = corpus.replace(tag, "")

    return corpus

def tokenizer(corpus):
    sentences = nltk.sent_tokenize(corpus)
    sentence_iter = []
    sentence_start = 0
    sentence_end = 0
    word_ctr = 0
    words = []

    for sentence in sentences:
        extracted_words = nltk.word_tokenize(sentence)
        sentence_start = word_ctr
        sentence_end = word_ctr + len(extracted_words) - 1
        word_ctr += len(extracted_words)
        sentence_iter.append((sentence_start, sentence_end))

        # sentence_ctr += 1
        # words.append(sentence_delimeter + str(sentence_ctr))

        for word in extracted_words:
            words.append(word)

    return words, sentence_iter

def remove_special_characters(tokens):
    sc_regex_string = "[{}]".format(re.escape(string.punctuation))
    sc_regex_compiled = re.compile(pattern=sc_regex_string)
    sentdelim_regex_string = r"@s\d*"
    sentdelim_regex_compiled = re.compile(pattern=sentdelim_regex_string)
    filtered_tokens = []

    for token in tokens:
        if sentdelim_regex_compiled.match(string=token) is None and token[:4] != "NOT_":
            token = sc_regex_compiled.sub(string=token, repl="")
        if token == "":
            continue
        filtered_tokens.append(token)

    return filtered_tokens


def expand_contractions(corpus):
    contraction_map = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "gonna": "going to",
        "gotta": "got to",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "'twas": "it was",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when has",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who has",
        "who've": "who have",
        "why's": "why has",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    contraction_regex_string = "({})".format('|'.join(contraction_map.keys()))
    contraction_regex_compiled = re.compile(pattern=contraction_regex_string, flags=re.IGNORECASE | re.DOTALL)

    def expand_corpus(contraction):
        contraction_match = contraction.group(0).lower()
        expanded_contraction = contraction_map.get(contraction_match)
        return expanded_contraction

    expanded_corpus = contraction_regex_compiled.sub(string=corpus, repl=expand_corpus)
    return expanded_corpus


def negate_tokens(tokens):
    negative_stop_words = ['no',
                           'not',
                           'ain',
                           'aren',
                           'couldn',
                           'didn',
                           'doesn',
                           'hadn',
                           'hasn',
                           'haven',
                           'isn',
                           'mightn',
                           'mustn',
                           'needn',
                           'shan',
                           'shouldn',
                           'wasn',
                           'wouldn']
    index = 0

    for token in tokens:
        if token in negative_stop_words:
            tokens[index+1] = "NOT_"+tokens[index+1]
        index = index+1

    return tokens

def remove_stopwords(tokens):
    stopwords_en = nltk.corpus.stopwords.words('english')
    filtered_tokens = []

    for token in tokens:
        if token not in stopwords_en:
            filtered_tokens.append(token)

    return filtered_tokens

def lemmatize(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    annotated_tokens = nltk.pos_tag(tokens, tagset="universal")
    lemmatized_tokens = []

    token_ctr = 0
    for token in tokens:
        pos_tag = annotated_tokens[token_ctr][1]
        if pos_tag == "ADJ":
            pos_tag = wordnet.ADJ
        elif pos_tag == "VERB":
            pos_tag = wordnet.VERB
        elif pos_tag == "NOUN":
            pos_tag = wordnet.NOUN
        elif pos_tag == "ADV":
            pos_tag = wordnet.ADV
        else:
            pos_tag = None
        lemmatized_token = wordnet_lemmatizer.lemmatize(token, pos_tag)
        lemmatized_tokens.append(lemmatized_token)

    return lemmatized_tokens

def normalize_corpus(corpus):
    normalized_corpus = []

    # corpus = corpus_case(corpus)
    corpus = clean_tags(corpus)
    corpus = corpus.lower()
    corpus = expand_contractions(corpus)
    tokens, sentence_n = tokenizer(corpus)
    tokens = negate_tokens(tokens)
    tokens = remove_stopwords(tokens)
    tokens = remove_special_characters(tokens)
    tokens = lemmatize(tokens)

    print(corpus)
    print(tokens)
    print(sentence_n)
    # print(nltk.corpus.stopwords.words('english'))
    return normalized_corpus


documentString = u"Let's have some fun! You should personally try it. " \
                 u"'Twas night before christmas. " \
                 u"Daren't do it around 9 o'clock. " \
                 u"Dog is the man's bestfriend. Ain't me."

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



normalize_corpus(documentString)

normalize = Normalize(sample)
corpus, tokens, sentence_n = normalize.get_normalized_corpus()
print(corpus)
print(tokens)
print(sentence_n)

