import nltk
import spacy
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

# nlp = spacy.load("en")
# vectorCount = nlp(documentString)

sentence_delimeter = u"@s"

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
        if sentdelim_regex_compiled.match(string=token) is None:
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
    corpus = corpus.lower()
    corpus = expand_contractions(corpus)
    tokens, sentence_n = tokenizer(corpus)
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



normalize_corpus(documentString)

