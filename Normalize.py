import nltk
import re
import string
import enchant
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pprint import pprint
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def normalize_text(text, contraction_map=None, tokenize_sentence=True, correct_spelling=False):
    """A function that performs text normalization.

    Tokenizes sentences and words, removes stopwords and special characters
     performs lemmatization on words, and further remove words which does not
     qualify in the part-of-speech tag map.

    :param text: string. the text to be normalized
    :param contraction_map: dictionary. supplying a contraction map automatically expands the contractions
        present in the text
    :param tokenize_sentence: boolean. if the text input should be tokenize into sentences
        It should be set to false if the text input is an array of sentences
    :param correct_spelling: boolean. if the spelling of words should be checked and corrected
        The module is too aggressive and greatly decreases performance, use at your own risk
    :return: {normalized, raw} dictionary. contains the normalized and raw sentences
    """

    def __clean_tags(text):
        """Removes the html tags present in the text

        :param text: the text to be cleaned
        :return: the text without any html tags
        """

        tags_index = list()
        open_tag = close_tag = -1

        for i in range(len(text)):
            if text[i] == "<":
                open_tag = i

            if text[i] == ">":
                close_tag = i + 1

            if open_tag > -1 and close_tag > -1:
                tags_index.append((open_tag, close_tag))
                open_tag = close_tag = -1

        tags = [text[index[0]:index[1]] for index in tags_index]
        for tag in tags:
            text = text.replace(tag, "")

        return text

    def __correct_word(word):
        """Corrects the misspelled word

        Corrects the word by removing irregular repeated letters,
        and further corrects it by suggesting possible intended words to be used
        using the PyEnchant library. You can check it at http://pythonhosted.org/pyenchant/api/enchant.html

        :param word: string. the word to be corrected
        :return: string. the corrected word
        """

        def __check_word(old_word):
            if wordnet.synsets(old_word):
                return old_word
            else:
                new_word = repeat_regex_compiled.sub(string=old_word, repl=match_substitution)
                new_word = new_word if new_word == old_word else __check_word(new_word)
                return new_word

        enchant_dict = enchant.Dict("en_US")
        match_substitution = r'\1\2\3'
        repeat_regex_string = r'(\w*)(\w)\2(\w*)'
        repeat_regex_compiled = re.compile(pattern=repeat_regex_string)

        initial_correct_word = __check_word(word)
        word_suggestions = enchant_dict.suggest(initial_correct_word)
        is_word_correct = enchant_dict.check(initial_correct_word)

        if is_word_correct:
            return initial_correct_word
        else:
            final_correct_word = word_suggestions[0] if word_suggestions else initial_correct_word
            return final_correct_word

    def __expand_contractions(text, contraction_map):
        """Expands the contractions present in the text

        :param text: the text to be expanded
        :param contraction_map: the dictionary to be used to expand the contractions
        :return: the expanded contractions form of the text
        """

        contraction_regex_string = "({})".format('|'.join(contraction_map.keys()))
        contraction_regex_compiled = re.compile(pattern=contraction_regex_string, flags=re.IGNORECASE | re.DOTALL)

        def expand_corpus(contraction):
            contraction_match = contraction.group(0).lower()
            expanded_contraction = contraction_map.get(contraction_match)
            return expanded_contraction

        expanded_corpus = contraction_regex_compiled.sub(string=text, repl=expand_corpus)
        return expanded_corpus


    clean_text = __clean_tags(text) if tokenize_sentence else text
    raw_sentences = nltk.sent_tokenize(clean_text) if tokenize_sentence else clean_text
    raw_sentences = [__expand_contractions(sentence, contraction_map)
                     for sentence in raw_sentences] if contraction_map else raw_sentences
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in raw_sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences, tagset="universal")

    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_en.extend(["n't", "'s", "'d", "'t", "'ve", "'ll"])

    sc_regex_string = "[{}]".format(re.escape(string.punctuation))
    sc_regex_compiled = re.compile(pattern=sc_regex_string)

    wordnet_lemmatizer = WordNetLemmatizer()
    word = 0
    tag = 1
    pos_tag_map = {
        "ADJ": wordnet.ADJ,
        "VERB": wordnet.VERB,
        "NOUN": wordnet.NOUN,
        "ADV": wordnet.ADV
    }

    normalized_sentences = list()
    for sentence in tagged_sentences:
        new_sentence = list()
        for tagged_word in sentence:
            if tagged_word[word] not in stopwords_en:
                is_special_character = sc_regex_compiled.sub(string=tagged_word[word], repl="") == ""
                tagged_word = None if is_special_character else tagged_word
                if tagged_word and tagged_word[tag] in pos_tag_map:
                    wordnet_tag = pos_tag_map[tagged_word[tag]]
                    lemmatized_word = wordnet_lemmatizer.lemmatize(tagged_word[word], wordnet_tag)
                    lemmatized_word = __correct_word(lemmatized_word) if correct_spelling else lemmatized_word
                    new_sentence.append(lemmatized_word.lower())
        normalized_sentences.append(new_sentence)

    return {
        "normalized": normalized_sentences,
        "raw": raw_sentences
    }

CONTRACTION_MAP = {
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
NEGATIVE_STOPWORDS = ['no', 'not', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
                    'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'wouldn']
