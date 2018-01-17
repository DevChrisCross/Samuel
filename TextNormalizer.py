import re
import string
import warnings
import enchant
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import product
from typing import Dict, Type
from textwrap import indent

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class TextNormalizer:
    def __init__(self, text: str, settings: "TextNormalizer.Settings" = None):
        self._text = text
        self._normalized = self._tokens = self._raw = None
        self._settings = settings if settings else TextNormalizer.Settings()

    def __call__(self, *args, **kwargs) -> "TextNormalizer":
        """
        Tokenizes sentences and words, then removes present stopwords and special characters, then performs
        lemmatization and further remove words which does not qualify in the part-of-speech tag map.
        """

        settings = self.settings
        clean_text = TextNormalizer.__remove_html_tags(self._text)
        raw_sentences = nltk.sent_tokenize(clean_text)
        if settings.expand_word_contraction:
            raw_sentences = [TextNormalizer.__expand_word_contractions(sentence, settings.contraction_map)
                             for sentence in raw_sentences]

        stopwords_en = nltk.corpus.stopwords.words('english')
        stopwords_en.extend(["n't", "'s", "'d", "'t", "'ve", "'ll"])

        lemmatizer = WordNetLemmatizer()
        tokenized_sentences = [sentence.split()
                               if settings.preserve_punctuation_emphasis or settings.preserve_special_character
                               else nltk.word_tokenize(sentence) for sentence in raw_sentences]

        pos_tagged_sentences = nltk.pos_tag_sents(tokenized_sentences, tagset="universal")

        def __is_special_character(word: str):
            return TextNormalizer.SPECIAL_CHARACTER_REGEX.sub(string=word, repl="") == ""

        normalized_sentences = list()
        for sentence in pos_tagged_sentences:
            new_sentence = list()
            for pos_tagged_word in sentence:
                word = pos_tagged_word[0]
                if __is_special_character(word):
                    if not settings.preserve_special_character:
                        continue
                    new_word = word
                else:
                    if len(word) < settings.minimum_word_length:
                        continue
                    new_word = word = TextNormalizer.__clean_left_surrounding_text(word)
                    if word in stopwords_en:
                        if not settings.preserve_stopword:
                            continue
                        new_word = word
                    else:
                        if settings.preserve_punctuation_emphasis:
                            TextNormalizer.__reconstruct_regex(settings.punctuation_emphasis_list)
                            filtered_word = TextNormalizer.__set_punctuation_emphasis(
                                word, settings.punctuation_emphasis_level)
                            new_word = filtered_word if filtered_word else word
                        pos_tag = pos_tagged_word[1]
                        if settings.enable_pos_tag_filter:
                            if pos_tag not in settings.pos_tag_map:
                                continue
                            wordnet_tag = settings.pos_tag_map[pos_tag]
                            new_word = word if settings.preserve_wordform else lemmatizer.lemmatize(word, wordnet_tag)
                            new_word = (TextNormalizer.__correct_word(new_word) if settings.correct_spelling
                                        else new_word)
                    new_word = new_word if settings.preserve_lettercase else new_word.lower()
                new_sentence.append(new_word)
            normalized_sentences.append(new_sentence)

        if settings.request_tokens:
            tokens = list()
            for sentence in normalized_sentences:
                for word in sentence:
                    tokens.append(word)
            self._tokens = tokens

        self._normalized = normalized_sentences
        self._raw = raw_sentences
        return self

    def __str__(self):
        return ("\n" + "-"*200
                + "\nNormalized Text:\n" + indent("\n".join([str(array) for array in self._normalized]), "\t")
                + "\n\nRaw Text:\n" + indent("\n".join(self._raw), "\t")
                + "\n\nTokens: " + str(self._tokens)
                + "\n" + ":"*200 + "\n[Settings]\n" + str(self._settings)
                + "\n" + "-"*200 + "\n")

    def append(self, text: str) -> "TextNormalizer":
        tn = TextNormalizer(text, self._settings)
        normalized = tn()
        self._text = " ".join([self._text, normalized.original_text])
        self._raw.extend(normalized.raw_text)
        self._normalized.extend(normalized.normalized_text)
        if self._tokens:
            self._tokens.extend(normalized.extracted_tokens)
        return self

    @classmethod
    def __reconstruct_regex(cls, emphasis_list: str) -> type(None):
        cls.PUNCTUATION_REGEX_STRING = "[{}]".format('|'.join(re.escape(emphasis_list)))
        cls.WORD_PUNCTUATION_REGEX_STRING = "(\w+[-|'|.])*\w+" + cls.PUNCTUATION_REGEX_STRING + "+"
        cls.PUNCTUATION_REGEX = re.compile(pattern=cls.PUNCTUATION_REGEX_STRING + "+")
        cls.WORD_PUNCTUATION_REGEX = re.compile(pattern=cls.WORD_PUNCTUATION_REGEX_STRING)

    @classmethod
    def __set_punctuation_emphasis(cls, word: str, emphasis_level: int) -> str:
        if cls.WORD_PUNCTUATION_REGEX.fullmatch(string=word):
            after_word_divider = None
            for match in re.finditer(pattern="\w+", string=word):
                after_word_divider = match.span()[1]
            word_part = word[:after_word_divider]
            punctuation_part = word[after_word_divider:]
            return word if len(punctuation_part) >= emphasis_level else word_part

        new_word = word
        for match in re.finditer(pattern="\w+", string=word):
            new_word = word[:match.span()[1]]
        return new_word

    @staticmethod
    def __remove_html_tags(text: str) -> str:
        opening_html_tag = "<"
        closing_html_tag = ">"
        html_tags_index = list()
        open_tag_index = close_tag_index = None

        for i in range(len(text)):
            character = text[i]
            if character == opening_html_tag:
                open_tag_index = i
            if character == closing_html_tag:
                close_tag_index = i + 1
            if open_tag_index and close_tag_index:
                html_tags_index.append((open_tag_index, close_tag_index))
                open_tag_index = close_tag_index = None

        html_tags = [text[index[0]:index[1]] for index in html_tags_index]
        for tag in html_tags:
            text = text.replace(tag, "")
        return text

    @staticmethod
    def __expand_word_contractions(text: str, contraction_map: Dict[str, str]) -> str:
        contraction_regex_string = "({})".format('|'.join(contraction_map.keys()))
        contraction_regex_compiled = re.compile(pattern=contraction_regex_string, flags=re.IGNORECASE | re.DOTALL)

        def expand_corpus(contraction: re) -> str:
            contraction_match = contraction.group(0).lower()
            expanded_contraction = contraction_map.get(contraction_match)
            return expanded_contraction

        expanded_corpus = contraction_regex_compiled.sub(string=text, repl=expand_corpus)
        return expanded_corpus

    @staticmethod
    def __clean_left_surrounding_text(word: str) -> str:
        for match in re.finditer(pattern="\w", string=word):
            first_letter = match.span()[0]
            return word[first_letter:]

    @staticmethod
    def __correct_word(word: str) -> str:
        """
        Corrects the word by removing irregular repeated letters, then suggests possible words intended to be used
        using the PyEnchant library. You can check it at http://pythonhosted.org/pyenchant/api/enchant.html
        """

        enchant_dict = enchant.Dict("en_US")
        match_substitution = r'\1\2\3'
        repeat_regex_string = r'(\w*)(\w)\2(\w*)'
        repeat_regex_compiled = re.compile(pattern=repeat_regex_string)

        def __check_word(old_word: str) -> str:
            if wordnet.synsets(old_word):
                return old_word
            else:
                new_word = repeat_regex_compiled.sub(string=old_word, repl=match_substitution)
                new_word = new_word if new_word == old_word else __check_word(new_word)
                return new_word

        initial_correct_word = __check_word(word)
        word_suggestions = enchant_dict.suggest(initial_correct_word)
        is_word_correct = enchant_dict.check(initial_correct_word)

        if is_word_correct:
            return initial_correct_word
        else:
            final_correct_word = word_suggestions[0] if word_suggestions else initial_correct_word
            return final_correct_word

    @property
    def original_text(self):
        return self._text

    @property
    def settings(self):
        return self._settings

    @property
    def normalized_text(self):
        return self._normalized

    @property
    def raw_text(self):
        return self._raw

    @property
    def extracted_tokens(self):
        return self._tokens

    @original_text.setter
    def original_text(self, value: str):
        self._text = value

    @settings.setter
    def settings(self, value: "TextNormalizer.Settings"):
        self._settings = value

    class Settings:
        def __init__(self):
            self._request_tokens = False
            self._preserve_lettercase = False
            self._minimum_word_length = 1

            self._expand_word_contraction = False
            self._contraction_map = TextNormalizer.DEFAULT_CONTRACTION_MAP
            self._preserve_stopword = False

            self._enable_pos_tag_filter = True
            self._pos_tag_map = TextNormalizer.DEFAULT_POS_TAG_MAP
            self._correct_spelling = False
            self._preserve_wordform = False

            self._preserve_special_character = False
            self._preserve_punctuation_emphasis = False
            self._punctuation_emphasis_list = TextNormalizer.DEFAULT_PUNCTUATION_EMPHASIS
            self._punctuation_emphasis_level = 1

        def __str__(self):
            all_properties = list(filter(lambda p: p.startswith("_") and not p.startswith("__"), dir(self)))

            def __filter_property_type(type: Type) -> dict:
                return {property[1:].replace("_", " "): getattr(self, property) for property in all_properties
                        if isinstance(getattr(self, property), type)}

            numeric_properties = {key: value for key, value in __filter_property_type(int).items()
                                  if not isinstance(value, bool)}
            toggled_properties = __filter_property_type(bool)
            enabled_properties = dict()
            disabled_properties = dict()
            for key, value in toggled_properties.items():
                if value:
                    enabled_properties.update({key: value})
                else:
                    disabled_properties.update({key: value})
            maps_properties = __filter_property_type(str)
            maps_properties.update(__filter_property_type(dict))

            return ("Numeric Properties: " + str(numeric_properties)
                    + "\nEnabled Properties: " + str(enabled_properties)
                    + "\nDisabled Properties: " + str(disabled_properties)
                    + "\nProperty Maps: " + str(maps_properties))

        def set_independent_properties(
                self, minimum_word_length: int, request_tokens: bool = False,
                preserve_lettercase: bool = False) -> "TextNormalizer.Settings":

            self._request_tokens = request_tokens
            self._preserve_lettercase = preserve_lettercase
            self._minimum_word_length = minimum_word_length
            return self

        def set_word_contraction_properties(
                self, expand_contraction: bool = True, preserve_stopword: bool = True,
                contraction_map: Dict[str, str] = None) -> "TextNormalizer.Settings":

            self._preserve_stopword = preserve_stopword
            if preserve_stopword:
                self._expand_word_contraction = expand_contraction
                if contraction_map:
                    self._contraction_map = contraction_map
            else:
                self._expand_word_contraction = False
            return self

        def set_pos_tag_properties(
                self, preserve_wordform: bool = False, correct_spelling: bool = False,
                enable_pos_tag_filter: bool = True, pos_tag_map: Dict[str, str] = None) -> "TextNormalizer.Settings":

            self._enable_pos_tag_filter = enable_pos_tag_filter
            if enable_pos_tag_filter:
                self.set_special_character_properties(preserve_punctuation_emphasis=False)
                self._preserve_wordform = preserve_wordform
                self._correct_spelling = correct_spelling
                if pos_tag_map:
                    self._pos_tag_map = pos_tag_map
            else:
                self._preserve_wordform = False
                self._correct_spelling = False
            return self

        def set_special_character_properties(
                self, preserve_special_character: bool = True, preserve_punctuation_emphasis: bool = True,
                punctuation_emphasis_level: int = 1, punctuation_emphasis_list: str = None) -> "TextNormalizer.Settings":

            self._preserve_special_character = preserve_special_character
            if preserve_punctuation_emphasis:
                self.set_pos_tag_properties(enable_pos_tag_filter=False)
                self._preserve_punctuation_emphasis = preserve_punctuation_emphasis
                self._punctuation_emphasis_level = punctuation_emphasis_level
                if punctuation_emphasis_list:
                    self._punctuation_emphasis_list = punctuation_emphasis_list
            return self

        @property
        def request_tokens(self):
            return self._request_tokens

        @property
        def preserve_lettercase(self):
            return self._preserve_lettercase

        @property
        def minimum_word_length(self):
            return self._minimum_word_length

        @property
        def expand_word_contraction(self):
            return self._expand_word_contraction

        @property
        def contraction_map(self):
            return self._contraction_map

        @property
        def preserve_stopword(self):
            return self._preserve_stopword

        @property
        def enable_pos_tag_filter(self):
            return self._enable_pos_tag_filter

        @property
        def pos_tag_map(self):
            return self._pos_tag_map

        @property
        def correct_spelling(self):
            return self._correct_spelling

        @property
        def preserve_wordform(self):
            return self._preserve_wordform

        @property
        def preserve_special_character(self):
            return self._preserve_special_character

        @property
        def preserve_punctuation_emphasis(self):
            return self._preserve_punctuation_emphasis

        @property
        def punctuation_emphasis_list(self):
            return self._punctuation_emphasis_list

        @property
        def punctuation_emphasis_level(self):
            return self._punctuation_emphasis_level

    DEFAULT_PUNCTUATION_EMPHASIS = "?!"

    PUNCTUATION_REGEX_STRING = "[{}]".format('|'.join(re.escape(DEFAULT_PUNCTUATION_EMPHASIS)))

    SPECIAL_CHARACTER_REGEX_STRING = "[{}]".format(re.escape(string.punctuation))

    WORD_PUNCTUATION_REGEX_STRING = "(\w+[-|'|.])*\w+" + PUNCTUATION_REGEX_STRING + "+"

    SPECIAL_CHARACTER_REGEX = re.compile(pattern=SPECIAL_CHARACTER_REGEX_STRING)

    PUNCTUATION_REGEX = re.compile(pattern=PUNCTUATION_REGEX_STRING + "+")

    WORD_PUNCTUATION_REGEX = re.compile(pattern=WORD_PUNCTUATION_REGEX_STRING)

    NEGATIVE_STOPWORDS = ['no', 'not', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
                          'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'wouldn']

    DEFAULT_POS_TAG_MAP = {
        "ADJ": wordnet.ADJ,
        "VERB": wordnet.VERB,
        "NOUN": wordnet.NOUN,
        "ADV": wordnet.ADV
    }

    DEFAULT_CONTRACTION_MAP = {
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


class SentiText:
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text.encode('utf-8'))
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = SentiText.allcap_differential(self.words_and_emoticons)

    @staticmethod
    def allcap_differential(words):
        """
        Check whether just some words in the input are ALL CAPS

        :param list words: The words to inspect
        :returns: `True` if some but not all items in `words` are ALL CAPS
        """
        is_different = False
        allcap_words = 0
        for word in words:
            if word.isupper():
                allcap_words += 1
        cap_differential = len(words) - allcap_words
        if 0 < cap_differential < len(words):
            is_different = True
        return is_different

    def _words_and_emoticons(self):
        """
        Removes leading and trailing punctuations
        Leaves contractions and most emoticons
        Does not preserve punctuation-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        words_punc_dict = self._words_plus_punc(self.text)
        wes = [we for we in wes if len(we) > 1]
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes

    def _words_plus_punc(self, text: str):
        """
        Returns mapping of form:
        {
            'cat,': 'cat',
            ',cat': 'cat',
        }
        """

        words = SentiText.__remove_punctuations(text)
        words = set(word for word in words if len(word) > 1)
        # the product gives ('cat', ',') and (',', 'cat')
        punc_before = {''.join(p): p[1] for p in product(SentiText.PUNCTUATIONS, words)}
        punc_after = {''.join(p): p[0] for p in product(words, SentiText.PUNCTUATIONS)}
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)
        return words_punc_dict

    @staticmethod
    def __remove_punctuations(text):
        punctuation_regex_string = '[{}]'.format(re.escape(string.punctuation))
        punctuation_regex_compiled = re.compile(pattern=punctuation_regex_string)
        clean_text = punctuation_regex_compiled.sub(repl='', string=text)
        return clean_text.split()

    PUNCTUATIONS = [".", "!", "?", ",", ";", ":", "-", "'", "\"", "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]

# deprecated version
# print(TextNormalizer.create_normalizer("Some string FADAD :)").normalize_text(
#     request_tokens=True, preserve_special_character=True, preserve_punctuation_emphasis=True,
#     punctuation_emphasis_list="?!", punctuation_emphasis_level=2, preserve_stopword=True,
#     minimum_word_length=2, enable_pos_tag_filter=True, preserve_lettercase=True, correct_spelling=True))


settings = (TextNormalizer.Settings()
            .set_independent_properties(minimum_word_length=2, request_tokens=True, preserve_lettercase=True)
            .set_special_character_properties(punctuation_emphasis_level=4)
            .set_word_contraction_properties()
            .set_pos_tag_properties(enable_pos_tag_filter=False))
textNormalizer = TextNormalizer("I hope this group of film-makers!!!! never re-unites. ever again. IT SUCKS????  >:(",
                                settings)
print(textNormalizer().append("Here you go! :)"))
