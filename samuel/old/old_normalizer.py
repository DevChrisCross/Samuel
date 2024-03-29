import re
import string
import warnings
import enchant
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import product
from typing import Dict, Type, List
from textwrap import indent
from samuel.constants.taggers import *
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class TextNormalizer:
    def __init__(self, text: str, settings: "TextNormalizer.Settings" = None):
        warnings.warn('Use TextNormalizer from samuel.normalizer instead.', DeprecationWarning)
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

        def __is_special_character(w: str):
            return TextNormalizer.SPECIAL_CHARACTER_REGEX.sub(string=w, repl="") == ""

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
            self._contraction_map = CONTRACTION_MAP
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

            def __filter_property_type(t: Type) -> dict:
                return {p[1:].replace("_", " "): getattr(self, p) for p in all_properties
                        if isinstance(getattr(self, p), t)}

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
                punctuation_emphasis_level: int = 1,
                punctuation_emphasis_list: str = None) -> "TextNormalizer.Settings":

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

    DEFAULT_POS_TAG_MAP = {
        "ADJ": wordnet.ADJ,
        "VERB": wordnet.VERB,
        "NOUN": wordnet.NOUN,
        "ADV": wordnet.ADV
    }
