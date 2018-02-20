import re
import enchant
import multiprocessing as mp
from string import punctuation
from textwrap import indent
from typing import Set, List, Union, Optional, Tuple
from warnings import filterwarnings
from enum import Enum
from nltk import sent_tokenize
from functools import partial
from nltk.corpus import stopwords
from numpy import ceil
from spacy import load, tokens
from samuel.constants.taggers import *
from html import unescape
from nltk.corpus import wordnet
from samuel.translator import TextTranslator, Language
filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Property(Enum):
    Letter_Case = "preserve_lettercase"
    Stop_Word = "preserve_stopword",
    Spelling = "correct_spelling",
    Special_Char = "preserve_special_char"


class TextNormalizer:
    _spacy_loader = load("en_core_web_lg")
    _enchant_dict = enchant.Dict("en_US")
    _repeat_regex = re.compile(pattern=r"(\w*)(\w)\2(\w*)")
    _stop_words = stopwords.words("english")
    _enchant = enchant.Dict("en_US")
    _sc_regex = re.compile("[{}0-9]".format(re.escape(punctuation)))

    def __init__(self, text: str, enable: Set[Property] = None,
                 pos_filters: Set[str] = None, punct_filters: Set[str] = None,
                 min_word_length: int = 2, norm_threshold: int = 2, spell_threshold: int = 4,
                 query: str = None, query_similarity_threshold: float = 0.5,
                 translate: bool = True):
        self._id = id(self)
        self._name = self.__class__.__name__

        print(self._name, self._id, "Setting up requirements")
        if pos_filters is None:
            pos_filters = set(TextNormalizer.POS_FILTER)
        if enable:
            if Property.Special_Char in enable:
                pos_filters.update(POS_UNIVERSAL["other"])
                if punct_filters is None:
                    punct_filters = set(TextNormalizer.PUNCT_FILTER)
            if Property.Stop_Word in enable:
                pos_filters.update(POS_UNIVERSAL["open_class"])
                pos_filters.update(POS_UNIVERSAL["closed_class"])
        else:
            punct_filters = set()
            enable = set()

        print(self._name, self._id, "Normalizing text")
        document = TextNormalizer._spacy_loader(unescape(text))
        self._raw_sents = list()
        self._sentences = list()
        self._tokens = list()

        def is_not_essential(token: tokens.Token) -> bool:
            return (token.is_space or token.is_digit
                    or token.is_left_punct or token.is_right_punct
                    or token.is_quote or token.is_bracket
                    or token.like_email or token.like_num or token.like_url)

        def contain_special_char(token: tokens.Token) -> Optional[bool]:
            return self._sc_regex.search(token.text)

        def filtered_tokens(span: tokens.Span) -> str:
            for token in span:
                base_word = token.lemma_
                if (not base_word
                        or contain_special_char(token)
                        or len(token.text) < min_word_length
                        or is_not_essential(token)
                        or (Property.Special_Char not in enable and token.text in punctuation)
                        or (token.text in punctuation and token.text not in punct_filters)
                        or (Property.Stop_Word not in enable
                            and (token.is_stop
                                 or token.norm_ in TextNormalizer._stop_words
                                 or base_word in TextNormalizer._stop_words))
                        or token.pos_ not in pos_filters
                        or (not self._enchant.check(token.text) and not token.pos_ == "PROPN")):
                    continue
                if (token.is_stop
                        or token.norm_ in TextNormalizer._stop_words
                        or base_word in TextNormalizer._stop_words):
                    base_word = token.text
                else:
                    if (Property.Spelling in enable
                            and len(base_word) > spell_threshold
                            and token.pos_ in ["ADJ", "ADV", "NOUN", "VERB"]):
                        base_word = TextNormalizer.__correct_word(base_word)
                if Property.Letter_Case in enable:
                    if token.is_upper:
                        base_word = base_word.upper()
                    if token.is_title:
                        base_word = base_word.capitalize()
                else:
                    base_word = base_word.lower()
                yield base_word

        print(self._name, self._id, "Filtering tokens and sentences")
        if query:
            # TODO: word network
            query = TextNormalizer._spacy_loader(unescape(query))
            for token in query:
                if token.dep_ == "nsubj":
                    query_subject = token.text
                    break

        translator = TextTranslator()
        for sentence in document.sents:
            if translate and not translator.is_language(unescape(sentence.text), Language.ENGLISH):
                translated_text = translator.translate_to(unescape(sentence.text))
                if translated_text:
                    sentence = list(TextNormalizer._spacy_loader(unescape(translated_text)).sents)[0]
                else:
                    continue

            if query and query.similarity(sentence) < query_similarity_threshold:
                continue

            accepted_tokens = list(filtered_tokens(sentence))
            if accepted_tokens and len(accepted_tokens) > norm_threshold:
                self._raw_sents.append(" ".join(sentence.text.split()))
                self._tokens.extend(accepted_tokens)
                self._sentences.append(accepted_tokens)
        print(self._name, self._id, "Text normalization done")

    @staticmethod
    def __correct_word(word: str) -> str:
        """
        Corrects the word by removing irregular repeated letters, then suggests possible words intended to be used
        using the PyEnchant library. You can check it at http://pythonhosted.org/pyenchant/api/enchant.html
        """
        if TextNormalizer._enchant_dict.check(word):
            return word

        def __check_word(old_word: str) -> str:
            if wordnet.synsets(old_word):
                return old_word
            else:
                new_word = TextNormalizer._repeat_regex.sub(string=old_word, repl=r"\1\2\3")
                new_word = new_word if new_word == old_word else __check_word(new_word)
                return new_word

        initial_correct_word = __check_word(word)
        is_word_correct = TextNormalizer._enchant_dict.check(initial_correct_word)

        if is_word_correct:
            return initial_correct_word
        else:
            word_suggestions = TextNormalizer._enchant_dict.suggest(initial_correct_word)
            final_correct_word = word_suggestions[0] if word_suggestions else initial_correct_word
            return final_correct_word

    @property
    def raw_sents(self):
        return self._raw_sents

    @property
    def sentences(self):
        return self._sentences

    @property
    def tokens(self):
        return self._tokens

    def __str__(self):
        return "\n".join([
            "Normalized Sentences",
            indent("\n".join([str(array) for array in self._sentences]), "\t"),
            "Tokens",
            indent(str(self._tokens), "\t")
        ])

    PUNCT_FILTER = ["?", "!"]

    POS_FILTER = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]


class NormalizerManager:
    def __init__(self, documents: Union[List[str], str], enable: Set[Property] = None,
                 pos_filters: Set[str] = None, punct_filters: Set[str] = None,
                 min_word_length: int = 1, batch_size: int = 200,
                 norm_threshold: int = 2, spell_threshold: int = 4,
                 query: str = None, query_similarity_threshold: float = 0.7
                 ):
        self._id = id(self)
        self._name = self.__class__.__name__

        self._raw_sents = list()
        self._sentences = list()
        self._tokens = list()

        if isinstance(documents, str):
            print(self._name, self._id, "Preparing document batches")
            self._partitions = NormalizerManager.partitioned_docs(documents, batch_size)
            documents = self._partitions

        print(len(self._partitions))
        print(self._name, self._id, "Preparing process pool")
        pool = mp.Pool()
        print(self._name, self._id, "Mapping document batches")
        result = pool.map_async(partial(TextNormalizer, enable=enable, pos_filters=pos_filters,
                                        punct_filters=punct_filters, min_word_length=min_word_length,
                                        norm_threshold=norm_threshold, query=query,
                                        query_similarity_threshold=query_similarity_threshold), documents)
        for tn in result.get():
            self._raw_sents.extend(tn.raw_sents)
            self._sentences.extend(tn.sentences)
            self._tokens.extend(tn.tokens)

        pool.close()
        pool.join()
        print(self._name, self._id, "Normalization pooling done")

    @staticmethod
    def partitioned_docs(documents: str, batch_size: int = 200) -> List[str]:
        sentences = list(sent_tokenize(documents))
        batch_count = int(ceil(len(sentences) / batch_size))

        def _partition() -> str:
            start_divider = 0
            for i in range(batch_count):
                start_divider = start_divider
                end_divider = start_divider + batch_size
                if i == batch_count - 1:
                    end_divider = len(sentences)
                yield " ".join(sentences[start_divider:end_divider])
                start_divider += batch_size

        return list(_partition())

    @property
    def raw_sents(self):
        return self._raw_sents

    @property
    def sentences(self):
        return self._sentences

    @property
    def tokens(self):
        return self._tokens

    @property
    def partitions(self):
        return self._partitions

    def __str__(self):
        return "\n".join([
            "Normalized Sentences",
            indent("\n".join([str(array) for array in self._sentences]), "\t"),
            "Tokens",
            indent(str(self._tokens), "\t")
        ])


if __name__ == "__main__":
    from samuel.test.test_document import single_test_document, document1, document3
    print(TextNormalizer("This is sentence is so pangit", norm_threshold=0))
    # cProfile.run("TextNormalizer(document, enable={Property.Spelling})", "Text_Normalizer")
    # tn_profiler = pstats.Stats("Text_Normalizer")
    # tn_profiler.strip_dirs().sort_stats("cumulative").print_stats(10)
    # tn_profiler.sort_stats('time').print_stats(10)

    # NormalizerManager(document, batch_count=16)
    # cProfile.run("NormalizerManager(document, enable={Property.Spelling}, batch_count=16)", "Text_Normalizer_MP")
    # tn_mp_profiler = pstats.Stats("Text_Normalizer_MP")
    # tn_mp_profiler.strip_dirs().sort_stats("cumulative").print_stats(10)
    # tn_mp_profiler.sort_stats('time').print_stats(10)
    pass
