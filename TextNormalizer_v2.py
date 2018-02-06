import re
import cProfile
import pstats
import multiprocessing as mp
from string import punctuation
from textwrap import indent
import enchant
from typing import Set, List, Union
from warnings import filterwarnings
from enum import Enum
from os import cpu_count
from functools import partial
from nltk.corpus import stopwords, wordnet
from spacy import load, tokens
from constants.taggers import *
filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Property(Enum):
    Letter_Case = "preserve_lettercase"
    Stop_Word = "preserve_stopword",
    Spelling = "correct_spelling",
    Special_Char = "preserve_special_char"


class TextNormalizer:
    _spacy_loader = load("en")
    _enchant_dict = enchant.Dict("en_US")
    _repeat_regex = re.compile(pattern=r"(\w*)(\w)\2(\w*)")
    _stop_words = stopwords.words("english")

    def __init__(self, text: str, enable: Set[Property] = None,
                 pos_filters: Set[str] = None, punct_filters: Set[str] = None,
                 min_word_length: int = 1):
        print("Setting up requirements: object", id(self))
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

        print("Normalizing text: object", id(self))
        document = TextNormalizer._spacy_loader(text)
        self._raw_sents = list()
        self._sentences = list()
        self._tokens = list()

        def is_not_essential(token: tokens.Token) -> bool:
            return (token.is_space or token.is_digit
                    or token.is_left_punct or token.is_right_punct
                    or token.is_quote or token.is_bracket
                    or token.like_email or token.like_num or token.like_url)

        def filtered_tokens(span: tokens.Span) -> str:
            for token in span:
                base_word = token.lemma_
                if (not base_word
                        or len(token.text) < min_word_length
                        or is_not_essential(token)
                        or (Property.Special_Char not in enable and token.text in punctuation)
                        or (token.text in punctuation and token.text not in punct_filters)
                        or (Property.Stop_Word not in enable
                            and (token.is_stop or token.norm_ in TextNormalizer._stop_words))
                        or token.pos_ not in pos_filters):
                    continue
                if token.is_stop or token.norm_ in TextNormalizer._stop_words:
                    base_word = token.text
                else:
                    if (Property.Spelling in enable
                            and len(base_word) > 4
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

        print("Filtering tokens and sentences: object", id(self))
        for sentence in document.sents:
            self._raw_sents.append(sentence.text.strip())
            accepted_tokens = list(filtered_tokens(sentence))
            if accepted_tokens:
                self._tokens.extend(accepted_tokens)
                self._sentences.append(accepted_tokens)
        print("Text normalization done: object", id(self))

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
                 min_word_length: int = 1, batch_count: int = None):
        self._raw_sents = list()
        self._sentences = list()
        self._tokens = list()

        if isinstance(documents, str):
            print("Preparing document batches: object", (id(self)))
            documents = load("en")(documents)
            sentences = [sentence for sentence in documents.sents]
            batch_count = batch_count if batch_count else cpu_count()
            divider = len(sentences) // batch_count

            def partitioned_docs() -> str:
                start_divider = 0
                for i in range(batch_count):
                    start_divider = start_divider
                    end_divider = start_divider + divider
                    if i == 3:
                        end_divider += len(sentences) % batch_count
                    yield " ".join([sentence.text for sentence in sentences[start_divider:end_divider]])
                    start_divider += divider

            documents = list(partitioned_docs())

        print("Preparing process pool: object", (id(self)))
        pool = mp.Pool()
        print("Mapping document batches: object", (id(self)))
        result = pool.map_async(partial(TextNormalizer, enable=enable, pos_filters=pos_filters,
                                punct_filters=punct_filters, min_word_length=min_word_length), documents)

        if result.get():
            print("Reducing document results: object", (id(self)))
        for tn in result.get():
            self._raw_sents.extend(tn.raw_sents)
            self._sentences.extend(tn.sentences)
            self._tokens.extend(tn.tokens)

        pool.close()
        pool.join()
        print("Normalization pooling done: object", (id(self)))

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


if __name__ == "__main__":
    # print(TextNormalizer(document, enable={Property.Spelling}))
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
