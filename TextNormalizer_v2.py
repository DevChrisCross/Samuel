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
                pos_filters.update(TextNormalizer.POS_UNIVERSAL["other"])
                if punct_filters is None:
                    punct_filters = set(TextNormalizer.PUNCT_FILTER)
            if Property.Stop_Word in enable:
                pos_filters.update(TextNormalizer.POS_UNIVERSAL["open_class"])
                pos_filters.update(TextNormalizer.POS_UNIVERSAL["closed_class"])
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

        def filtered_tokens(sentence: tokens.Span) -> str:
            for token in sentence:
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
            accepted_tokens = [token for token in filtered_tokens(sentence)]
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

    POS_UNIVERSAL = {
        "open_class": ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"],
        "closed_class": ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"],
        "other": ["PUNCT", "SYM", "X"]
    }


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

            documents = [document for document in partitioned_docs()]

        print("Preparing process pool: object", (id(self)))
        pool = mp.Pool()
        print("Mapping document batches: object", (id(self)))
        result = pool.map_async(partial(TextNormalizer, enable=enable, pos_filters=pos_filters,
                               punct_filters=punct_filters, min_word_length=min_word_length),
                       documents)

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

documents = ["""
My Hands are down, of all the smartphones I have used so far, iPhone 8 Plus got the best battery life. I am not a heavy user. 
All I do is make few quick calls, check emails, quick update of social media and maps and navigation once in a while. 
On average with light use (excluding maps and navigation), iPhone 8 Plus lasts for 4 full days! You heard it right, 
4 full days! At the end of the 4th day, I am usually left with 5-10% of battery and that's about the time I charge the phone. 
The heaviest I used it was once when I had to rely on GPS for a full day. I started with 100% on the day I was travelling and by the end of the day, 
I had around 70% left. And I was able get through the next two days without any issues (light use only).""",

"""The last iPhone I used was an iPhone 5 and it is very clear that the smartphone cameras have come a long way. 
iPhone 8 Plus produces very crisp photos without any over saturation, which is what I really appreciate. 
Even though I got used to Samsung's over saturated photos over the last 3 years, whenever I see a photo true to real life colours, 
it really appeals me. When buying this phone, my main concern with camera was its performance in low light as I was used to pretty awesome 
performance on my Note 4. iPhone 8 Plus did not disappoint me. I was able to capture some shots at a work function and they looked truly amazing. 
Auto HDR seems very on point in my opinion. You will see these in the link below. Portrait mode has been somewhat consistent. 
I felt that it does not perform as well on a very bright day. But overall, given that it is still in beta, it works quite well (See Camaro SS photo). 
Video recording wise, it is pretty good at the standard 1080p 30fps. I am yet to try any 4k 60fps shots. But based on what I have seen from tech reviewers, 
it is pretty awesome.""",

"""For a LCD panel, iPhone 8 Plus display is great. Colours are accurate and it gets bright enough for outdoor use. Being a 1080p panel, 
I think it really contributes to the awesome battery life that I have been experiencing. Talking about Touch ID, 
I think it still is the most convenient way to unlock your phone and make any payments. For me personally, it works 99% of the time and in my experience, 
it still is the benchmark of fingerprint unlocking of any given smartphone.""",

"""I have missed iOS a lot over the last 3 years and it feels good to be back. Super smooth and no hiccups. 
I know few people have experienced some bugs recently. I guess I was one of the lucky ones not to have any issues. 
Maybe it was already patched when I bought the phone. However, 
my only complaint is the fact that iOS still does not let you clear all notifications from a particular app at once. 
I really would like to see fixed in a future update. Customisation wise, I do not have any issues because I hardly customised my Note 4. 
Only widgets I had running were weather and calendar. Even then, I would still open up the actual app to seek more detail. However, 
I still do not use iCloud. I really wish Apple would have given us more online storage for backup. 
5GB is hardly enough these days to backup everything on your phone. One of my mates suggested that Apple should ship the phone with iCloud storage same as the phone. 
It surely would be awesome. But business wise, I cannot see Apple going ahead with such a decision. But in my opinion, 
iCloud users should get at least 15GB free. Coming from an Android, I thought it would make sense to keep using my Google account to 
sync contacts and photos as it would take away the hassle of setting everything up from scratch. Only issue is sometimes I feel like 
iOS restricting the background app refresh of Google apps such as Google photos. For example, I always have to keep Google Photos 
running in order to allow "background upload", which makes no sense. Same goes for OneDrive. Overall, navigation around the OS is easy and convenient.""",

"""I really think Apple Maps still needs lot of catching up. Over the last few weeks, I managed to use it couple of times. Navigation wise, it seem to 
be good. But when it comes to looking up a place just by name seems like a real pain in the ass. Literally nothing shows up! Maybe it is a different 
story in other countries. But for now, Google Maps is the number 1 on my list.""",

"""People seem to be complaining about Apple's decision to stick with the same design for 4 generations of phones. To be honest I quite adore this design. 
It seems like a really timeless and well-aged design. The new glass back adds a little modern and polished look to the phone and it really helps grip 
the phone if you are not using a case. Overall, iPhone 8 Plus is a great smartphone for every day use, especially with that killer battery life. 
I do not really regret not getting an iPhone X, because in my opinion, first iteration will always be problematic. 8 Plus is the final iteration 
of that particular design and have constantly improved. I am sure for my usage, the specs are more than enough to get me through the next 2-3 years.
"""]

document = " ".join(documents)

if __name__ == "__main__":
    # print(TextNormalizer(document, enable={Property.Spelling}))
    # cProfile.run("TextNormalizer(document, enable={Property.Spelling})", "Text_Normalizer")
    # tn_profiler = pstats.Stats("Text_Normalizer")
    # tn_profiler.strip_dirs().sort_stats("cumulative").print_stats(10)
    # tn_profiler.sort_stats('time').print_stats(10)

    NormalizerManager(document, batch_count=16)
    # cProfile.run("NormalizerManager(document, enable={Property.Spelling}, batch_count=16)", "Text_Normalizer_MP")
    # tn_mp_profiler = pstats.Stats("Text_Normalizer_MP")
    # tn_mp_profiler.strip_dirs().sort_stats("cumulative").print_stats(10)
    # tn_mp_profiler.sort_stats('time').print_stats(10)



