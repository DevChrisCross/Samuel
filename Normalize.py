from itertools import product
import nltk
import re
import string
import enchant
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pprint import pprint
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Normalizer:
    def __init__(self, text: str):
        self._text = text
        self._normalized = self._tokens = self._raw = None

    def original_text(self):
        return self._text

    def normalized_text(self):
        return self._normalized

    def raw_text(self):
        return self._raw

    def extracted_tokens(self):
        return self._tokens

    def normalize_text(self, request_tokens: bool =False, expand_contraction: bool =False, contraction_map: dict =None,
                       enable_pos_tag: bool =True, pos_tag_map: dict =None, correct_spelling: bool =False,
                       preserve_special_character: bool =False, preserve_punctation_emphasis: bool =False,
                       punctuation_emphasis_list: str =None, punctuation_emphasis_level: int =1,
                       preserve_lettercase: bool =False, preserve_wordform: bool =False, preserve_stopword: bool =False,
                       minimum_word_length: int =1):
        """
        Tokenizes sentences and words, then removes present stopwords and special characters, then performs
        lemmatization and further remove words which does not qualify in the part-of-speech tag map.

        :param contraction_map: expands the contractions in the given text
        :param correct_spelling: checks and corrects the words in the given text
            Be warned that enabling the module will perform an aggressive approach and will greatly decrease performance
        :return: dictionary of normalized and raw sentences and its tokens
        """

        clean_text = Normalizer.__remove_html_tags(self._text)
        raw_sentences = nltk.sent_tokenize(clean_text)
        if expand_contraction:
            contraction_map = contraction_map if contraction_map else Normalizer.DEFAULT_CONTRACTION_MAP
            raw_sentences = [Normalizer.__expand_contractions(sentence, contraction_map) for sentence in raw_sentences]

        stopwords_en = nltk.corpus.stopwords.words('english')
        stopwords_en.extend(["n't", "'s", "'d", "'t", "'ve", "'ll"])

        lemmatizer = WordNetLemmatizer()
        tokenized_sentences = [sentence.split() if preserve_punctation_emphasis or preserve_special_character
                               else nltk.word_tokenize(sentence) for sentence in raw_sentences]

        pos_tagged_sentences = nltk.pos_tag_sents(tokenized_sentences, tagset="universal")
        pos_tag_map = pos_tag_map if pos_tag_map else Normalizer.DEFAULT_POS_TAG_MAP

        special_character_regex_string = "[{}]".format(re.escape(string.punctuation))
        special_character_regex_compiled = re.compile(pattern=special_character_regex_string)

        word_with_punctuation_regex_compiled = punctuation_regex_compiled = None
        if preserve_punctation_emphasis and punctuation_emphasis_list:
            punctuation_regex_string = "[{}]".format('|'.join(re.escape(punctuation_emphasis_list)))
            punctuation_regex_compiled = re.compile(pattern=punctuation_regex_string + "+")
            word_with_punctuation_regex_string = "\w+" + punctuation_regex_string + "+"
            word_with_punctuation_regex_compiled = re.compile(pattern=word_with_punctuation_regex_string)

        def __is_special_character(word: str):
            return special_character_regex_compiled.sub(string=word, repl="") == ""

        normalized_sentences = list()
        for sentence in pos_tagged_sentences:
            new_sentence = list()
            for pos_tagged_word in sentence:
                word = pos_tagged_word[0]
                new_word = None
                if __is_special_character(word):
                    if not preserve_special_character:
                        continue
                    new_word = word
                else:
                    if len(word) < minimum_word_length:
                        continue
                    new_word = word = Normalizer.__clean_left_surrounding_text(word)
                    if word in stopwords_en:
                        if not preserve_stopword:
                            continue
                        new_word = word
                    else:
                        if preserve_punctation_emphasis:
                            filtered_word = Normalizer.__set_punctuation_emphasis(
                                word, punctuation_emphasis_level, special_character_regex_string,
                                word_with_punctuation_regex_compiled, punctuation_regex_compiled)
                            new_word = word = filtered_word if filtered_word else word
                        pos_tag = pos_tagged_word[1]
                        if enable_pos_tag:
                            if pos_tag not in pos_tag_map:
                                continue
                            wordnet_tag = pos_tag_map[pos_tag]
                            new_word = word if preserve_wordform else lemmatizer.lemmatize(word, wordnet_tag)
                        new_word = Normalizer.__correct_word(new_word) if correct_spelling else new_word
                    new_word = new_word if preserve_lettercase else new_word.lower()
                new_sentence.append(new_word)
            normalized_sentences.append(new_sentence)

        if request_tokens:
            tokens = list()
            for sentence in normalized_sentences:
                for word in sentence:
                    tokens.append(word)
            self._tokens = tokens

        self._normalized = normalized_sentences
        self._raw = raw_sentences
        return self

    @staticmethod
    def __remove_html_tags(text: str):
        """
        Removes the html tags in the text
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

    @staticmethod
    def __expand_contractions(text: str, contraction_map: dict):
        """
        Expands the contractions in the text
        """

        contraction_regex_string = "({})".format('|'.join(contraction_map.keys()))
        contraction_regex_compiled = re.compile(pattern=contraction_regex_string, flags=re.IGNORECASE | re.DOTALL)

        def expand_corpus(contraction):
            contraction_match = contraction.group(0).lower()
            expanded_contraction = contraction_map.get(contraction_match)
            return expanded_contraction

        expanded_corpus = contraction_regex_compiled.sub(string=text, repl=expand_corpus)
        return expanded_corpus

    @staticmethod
    def __clean_left_surrounding_text(word: str):
        for match in re.finditer(pattern="\w+", string=word):
            first_letter = match.span()[0]
            return word[first_letter:]

    @staticmethod
    def __set_punctuation_emphasis(word: str, emphasis_level: int, special_character_regex: re,
                                   word_punctuation_regex: re, punctuation_regex: re):
        if word_punctuation_regex.fullmatch(string=word):
            after_word_divider = None
            for match in punctuation_regex.finditer(string=word):
                after_word_divider = match.span()[0]
            word_part = word[:after_word_divider]
            punctuation_part = word[after_word_divider:]
            return word if len(punctuation_part) >= emphasis_level else word_part

        # for words that have punctuations not included in the emphasis list
        modified_sc_regex_compiled = re.compile(special_character_regex
                                                .replace("-", "")
                                                .replace("'", "")
                                                .replace(".", ""))
        for match in modified_sc_regex_compiled.finditer(string=word):
            return word[:match.span()[0]]

    @staticmethod
    def __correct_word(word: str):
        """
        Corrects the word by removing irregular repeated letters, then suggests possible words intended to be used
        using the PyEnchant library. You can check it at http://pythonhosted.org/pyenchant/api/enchant.html
        """

        enchant_dict = enchant.Dict("en_US")
        match_substitution = r'\1\2\3'
        repeat_regex_string = r'(\w*)(\w)\2(\w*)'
        repeat_regex_compiled = re.compile(pattern=repeat_regex_string)

        def __check_word(old_word: str):
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

    def __str__(self):
        return "[Normalized Text]: " + str(self._normalized) + "\n[Raw Text]: " + str(self._raw)\
               + "\n[Tokens]: " + str(self._tokens)

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


print(Normalizer( """
The Elder Scrolls V: Skyrim is an open world action role-playing video game developed by Bethesda Game Studios and published by Bethesda Softworks.
It is the fifth installment in The Elder Scrolls series, following The Elder Scrolls IV: Oblivion.
Skyrim's main story revolves around the player character and their effort to defeat Alduin the World-Eater, a dragon who is prophesied to destroy the world.
The game is set two hundred years after the events of Oblivion and takes place in the fictional province of Skyrim.
The player completes quests and develops the character by improving skills.
Skyrim continues the open world tradition of its predecessors by allowing the player to travel anywhere in the game world at any time, and to ignore or postpone the main storyline indefinitely.
The player may freely roam over the land of Skyrim, which is an open world environment consisting of wilderness expanses, dungeons, cities, towns, fortresses and villages.
Players may navigate the game world more quickly by riding horses, or by utilizing a fast-travel system which allows them to warp to previously, and players have the option to develop their character.
At the beginning of the game, players create their character by selecting one of several races, including humans, orcs, elves and anthropomorphic cat or lizard-like creatures, and then customizing their character's appearance, discovered locations.
Over the course of the game, players improve their character's skills, which are numerical representations of their ability in certain areas.
There are eighteen skills divided evenly among the three schools of combat, magic, and stealth.
Skyrim is the first entry in The Elder Scrolls to include Dragons in the game's wilderness.
Like other creatures, Dragons are generated randomly in the world and will engage in combat.
""")
      .normalize_text(preserve_lettercase=True))

# preserve_special_character=True, preserve_punctation_emphasis=True,
#                       punctuation_emphasis_list="?!", punctuation_emphasis_level=2, preserve_stopword=True,
#                       minimum_word_length=2, enable_pos_tag=False, preserve_lettercase=True