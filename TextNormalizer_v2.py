import cProfile
import pstats
from enum import Enum
from typing import Dict, Type, List
from pprint import pprint
from warnings import filterwarnings
filterwarnings(action='ignore', category=UserWarning, module='gensim')


class TextNormalizer:
    def __init__(self, text: str, enable: List["Property"] = None, pos_filters: List[str] = None,
                 min_word_length: int = 1):
        if pos_filters is None:
            pos_filters = TextNormalizer.POS_FILTER
        if enable:
            enable.append(Property.Pos_Filter)
            if Property.Special_Char in enable:
                pos_filters.extend(TextNormalizer.POS_UNIVERSAL["other"])
            if Property.Spelling in enable:
                import re
                import enchant
                from nltk.corpus import wordnet
        else:
            enable = [Property.Pos_Filter]

        import spacy
        document = spacy.load("en")(text)

        from nltk.corpus import stopwords
        stop_words = stopwords.words("english")

        self._sentences = list()
        self._tokens = list()

        for sentence in document.sents:
            accepted_tokens = list()
            for token in sentence:
                word = token.lemma_
                if word == "":
                    continue
                if len(token) < min_word_length:
                    continue
                if (token.is_space
                        or token.is_digit
                        or token.is_left_punct
                        or token.is_right_punct
                        or token.is_quote
                        or token.is_bracket
                        or token.like_email
                        or token.like_num
                        or token.like_url):
                    continue
                if token.is_punct and Property.Special_Char not in enable:
                    continue
                if (token.is_stop or word in stop_words) and Property.Stopword not in enable:
                    continue
                else:
                    if Property.Pos_Filter in enable and token.pos_ not in pos_filters:
                        continue
                    if Property.Spelling in enable:
                        word = TextNormalizer.__correct_word(word, re, wordnet, enchant)
                if Property.Letter_Case in enable:
                    if token.is_upper:
                        word = word.upper()
                    if token.is_title:
                        word = word.capitalize()
                accepted_tokens.append(word)
            self._tokens.extend(accepted_tokens)
            if accepted_tokens:
                self._sentences.append(accepted_tokens)

    @staticmethod
    def __correct_word(word: str, re, wordnet, enchant) -> str:
        """
        Corrects the word by removing irregular repeated letters, then suggests possible words intended to be used
        using the PyEnchant library. You can check it at http://pythonhosted.org/pyenchant/api/enchant.html
        """

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
        enchant_dict = enchant.Dict("en_US")
        is_word_correct = enchant_dict.check(initial_correct_word)

        if is_word_correct:
            return initial_correct_word
        else:
            word_suggestions = enchant_dict.suggest(initial_correct_word)
            final_correct_word = word_suggestions[0] if word_suggestions else initial_correct_word
            return final_correct_word

    POS_FILTER = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]

    POS_UNIVERSAL = {
        "open_class": ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"],
        "closed_class": ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"],
        "other": ["PUNCT", "SYM", "X"]
    }


class Property(Enum):
    Letter_Case = "preserve_lettercase"
    Stopword = "preserve_stopword",
    Spelling = "correct_spelling",
    Pos_Filter = "pos_filter",
    Special_Char = "preserve_special_char"


document3 = """
Hands down, of all the smartphones I have used so far, iPhone 8 Plus got the best battery life. I am not a heavy user. 
All I do is make few quick calls, check emails, quick update of social media and maps and navigation once in a while. 
On average with light use (excluding maps and navigation), iPhone 8 Plus lasts for 4 full days! You heard it right, 
4 full days! At the end of the 4th day, I am usually left with 5-10% of battery and that's about the time I charge the phone. 
The heaviest I used it was once when I had to rely on GPS for a full day. I started with 100% on the day I was travelling and by the end of the day, 
I had around 70% left. And I was able get through the next two days without any issues (light use only).

The last iPhone I used was an iPhone 5 and it is very clear that the smartphone cameras have come a long way. 
iPhone 8 Plus produces very crisp photos without any over saturation, which is what I really appreciate. 
Even though I got used to Samsung's over saturated photos over the last 3 years, whenever I see a photo true to real life colours, 
it really appeals me. When buying this phone, my main concern with camera was its performance in low light as I was used to pretty awesome 
performance on my Note 4. iPhone 8 Plus did not disappoint me. I was able to capture some shots at a work function and they looked truly amazing. 
Auto HDR seems very on point in my opinion. You will see these in the link below. Portrait mode has been somewhat consistent. 
I felt that it does not perform as well on a very bright day. But overall, given that it is still in beta, it works quite well (See Camaro SS photo). 
Video recording wise, it is pretty good at the standard 1080p 30fps. I am yet to try any 4k 60fps shots. But based on what I have seen from tech reviewers, 
it is pretty awesome.

For a LCD panel, iPhone 8 Plus display is great. Colours are accurate and it gets bright enough for outdoor use. Being a 1080p panel, 
I think it really contributes to the awesome battery life that I have been experiencing. Talking about Touch ID, 
I think it still is the most convenient way to unlock your phone and make any payments. For me personally, it works 99% of the time and in my experience, 
it still is the benchmark of fingerprint unlocking of any given smartphone.

I have missed iOS a lot over the last 3 years and it feels good to be back. Super smooth and no hiccups. 
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
running in order to allow "background upload", which makes no sense. Same goes for OneDrive. Overall, navigation around the OS is easy and convenient.

I really think Apple Maps still needs lot of catching up. Over the last few weeks, I managed to use it couple of times. Navigation wise, it seem to 
be good. But when it comes to looking up a place just by name seems like a real pain in the ass. Literally nothing shows up! Maybe it is a different 
story in other countries. But for now, Google Maps is the number 1 on my list.

People seem to be complaining about Apple's decision to stick with the same design for 4 generations of phones. To be honest I quite adore this design. 
It seems like a really timeless and well-aged design. The new glass back adds a little modern and polished look to the phone and it really helps grip 
the phone if you are not using a case. Overall, iPhone 8 Plus is a great smartphone for every day use, especially with that killer battery life. 
I do not really regret not getting an iPhone X, because in my opinion, first iteration will always be problematic. 8 Plus is the final iteration 
of that particular design and have constantly improved. I am sure for my usage, the specs are more than enough to get me through the next 2-3 years.
"""

cProfile.run("TextNormalizer(document3)", "summarizer")
p = pstats.Stats("summarizer")
p.strip_dirs().sort_stats("cumulative").print_stats(10)
p.sort_stats('time').print_stats(10)
