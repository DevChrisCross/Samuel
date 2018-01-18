import math
import nltk.data
from typing import Dict, List
from TextNormalizer import SentiText

BOOSTER_INCREMENT = 0.293
BOOSTER_DECREMENT = -0.293
CAPITAL_INCREMENT = 0.733
NEGATION_SCALAR = -0.74

NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
          "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
          "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "uhuh", "wasnt",
          "werent", "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't", "without", "wont", "wouldnt",
          "won't", "wouldn't", "rarely", "seldom", "despite"]

POSITIVE_WORD_BOOSTERS = ["absolutely", "amazingly", "awfully", "completely", "considerably", "decidedly", "deeply",
                          "effing", "enormously", "entirely", "especially", "exceptionally", "extremely", "fabulously",
                          "flipping", "flippin", "fricking", "frickin", "frigging", "friggin", "fully", "fucking",
                          "greatly", "hella", "highly", "hugely", "incredibly", "intensely", "majorly", "more", "most",
                          "particularly", "purely", "quite", "really", "remarkably", "so", "substantially",
                          "thoroughly", "totally", "tremendously", "uber", "unbelievably", "unusually", "utterly",
                          "very"]

NEGATIVE_WORD_BOOSTERS = ["almost", "barely", "hardly", "just enough", "kind of", "kinda", "kindof", "kind-of", "less",
                          "little", "marginally", "occasionally", "partly", "scarcely", "slightly", "somewhat",
                          "sort of", "sorta", "sortof", "sort-of"]

BOOSTER_DICT = dict.fromkeys(POSITIVE_WORD_BOOSTERS, BOOSTER_INCREMENT)
BOOSTER_DICT.update(dict.fromkeys(NEGATIVE_WORD_BOOSTERS, BOOSTER_DECREMENT))

SPECIAL_CASE_IDIOMS = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2, "cut the mustard": 2,
                       "kiss of death": -1.5, "hand to mouth": -2}


class Lexicon:
    def __init__(self, lexicon_filepath: str = "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt"):
        lexicon_textfile = nltk.data.load(lexicon_filepath)
        lexicon_dict = dict()
        for line in lexicon_textfile.split('\n'):
            word, measure = line.strip().split('\t')[0:2]
            lexicon_dict[word] = float(measure)
        self._lexicon = lexicon_dict

    def lexicon_dict(self):
        return self._lexicon


class SentimentIntensityAnalyzer:
    def __init__(self, lexicon_dict: Dict[str, float]):
        self._lexicon = lexicon_dict

    def polarity_scores(self, text: str):
        sentitext = SentiText(text)
        # text, words_and_emoticons, is_cap_diff = self.preprocess(text)

        word_valence = list()
        words_with_emoticons = sentitext.words_and_emoticons
        # print(words_with_emoticons)

        for i, item in enumerate(words_with_emoticons):

            valence = 0
            if ((i < len(words_with_emoticons) - 1
                 and words_with_emoticons[i].lower() == "kind" and words_with_emoticons[i+1].lower() == "of")
                    or item.lower() in BOOSTER_DICT):
                word_valence.append(valence)
                continue
            word_valence.append(self.sentiment_valence(valence, sentitext, item, i))
        word_valence = self._but_word_check(words_with_emoticons, word_valence)
        # print(word_valence)
        return self.score_valence(word_valence, text)

    def sentiment_valence(self, valence: float, sentitext, item: str, current_word_index: int):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lower = item.lower()

        if item_lower in self._lexicon:
            valence = self._lexicon[item_lower]

            if item.isupper() and is_cap_diff:
                valence = valence + (CAPITAL_INCREMENT if valence > 0 else -CAPITAL_INCREMENT)

            def preceding_word_booster_effect(word: str, valence: float, is_cap_diff: bool):
                scalar = 0.0
                word_lower = word.lower()
                if word_lower in BOOSTER_DICT:
                    scalar = BOOSTER_DICT[word_lower]
                    scalar = scalar * (-1 if valence < 0 else 1)
                    if word.isupper() and is_cap_diff:
                        scalar = scalar + (CAPITAL_INCREMENT if valence > 0 else -CAPITAL_INCREMENT)
                return scalar

            for preceding_word_distance in range(0, 3):
                preceding_word_index = current_word_index - (preceding_word_distance + 1)
                preceding_word = words_and_emoticons[preceding_word_index]

                if preceding_word_distance < current_word_index and preceding_word.lower() not in self._lexicon:
                    scalar = preceding_word_booster_effect(preceding_word, valence, is_cap_diff)
                    if preceding_word_distance and not scalar == 0:
                        scalar = scalar * (0.95 if preceding_word_distance == 1 else 0.9)

                    valence += scalar
                    valence = self._negation_word_check(valence, words_and_emoticons, preceding_word_distance, current_word_index)

                    if preceding_word_distance == 2:
                        valence = self._idiom_word_check(valence, words_and_emoticons, current_word_index)

            valence = self._least_word_check(valence, words_and_emoticons, current_word_index)
        return valence

    def _negation_word_check(self, valence, word_tokens, preceding_word_distance, current_word_index):

        def is_word_negated(words: List[str], include_nt: bool = True):
            negative_words = list(NEGATE)
            for word in negative_words:
                if word in words:
                    return True
            if include_nt:
                for word in words:
                    if "n't" in word:
                        return True
            if "least" in words:
                i = words.index("least")
                if i > 0 and words[i - 1] != "at":
                    return True
            return False

        immediate_distance = 1
        preceding_word = word_tokens[current_word_index - (preceding_word_distance + immediate_distance)]
        if preceding_word_distance == 0:
            if is_word_negated([preceding_word]):
                valence *= NEGATION_SCALAR

        if preceding_word_distance == 1:
            if (word_tokens[current_word_index - 2] == "never"
                    and (word_tokens[current_word_index - 1] == "so"
                         or word_tokens[current_word_index - 1] == "this")):
                valence *= 1.5
            elif is_word_negated([preceding_word]):
                valence *= NEGATION_SCALAR

        if preceding_word_distance == 2:
            if (word_tokens[current_word_index - 3] == "never"
                    and (word_tokens[current_word_index - 2] == "so"
                         or word_tokens[current_word_index - 2] == "this")
                    or (word_tokens[current_word_index - 1] == "so"
                        or word_tokens[current_word_index - 1] == "this")):
                valence *= 1.25
            elif is_word_negated([preceding_word]):
                valence *= NEGATION_SCALAR

        return valence

    def _idiom_word_check(self, valence, word_tokens, current_word_index):
        current_word = word_tokens[current_word_index]
        first_preceding_word = word_tokens[current_word_index - 1]
        second_preceding_word = word_tokens[current_word_index - 2]
        third_preceding_word = word_tokens[current_word_index - 3]

        phrase_sequences = [
            f"{first_preceding_word} {current_word}",
            f"{second_preceding_word} {first_preceding_word} {current_word}",
            f"{second_preceding_word} {first_preceding_word}",
            f"{third_preceding_word} {second_preceding_word} {first_preceding_word}",
            f"{third_preceding_word} {second_preceding_word}"
        ]

        for word_phrase in phrase_sequences:
            if word_phrase in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[word_phrase]
                break

        if len(word_tokens)-1 > current_word_index:
            phrase = f"{current_word} {word_tokens[current_word_index + 1]}"
            if phrase in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[phrase]

        if len(word_tokens)-1 > current_word_index+1:
            phrase = f"{current_word} {word_tokens[current_word_index + 1]} {word_tokens[current_word_index + 2]}"
            if phrase in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[phrase]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        if phrase_sequences[4] in BOOSTER_DICT or phrase_sequences[2] in BOOSTER_DICT:
            valence = valence + BOOSTER_DECREMENT

        return valence

    def _least_word_check(self, valence, word_tokens, current_word_index):
        # check for negation case using "least"
        if (current_word_index > 1
                and word_tokens[current_word_index - 1].lower() not in self._lexicon
                and word_tokens[current_word_index - 1].lower() == "least"):
            if (word_tokens[current_word_index - 2].lower() != "at"
                    and word_tokens[current_word_index - 2].lower() != "very"):
                valence = valence * NEGATION_SCALAR

        elif (current_word_index > 0
              and word_tokens[current_word_index - 1].lower() not in self._lexicon
              and word_tokens[current_word_index - 1].lower() == "least"):
            valence = valence * NEGATION_SCALAR
        return valence

    def _but_word_check(self, word_tokens, word_valence):
        # check for modification in sentiment due to contrastive conjunction 'but'
        if 'but' in word_tokens or 'BUT' in word_tokens:
            try:
                but_word_index = word_tokens.index('but')
            except ValueError:
                but_word_index = word_tokens.index('BUT')

            for valence in word_valence:
                valence_index = word_valence.index(valence)
                if valence_index < but_word_index:
                    word_valence.pop(valence_index)
                    word_valence.insert(valence_index, valence * 0.5)
                elif valence_index > but_word_index:
                    word_valence.pop(valence_index)
                    word_valence.insert(valence_index, valence * 1.5)

        return word_valence

    def score_valence(self, word_valence, text):
        if word_valence:
            total_valence = float(sum(word_valence))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if total_valence > 0:
                total_valence += punct_emph_amplifier
            elif total_valence < 0:
                total_valence -= punct_emph_amplifier

            # print(total_valence)

            def normalize_valence(score, alpha: int = 15):
                norm_score = score / math.sqrt((score * score) + alpha)
                return norm_score

            compound = normalize_valence(total_valence)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._separate_sentiment_score(word_valence)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg" : round(neg, 3),
             "neu" : round(neu, 3),
             "pos" : round(pos, 3),
             "compound" : round(compound, 4)}

        return sentiment_dict

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_exclamation_point(text)
        qm_amplifier = self._amplify_question_mark(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    def _amplify_exclamation_point(self, text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        exclamation_point_count = text.count("!")
        if exclamation_point_count > 4:
            exclamation_point_count = 4
        ep_amplifier = exclamation_point_count*0.292
        return ep_amplifier

    def _amplify_question_mark(self, text):
        # check for added emphasis resulting from question marks (2 or 3+)
        question_mark_count = text.count("?")
        qm_amplifier = 0
        if question_mark_count > 1:
            if question_mark_count <= 3:
                qm_amplifier = question_mark_count*0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    def _separate_sentiment_score(self, word_valence):
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in word_valence:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) +1) # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) -1) # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count


def unsupervised_extractor(review: str, threshold: float = 0.1, verbose: bool = False):
    """

    :param review:
    :param threshold:
    :param verbose:
    :return:
    """

    lexicon = Lexicon().lexicon_dict()
    analyzer = SentimentIntensityAnalyzer(lexicon)
    scores = analyzer.polarity_scores(review)
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold else 'negative'
    if verbose:
        positive = str(round(scores['pos'], 2)*100) + '%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100) + '%'
        neutral = str(round(scores['neu'], 2)*100) + '%'
        compound = str(round(scores['compound'], 2) * 100) + '%'
        # print("Compound " + str(compound))
        # print("Positive " + str(positive))
        # print("Final " + str(final))
        # print("Negative " + str(negative))
        # print("Neutral " + str(neutral))
    return final_sentiment

sample_data = [("I hope this group of highly film-makers!!!! never re-unites. ever again. IT SUCKS!!!! >:(", "negative"),
              ("a mesmerizing film that certainly keeps your attention... Ben Daniels is fascinating (and courageous) to watch..", "positive"),
              ("Worst horror film ever but funniest film ever rolled in one you have got to see this film it is so cheap it is unbeliaveble but you have to see it really!!!! P.s watch the carrot", "positive")]

# for review, review_sentiment in sample_data:
#     print("Review")
#     print(review)
#     print("Labeled Sentiment: ", review_sentiment)
#     final_sentiment = unsupervised_extractor(review, threshold=0.1, verbose=True)
#     print("Final Sentiment: " + final_sentiment)
#     print("-" * 60)


