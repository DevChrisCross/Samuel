import math
from typing import Dict, List, Tuple
from samuel.constants.vader import *


class TextSentimentClassifier:
    def __init__(self, text: str, tokens: List[str], neutrality_threshold: float = 0.1):
        if not isinstance(text, str):
            text = str(text.encode('utf-8'))
        self._text = text
        self._lexicon = VADER
        self._tokens = tokens

        is_different = False
        allcap_words = 0
        for word in self._tokens:
            if word.isupper():
                allcap_words += 1
        cap_differential = len(self._tokens) - allcap_words
        if 0 < cap_differential < len(self._tokens):
            is_different = True

        self._is_cap_diff = is_different
        polarity_scores = self.__polarity_scores()

        if neutrality_threshold * -1 < polarity_scores["compound"] < neutrality_threshold:
            final_sentiment = "neutral"
        else:
            final_sentiment = "positive" if polarity_scores["compound"] > 0 else 'negative'
            
        positive = polarity_scores['pos']
        negative = polarity_scores['neg']
        neutral = str(round(polarity_scores["neu"], 2) * 100) + '%'
        # norm = np.linalg.norm([positive, negative], ord=1)
        # positive /= norm
        # negative /= norm
        positive = str(round(positive, 2) * 100) + '%'
        negative = str(round(negative, 2) * 100) + '%'

        self._scores = {
            "final_sentiment": final_sentiment,
            "compound": polarity_scores["compound"],
            "percentage": {
                "positive": positive,
                "negative": negative,
                "neutral": neutral
            }
        }

    def __polarity_scores(self) -> Dict[str, float]:
        word_valences = list()
        for index, token in enumerate(self._tokens):
            valence = 0
            if ((index < len(self._tokens) - 1
                 and self._tokens[index].lower() == "kind"
                 and self._tokens[index + 1].lower() == "of")
                    or token.lower() in BOOSTER_DICT):
                word_valences.append(valence)
                continue
            word_valences.append(self.__sentiment_valence(valence, token, index))
        word_valences = self.__but_word_check(word_valences)
        return self.__score_valence(word_valences)

    def __sentiment_valence(self, valence: float, token: str, token_index: int) -> float:
        token_lower = token.lower()
        if token_lower in self._lexicon:
            valence = self._lexicon[token_lower]
            if token.isupper() and self._is_cap_diff:
                valence += (CAPITAL_INCREMENT if valence > 0 else -CAPITAL_INCREMENT)

            def preceding_word_booster_effect(word: str, current_valence: float) -> float:
                _scalar = 0.0
                word_lower = word.lower()
                if word_lower in BOOSTER_DICT:
                    _scalar = BOOSTER_DICT[word_lower]
                    _scalar *= (-1 if current_valence < 0 else 1)
                    if word.isupper() and self._is_cap_diff:
                        _scalar += (CAPITAL_INCREMENT if current_valence > 0 else -CAPITAL_INCREMENT)
                return _scalar

            for preceding_word_dist in range(0, 3):
                preceding_word_index = token_index - (preceding_word_dist + 1)
                preceding_word = self._tokens[preceding_word_index]
                if preceding_word_dist < token_index and preceding_word.lower() not in self._lexicon:
                    scalar = preceding_word_booster_effect(preceding_word, valence)

                    if preceding_word_dist and not scalar == 0:
                        scalar *= (0.95 if preceding_word_dist == 1 else 0.9)
                    valence += scalar
                    valence = self.__negation_word_check(valence, preceding_word_dist, token_index)

                    if preceding_word_dist == 2:
                        valence = self.__idiom_word_check(valence, token_index)
            valence = self.__least_word_check(valence, token_index)

        return valence

    def __negation_word_check(self, valence: float, word_distance: int, token_index: int) -> float:

        def is_word_negated(words: List[str], include_nt: bool = True) -> bool:
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
        preceding_word = self._tokens[token_index - (word_distance + immediate_distance)]
        if word_distance == 0:
            if is_word_negated([preceding_word]):
                valence *= NEGATION_SCALAR

        if word_distance == 1:
            if (self._tokens[token_index - 2] == "never"
                    and (self._tokens[token_index - 1] == "so"
                         or self._tokens[token_index - 1] == "this")):
                valence *= 1.5
            elif is_word_negated([preceding_word]):
                valence *= NEGATION_SCALAR

        if word_distance == 2:
            if (self._tokens[token_index - 3] == "never"
                    and (self._tokens[token_index - 2] == "so"
                         or self._tokens[token_index - 2] == "this")
                    or (self._tokens[token_index - 1] == "so"
                        or self._tokens[token_index - 1] == "this")):
                valence *= 1.25
            elif is_word_negated([preceding_word]):
                valence *= NEGATION_SCALAR

        return valence

    def __idiom_word_check(self, valence: float, token_index: int) -> float:
        current_word = self._tokens[token_index]
        first_preceding_word = self._tokens[token_index - 1]
        second_preceding_word = self._tokens[token_index - 2]
        third_preceding_word = self._tokens[token_index - 3]

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

        if len(self._tokens) - 1 > token_index:
            phrase = f"{current_word} {self._tokens[token_index + 1]}"
            if phrase in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[phrase]

        if len(self._tokens) - 1 > token_index + 1:
            phrase = f"{current_word} {self._tokens[token_index + 1]} {self._tokens[token_index + 2]}"
            if phrase in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[phrase]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        if phrase_sequences[4] in BOOSTER_DICT or phrase_sequences[2] in BOOSTER_DICT:
            valence += BOOSTER_DECREMENT

        return valence

    def __least_word_check(self, valence: float, token_index: int) -> float:
        # check for negation case using "least"
        if (token_index > 1
                and self._tokens[token_index - 1].lower() not in self._lexicon
                and self._tokens[token_index - 1].lower() == "least"):
            if (self._tokens[token_index - 2].lower() != "at"
                    and self._tokens[token_index - 2].lower() != "very"):
                valence *= NEGATION_SCALAR

        elif (token_index > 0
              and self._tokens[token_index - 1].lower() not in self._lexicon
              and self._tokens[token_index - 1].lower() == "least"):
            valence *= NEGATION_SCALAR

        return valence

    def __but_word_check(self, word_valences: List[float]) -> List[float]:
        # check for modification in sentiment due to contrastive conjunction 'but'
        if 'but' in self._tokens or 'BUT' in self._tokens:
            try:
                but_word_index = self._tokens.index('but')
            except ValueError:
                but_word_index = self._tokens.index('BUT')

            for valence in word_valences:
                valence_index = word_valences.index(valence)
                if valence_index < but_word_index:
                    word_valences.pop(valence_index)
                    word_valences.insert(valence_index, valence * 0.5)
                elif valence_index > but_word_index:
                    word_valences.pop(valence_index)
                    word_valences.insert(valence_index, valence * 1.5)

        return word_valences

    def __score_valence(self, word_valences: List[float]) -> Dict[str, float]:
        if word_valences:
            total_valence = float(sum(word_valences))
            punct_emph_amplifier = self.__punctuation_emphasis()

            if total_valence > 0:
                total_valence += punct_emph_amplifier
            elif total_valence < 0:
                total_valence -= punct_emph_amplifier

            def normalize_valence(score, alpha: int = 15):
                norm_score = score / math.sqrt((score * score) + alpha)
                return norm_score

            compound = normalize_valence(total_valence)
            pos_sum, neg_sum, neu_count = self.__discriminate_sentiment_score(word_valences)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)
        else:
            compound = pos = neg = neu = 0.0

        return {
            "neg": round(neg, 3),
            "neu": round(neu, 3),
            "pos": round(pos, 3),
            "compound": round(compound, 4)
        }

    def __punctuation_emphasis(self) -> float:
        exclamation_point_count = self._text.count("!")
        if exclamation_point_count > 4:
            exclamation_point_count = 4
        ep_amplifier = exclamation_point_count * 0.292

        question_mark_count = self._text.count("?")
        qm_amplifier = 0
        if question_mark_count > 1:
            if question_mark_count <= 3:
                qm_amplifier = question_mark_count * 0.18
            else:
                qm_amplifier = 0.96

        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def __discriminate_sentiment_score(word_valences: List[float]) -> Tuple[float, float, float]:
        pos_sum = neg_sum = neu_count = 0.0
        for sentiment_score in word_valences:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    @property
    def sentiment_score(self):
        return self._scores


if __name__ == "__main__":
    pass
