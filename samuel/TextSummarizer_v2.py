import numpy as np
from samuel.TextNormalizer_v2 import TextNormalizer
from typing import List, Dict, Tuple


class TextSummarizer:
    def __init__(self, sentences: List[List[str]]):
        TextSummarizer.__build_term_freq(sentences)

    @staticmethod
    def __build_term_freq(sentences: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:

        def word_dict() -> Dict[str, int]:
            index = 0
            word_set = set()
            for sentence in sentences:
                for word in sentence:
                    if word not in word_set:
                        yield word, index
                        index += 1
                        word_set.add(word)

        def word_vec(dictionary: Dict[str, int]) -> np.ndarray:
            for sentence in sentences:
                vector = np.zeros((len(dictionary),), dtype=int)
                for word in sentence:
                    vector[dictionary[word]] += 1
                yield vector

        word_dictionary = {word: i for word, i in word_dict()}
        word_matrix = np.array([vector for vector in word_vec(word_dictionary)])
        init_doc_freq = np.full((len(word_dictionary),), np.log(float(len(word_matrix))))
        word_doc_freq = np.divide(init_doc_freq, np.sum(word_matrix, axis=0))
        word_doc_freq = np.diag(word_doc_freq)

        return word_matrix, word_doc_freq


if __name__ == "__main__":
    tn = TextNormalizer("I have so many thoughts in my mind. Would you mind if I share them to you?")
    TextSummarizer(tn.sentences)