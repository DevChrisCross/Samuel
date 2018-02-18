import numpy as np
from samuel.normalizer import TextNormalizer, NormalizerManager
from typing import List, Dict, Tuple, Callable, Iterable, Union


class TextSummarizer:
    def __init__(self, raw_sents: List[str], norm_sents: List[List[str]], summary_length: int,
                 sort_by_score: bool = False):
        self._id = id(self)
        self._name = self.__class__.__name__

        self._sort_by_score = sort_by_score
        self._summary_length = summary_length
        self._sentences = raw_sents
        self._norm_sents = norm_sents
        self._cosine_matrix = self.__build_cosine_matrix(norm_sents)

    def __build_cosine_matrix(self, sentences: List[List[str]]) -> np.ndarray:
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
                vector = np.zeros((len(dictionary),), dtype=np.int)
                for word in sentence:
                    vector[dictionary[word]] += 1
                yield vector

        print(self._name, self._id, "Constructing cosine similarity matrix")
        word_dictionary = {word: i for word, i in word_dict()}
        term_freq = np.array([vector for vector in word_vec(word_dictionary)], dtype=np.float64)
        inv_doc_freq = np.full((len(word_dictionary),), np.log(float(len(term_freq))), dtype=np.float64)
        inv_doc_freq = np.divide(inv_doc_freq, np.sum(term_freq, axis=0))
        tf_idf = np.multiply(term_freq, inv_doc_freq)

        inv_doc_freq = np.repeat([np.square(inv_doc_freq)], len(term_freq), axis=0)
        init_cos_matrix = np.dot(term_freq, np.transpose(np.multiply(term_freq, inv_doc_freq)))
        norms = np.array([np.linalg.norm(tf_idf[i]) for i in range(len(tf_idf))], dtype=np.float64)
        matrix_norm = np.outer(norms, norms)
        cos_matrix = np.divide(init_cos_matrix, matrix_norm)

        return cos_matrix

    @staticmethod
    def __apply_cos_threshold(cos_matrix: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        for i in range(len(cos_matrix)):
            vector = np.zeros((len(cos_matrix),), dtype=np.float64)
            for j in range(len(cos_matrix)):
                vector[j] = 1 if cos_matrix[i][j] > threshold else 0
            yield vector

    @staticmethod
    def __apply_right_stochastic(cos_matrix: np.ndarray) -> np.ndarray:
        for i in range(len(cos_matrix)):
            yield np.divide(cos_matrix[i], np.linalg.norm(cos_matrix[i], ord=1))

    @staticmethod
    def __power_method(transition_matrix: np.ndarray, state_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       initial_state: np.ndarray = None, normalize: bool = True, error_tol: float = 1e-3) -> np.ndarray:
        time_t = 0
        delta = np.inf
        _length = len(transition_matrix)
        states = np.empty((0, _length), dtype=np.float64)
        states = np.append(states, np.array([initial_state if initial_state
                                             else np.full((_length,), 1/_length, dtype=np.float64)],
                                            dtype=np.float64), axis=0)

        while delta > error_tol:
            time_t += 1
            states = np.append(states, np.array([list(state_func(transition_matrix, states[time_t-1]))],
                                                dtype=np.float64), axis=0)
            delta = np.linalg.norm(np.subtract(states[time_t], states[time_t - 1]))

        if normalize:
            states[time_t] = np.divide(states[time_t], np.amax(states[time_t]))
        return states[time_t]

    def __score_aggregator(self, scores: np.ndarray) -> Tuple[str, List[Dict[str, Union[float, str]]]]:
        print(self._name, self._id, "Aggregating scores and summary")
        score_sents = [{"index": i, "score": scores[i-1], "sentence": sentence}
                       for i, sentence in enumerate(self._sentences)]
        score_sents.sort(key=lambda sentence: sentence["score"], reverse=True)
        summary = score_sents[:self._summary_length]
        if not self._sort_by_score:
            summary.sort(key=lambda sentence: sentence["index"], reverse=True)
        summary_text = " ".join([sentence["sentence"] for sentence in summary])
        print(self._name, self._id, "Summarization done")
        return summary_text, score_sents

    def continuous_lexrank(self, damping_factor: float = 0.85):
        # _length = len(self._norm_sents)
        # _damping_factor = damping_factor

        def state_func(t_matrix: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
            # for u in range(_length):
            #     state = np.sum([t_matrix[u, v] / np.sum(t_matrix[:, v]) for v in range(_length)])
            #     state *= prev_state[u]
            #     state *= (1 - _damping_factor)
            #     state += _damping_factor / _length
            #     yield state
            return np.dot(t_matrix.transpose(), prev_state)

        transition_matrix = np.array(list(self.__apply_cos_threshold(self._cosine_matrix)), dtype=np.float64)
        transition_matrix = np.array(list(self.__apply_right_stochastic(transition_matrix)), dtype=np.float64)
        print(self._name, self._id, "Computing stationary distribution")
        scores = TextSummarizer.__power_method(transition_matrix, state_func)
        return self.__score_aggregator(scores)

    def pointwise_divrank(self, alpha: float = 0.25, _lambda: float = 0.9, beta: float = None,
                          prior_distribution: np.ndarray = None):
        _length = len(self._norm_sents)
        _alpha = alpha
        _beta = beta

        print("Establishing transition matrix: object", id(self))
        transition_matrix = self._cosine_matrix
        transition_matrix = np.array(list(self.__apply_cos_threshold(transition_matrix)), dtype=np.float64)
        transition_matrix = np.array(list(self.__apply_right_stochastic(transition_matrix)), dtype=np.float64)
        organic_matrix = np.array(list([(1 - _alpha) if i == j else (_alpha * transition_matrix[i, j])
                                        for j in range(_length)] for i in range(_length)), dtype=np.float64)

        print("Establishing prior distribution: object", id(self))
        if not prior_distribution:
            if _beta:
                prior_distribution = np.array([np.power(i + 1, _beta * -1) for i in range(_length)], dtype=np.float64)
            else:
                prior_distribution = np.full((_length,), 1 / _length, dtype=np.float64)

        n_visit = np.full((_length,), 1)
        def state_func(t_matrix: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
            for u in range(_length):
                state = 0
                for v in range(_length):
                    if transition_matrix[u, v]:
                        n_visit[v] += 1
                    state += (organic_matrix[u, v] * n_visit[v]
                              / np.sum([organic_matrix[u, z] * n_visit[z] for z in range(_length)]))
                # state = np.sum([(organic_matrix[u, v] * prev_state[v])
                #                 / np.sum([organic_matrix[z, u] * prev_state[z] for z in range(_length)])
                #                 for v in range(_length)])
                state *= prev_state[u]
                state *= _lambda
                state += ((1 - _lambda) * prior_distribution[u])
                yield state

        print("Computing stationary distribution: object", id(self))
        scores = TextSummarizer.__power_method(transition_matrix, state_func)
        return self.__score_aggregator(scores)

    def grasshopper(self, _lambda: float = 0.5, alpha: float = None, prior_distribution: np.ndarray = None):
        _length = len(self._norm_sents)
        _alpha = alpha

        print(self._name, self._id, "Establishing transition matrix")
        transition_matrix = self._cosine_matrix
        transition_matrix = np.array(list(self.__apply_cos_threshold(transition_matrix)), dtype=np.float64)
        transition_matrix = np.array(list(self.__apply_right_stochastic(transition_matrix)), dtype=np.float64)
        transition_matrix = np.multiply(_lambda, transition_matrix)

        print(self._name, self._id, "Establishing a teleporting random walk")
        if not prior_distribution:
            if alpha:
                prior_distribution = np.array([np.power(i + 1, _alpha * -1) for i in range(_length)], dtype=np.float64)
            else:
                prior_distribution = np.full((_length,), 1/_length, dtype=np.float64)
        all_one_vector = np.full(_length, 1, dtype=np.float64)
        prior_distribution = np.outer(all_one_vector, prior_distribution)
        prior_distribution = np.multiply((1 - _lambda), prior_distribution)

        teleporting_random_walk = np.add(transition_matrix, prior_distribution)
        markov_chain_tracker = list(range(_length))

        def state_func(t_matrix: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
            return np.dot(t_matrix.transpose(), prev_state)

        print(self._name, self._id, "Computing stationary distribution")
        stationary_distribution = TextSummarizer.__power_method(teleporting_random_walk, state_func).tolist()
        grank_one = stationary_distribution.index(max(stationary_distribution))
        num_of_ranked = 1

        def absorb_state(index: int) -> Iterable[float]:
            sentence_index = markov_chain_tracker.pop(index)
            markov_chain_tracker.insert(0, sentence_index)
            print(self._name, self._id, "Absorbed state: sentence", sentence_index)

            absorbing_state = np.full((_length,), 0, dtype=np.float64)
            absorbing_state[index] = 1
            teleporting_random_walk[index] = absorbing_state
            sentence_vector = teleporting_random_walk.pop(index)
            teleporting_random_walk.insert(0, sentence_vector)

            submatrix_q = np.array(teleporting_random_walk)
            submatrix_q = submatrix_q[num_of_ranked:_length, num_of_ranked:_length]
            identity_matrix = np.diag(np.full((_length - num_of_ranked,), 1, dtype=np.float64))
            fundamental_matrix = np.linalg.inv((identity_matrix - submatrix_q))
            all_one_vector = np.full(_length - num_of_ranked, 1, dtype=np.float64)
            n_visit = np.dot(fundamental_matrix, all_one_vector) / (_length - num_of_ranked)
            return n_visit.tolist()

        print(self._name, self._id, "Computing N visits for ranked states")
        teleporting_random_walk = teleporting_random_walk.tolist()
        visit_n = absorb_state(grank_one)
        for i in range(1, _length):
            print(self._name, self._id, "N visit iteration:", i)
            num_of_ranked += 1
            sentence_index = visit_n.index(max(visit_n))
            sentence_index += num_of_ranked - 1
            visit_n = absorb_state(sentence_index)

        print(self._name, self._id, "Computing summary scores")
        scores = list(range(_length))
        for i in range(_length):
            scores[markov_chain_tracker.pop()] = _length - i
        scores = np.divide(scores, np.linalg.norm(scores, ord=1))
        scores = np.divide(scores, max(scores))
        return self.__score_aggregator(scores)

    def mmr(self, query: str, score_sents: List[Dict[str, Union[float, str]]], _lambda: float = 0.7):
        print(self._name, self._id, "Reconstructing cosine matrix")
        q = TextNormalizer(query)
        norm_sents = self._norm_sents[:]
        norm_sents.append(q.tokens)
        cosine_matrix = self.__build_cosine_matrix(norm_sents)
        mmr_scores = list()
        if cosine_matrix[-1][-1] == np.nan:
            raise ValueError("Invalid query")

        def compute_mmr(index):
            similarity_scores = [cosine_matrix[index][s["index"]] for s in mmr_scores]
            maximum_similarity = max(similarity_scores) if similarity_scores else 0
            mmr = _lambda * (cosine_matrix[index][-1] - (1 - _lambda) * maximum_similarity)
            return mmr

        print(self._name, self._id, "Computing mmr scores: object")
        while score_sents:
            sentence = max(score_sents, key=lambda s: s["score"])
            sentence["score"] = compute_mmr(sentence["index"])
            mmr_scores.append(sentence)
            score_sents.remove(sentence)

        min_sent = min(mmr_scores, key=lambda s: s["score"])
        max_sent = max(mmr_scores, key=lambda s: s["score"])
        min_mmr = min_sent["score"]
        max_mmr = max_sent["score"]
        scores = np.array([(sentence["score"] - min_mmr) / (max_mmr - min_mmr) for sentence in mmr_scores],
                          dtype=np.float64)
        return self.__score_aggregator(scores)


if __name__ == "__main__":
    # from samuel.test.test_document import single_test_document, test_documents
    # tn = TextNormalizer(single_test_document)
    # tn = NormalizerManager(single_test_document)
    # ts = TextSummarizer(tn.raw_sents, tn.sentences, 10)
    # summary, scores = ts.continuous_lexrank()
    # summary, scores = ts.mmr("I literally have never heard the phrase “MMA gloves” and I listen to the JRE at least once a week.", scores)
    # summary, scores = ts.continuous_lexrank()
    # summary, scores = ts.pointwise_divrank()
    # print(summary)
    # print(ts.mmr("iphone",)[0])
    pass
