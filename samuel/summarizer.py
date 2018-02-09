import numpy as np
from samuel.normalizer import TextNormalizer
from typing import List, Dict, Tuple, Callable, Iterable, Union


class TextSummarizer:
    def __init__(self, raw_sents: List[str], norm_sents: List[List[str]], summary_length: int,
                 sort_by_score: bool = False):
        self._sort_by_score = sort_by_score
        self._summary_length = summary_length
        self._sentences = raw_sents
        self._norm_sents = norm_sents
        self._cosine_matrix = TextSummarizer.__build_cosine_matrix(norm_sents)

    @staticmethod
    def __build_cosine_matrix(sentences: List[List[str]]) -> np.ndarray:
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

        print("Constructing cosine similarity matrix: @TextNormalizer", )
        word_dictionary = {word: i for word, i in word_dict()}
        term_freq = np.array([vector for vector in word_vec(word_dictionary)], dtype=np.float64)
        inv_doc_freq = np.full((len(word_dictionary),), np.log(float(len(term_freq))), dtype=np.float64)
        inv_doc_freq = np.divide(inv_doc_freq, np.sum(term_freq, axis=0))
        tf_idf = np.multiply(term_freq, inv_doc_freq)

        def cos_matrix():
            for i in range(len(tf_idf)):
                vector = np.zeros((len(tf_idf),), dtype=np.float64)
                for j in range(len(tf_idf)):
                    value = (np.sum(np.multiply.reduce([term_freq[i], term_freq[j], np.square(inv_doc_freq)]))
                             / (np.linalg.norm(tf_idf[i]) * np.linalg.norm(tf_idf[j])))
                    vector[j] = value
                yield vector


        return np.array(list(cos_matrix()), dtype=np.float64)

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
        print("Aggregating scores and summary: object", id(self))
        score_sents = [{"index": i, "score": scores[i-1], "sentence": sentence}
                       for i, sentence in enumerate(self._sentences)]
        score_sents.sort(key=lambda sentence: sentence["score"], reverse=True)
        summary = score_sents[:self._summary_length]
        if not self._sort_by_score:
            summary.sort(key=lambda sentence: sentence["index"], reverse=True)
        summary_text = " ".join([sentence["sentence"] for sentence in summary])
        print("Summarization done: object", id(self))
        return summary_text, score_sents

    def continuous_lexrank(self, damping_factor: float = 0.85):
        _length = len(self._norm_sents)
        _damping_factor = damping_factor

        def state_func(t_matrix: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
            for u in range(_length):
                state = np.sum([t_matrix[u, v] / np.sum(t_matrix[:, v]) for v in range(_length)])
                state *= prev_state[u]
                state *= (1 - _damping_factor)
                state += _damping_factor / _length
                yield state

        transition_matrix = np.array(list(self.__apply_right_stochastic(self._cosine_matrix)), dtype=np.float64)
        print("Computing stationary distribution: object", id(self))
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

        print("Establishing transition matrix: object", id(self))
        transition_matrix = self._cosine_matrix
        transition_matrix = np.array(list(self.__apply_cos_threshold(transition_matrix)), dtype=np.float64)
        transition_matrix = np.array(list(self.__apply_right_stochastic(transition_matrix)), dtype=np.float64)
        transition_matrix = np.multiply(_lambda, transition_matrix)

        print("Establishing a teleporting random walk: object", id(self))
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

        print("Computing stationary distribution: object", id(self))
        stationary_distribution = TextSummarizer.__power_method(teleporting_random_walk, state_func).tolist()
        grank_one = stationary_distribution.index(max(stationary_distribution))
        num_of_ranked = 1

        def absorb_state(index: int) -> Iterable[float]:
            sentence_index = markov_chain_tracker.pop(index)
            markov_chain_tracker.insert(0, sentence_index)

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

        print("Computing N visits for ranked states: object", id(self))
        teleporting_random_walk = teleporting_random_walk.tolist()
        visit_n = absorb_state(grank_one)
        for i in range(1, _length):
            num_of_ranked += 1
            sentence_index = visit_n.index(max(visit_n))
            sentence_index += num_of_ranked - 1
            visit_n = absorb_state(sentence_index)

        print("Computing summary scores: object", id(self))
        scores = list(range(_length))
        for i in range(_length):
            scores[markov_chain_tracker.pop()] = _length - i
        scores = np.divide(scores, np.linalg.norm(scores, ord=1))
        scores = np.divide(scores, max(scores))
        return self.__score_aggregator(scores)

    def mmr(self, query: str, score_sents: List[Dict[str, Union[float, str]]], _lambda: float = 0.7):
        q = TextNormalizer(query)
        norm_sents = self._norm_sents[:]
        norm_sents.append(q.tokens)
        cosine_matrix = TextSummarizer.__build_cosine_matrix(norm_sents)
        mmr_scores = list()

        def compute_mmr(index):
            similarity_scores = [cosine_matrix[index][s["index"]] for s in mmr_scores]
            maximum_similarity = max(similarity_scores) if similarity_scores else 0
            mmr = _lambda * (cosine_matrix[index][-1] - (1 - _lambda) * maximum_similarity)
            return mmr

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

    document1 = '''Iraqi vice president taha yassin ramadan announced today, sunday, that iraq refuses to back down from its
    decision to stop cooperating with disarmament inspectors before its demands are met. iraqi vice president taha yassin
    ramadan announced today, thursday, that iraq rejects cooperating with the united nations except on the issue of lifting 
    the blockade imposed upon it since the year 1990. Ramadan told reporters in baghdad that "iraq cannot deal positively 
    with whoever represents the security council unless there was a clear stance on the issue of lifting the blockade off 
    of it. Baghdad had decided late last october to completely cease cooperating with the inspectors of the united nations 
    special commision (unscom), in charge of disarming iraq's weapons, and whose work became very limited since the fifth 
    of august, and announced it will not resume its cooperation with the commission even if it were subjected to a military 
    operation. The russian foreign minister, igor ivanov, warned today, wednesday against using force against iraq, which 
    will destroy, according to him, seven years of difficult diplomatic work and will complicate the regional situation in 
    the area. Ivanov contended that carrying out air strikes against iraq, who refuses to cooperate with the united nations 
    inspectors, "will end the tremendous work achieved by the international group during the past seven years and will 
    complicate the situation in the region." Nevertheless, ivanov stressed that baghdad must resume working with the special 
    commission in charge of disarming the iraqi weapons of mass destruction (unscom). The special representative of the 
    united nations secretary-general in baghdad, prakash shah, announced today, wednesday, after meeting with the iraqi 
    deputy prime minister tariq aziz, that iraq refuses to back down from its decision to cut off cooperation with the 
    disarmament inspectors. British prime minister tony blair said today, sunday, that the crisis between the international 
    community and iraq "did not end" and that britain is still ready, prepared, and able to strike iraq." In a gathering 
    with the press held at the prime minister's office, blair contended that the crisis with iraq " will not end until iraq 
    has absolutely and unconditionally respected its commitments" towards the united nations. A spokesman for tony blair had
    indicated that the british prime minister gave permission to british air force tornado planes stationed to kuwait to 
    join the aerial bombardment against iraq.'''

    from samuel.test.test_document import single_test_document
    tn = TextNormalizer(single_test_document)
    ts = TextSummarizer(tn.raw_sents, tn.sentences, 5)
    # summary, scores = ts.grasshopper()
    # summary, scores = ts.continuous_lexrank()
    summary, scores = ts.pointwise_divrank()
    print(summary)
    # print(ts.mmr("iphone",)[0])
