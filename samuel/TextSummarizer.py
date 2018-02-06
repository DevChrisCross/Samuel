import numpy as np
import warnings
from typing import Callable, Tuple, Optional, Dict, Union, Set, List
from enum import Enum
from samuel.TextNormalizer import TextNormalizer
from textwrap import fill, indent

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class TextSummarizer:
    Word = str
    WordCountRow = Dict[Word, int]
    WordVector = Dict[int, WordCountRow]
    WordDictionary = Set[Word]
    NormalizedSentence = List[Word]

    def __init__(self, normalized_text: "TextNormalizer", settings: "TextSummarizer.Settings" = None):
        self._text = normalized_text.original_text
        self._raw_text = normalized_text.raw_text
        self._normalized_text = normalized_text.normalized_text

        self._word_vector, self._word_dictionary = TextSummarizer.__build_term_frequency(self._normalized_text)
        self._idf = TextSummarizer.__build_inverse_document_frequency(self._word_vector, self._word_dictionary)
        self._cosine_matrix = TextSummarizer.__build_cosine_matrix(self._word_vector, self._idf)

        self._summary_text = None
        self._sentences_score = [{
            "index": i,
            "raw_text": self._raw_text[i],
            "norm_text": self._normalized_text[i]
        } for i in range(len(self._raw_text))]
        self._settings = settings if settings else TextSummarizer.Settings()

    def __call__(self, summary_length: int, sort_by_score: bool = False, *args, **kwargs) -> "TextSummarizer":
        """
        Summarizes a document using the specified ranking algorithm set by the user.

        :param summary_length: the number of sentences needed in the summary
        :param sort_by_score: boolean. if the sentences should be sorted by appearance or score
        """

        settings = self._settings
        rank = settings.Rank
        rank_function_map = {
            rank.DIVRANK.name: self.__divrank,
            rank.LEXRANK.name: self.__continuous_lexrank,
            rank.GRASSHOPPER.name: self.__grasshopper,
        }
        scorebase = settings.ranking_mode["name"]
        # the try-except block specifically only accomodates internal errors that occur within
        # the grasshopper algorithm in rare cases e.g. Singular Matrix
        try:
            rank_function_map[scorebase]()
        except Exception:
            rank_function_map[rank.DIVRANK.name]()

        rerank = settings.Rerank
        if settings.reranking_mode:
            if settings.reranking_mode["name"] == rerank.MAXIMAL_MARGINAL_RELEVANCE.name:
                self.__maximal_marginal_relevance()
            if settings.reranking_mode["name"] == rerank.GRASSHOPPER.name:
                self.__grasshopper()
            scorebase = settings.reranking_mode["name"]
        self._sentences_score.sort(key=lambda sentence: sentence[scorebase], reverse=True)

        _summary = self._sentences_score[:summary_length]
        scorebase = scorebase if sort_by_score else "index"
        _summary.sort(key=lambda sentence: sentence[scorebase], reverse=sort_by_score)
        summary_text = " ".join([sentence["raw_text"] for sentence in _summary])
        self._summary_text = summary_text
        return self

    def __str__(self):
        return ("\n" + "-"*200
                + "\nNormalized Text:\n" + indent("\n".join([str(array) for array in self._normalized_text]),"\t")
                + "\n\nSummary:\n" + indent(fill(str(self._summary_text), width=150), "\t")
                + "\n" + ":"*200 + "\n[Settings]\n" + str(self._settings)
                + "\n" + "-"*200 + "\n")

    def __continuous_lexrank(self):
        """
        Lexical PageRank
        Computes the Lexrank for the corresponding given cosine matrix. A ranking algorithm which involves computing
        sentence importance based on the concept of eigen vector centrality in a graph representation of sentences.

        LexRank: Graph-based Lexical Centrality as Salience in Text Summarization
            Güneş Erkan         gerkan@umich.edu
            Dragomir R. Radev   radev@umich.edu
            Department of EECS, School of Information
            University of Michigan, Ann Arbor, MI 48109 USA
        You can check the algorithm at https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html
        """

        settings = self._settings
        params = params = settings.parameters_of(settings.Rank.LEXRANK)
        num_of_sentences = len(self._sentences_score)
        damping_factor = params["damping_factor"]

        cosine_matrix = self._cosine_matrix[:]
        cosine_matrix = [[cosine_matrix[i][j] / sum([cosine_matrix[k][j] for k in range(num_of_sentences)])
                          for j in range(num_of_sentences)]
                         for i in range(num_of_sentences)]
        cosine_matrix = [np.divide(cosine_matrix[i], sum(cosine_matrix[i])) for i in range(num_of_sentences)]

        uniform_matrix = np.full(shape=[num_of_sentences, num_of_sentences], fill_value=1 / num_of_sentences)
        transition_matrix = np.multiply(uniform_matrix, damping_factor)
        transition_matrix += np.multiply((1 - damping_factor), cosine_matrix)

        initial_state = np.full(shape=num_of_sentences, fill_value=1 / num_of_sentences)

        def generate_lexrank(old_state):
            return np.dot(transition_matrix, old_state)

        lexrank_scores = TextSummarizer.__power_method(initial_state, generate_lexrank, settings.threshold)
        for i in range(num_of_sentences):
            self._sentences_score[i][settings.ranking_mode["name"]] = float("{0:.6f}".format(lexrank_scores[i]))

    def __divrank(self):
        """
        Diverse Rank: Computes the divrank for the corresponding given cosine matrix. A novel ranking algorithm
        based on a reinforced random walk in an information network.

        DivRank: the Interplay of Prestige and Diversity in Information Networks
            Qiaozhu Mei     qmei@umich.edu
            Jian Guo        guojian@umich.edu
            Dragomir Radev  radev@umich.edu
            School of Information, Department of EECS, Department of Statistics
            University of Michigan, Ann Arbor, MI 48109 USA
        You can check the algorithm at https://pdfs.semanticscholar.org/0ba4/ab185a116f11286460463357720853fa81a7.pdf
        """

        settings = self._settings
        params = settings.parameters_of(settings.Rank.DIVRANK)
        num_of_sentences = len(self._sentences_score)

        alpha_value = params["alpha_value"]
        beta_value = params["beta_value"]
        lambda_value = params["lambda_value"]

        cosine_matrix = self._cosine_matrix[:]
        cosine_matrix = [[(1 if cosine_matrix[i][j] > params["cos_threshold"] else 0) for j in range(num_of_sentences)]
                         for i in range(num_of_sentences)]
        cosine_matrix = [np.divide(cosine_matrix[i], sum(cosine_matrix[i])) for i in range(num_of_sentences)]

        organic_matrix = [[(1-lambda_value) if i == j else (alpha_value*cosine_matrix[i][j])
                           for j in range(num_of_sentences)]
                          for i in range(num_of_sentences)]

        # track Nt --------------------------------------------------------------------------------------------
        # identity_matrix = np.diag([1 for _ in range(num_of_sentences)])
        # pprint((identity_matrix - cosine_matrix))
        # fundamental_matrix = np.linalg.inv((identity_matrix - cosine_matrix))
        # all_one_vector = np.full(shape=num_of_sentences, fill_value=1)
        # n_visit = np.dot(fundamental_matrix, all_one_vector)
        # pprint(n_visit)

        prior_distribution = [np.power(i + 1, beta_value * -1) if beta_value else (1 / num_of_sentences)
                              for i in range(num_of_sentences)]

        transition_matrix = np.ndarray(shape=(num_of_sentences, num_of_sentences), dtype=float)
        n_visit = np.full(shape=num_of_sentences, fill_value=1)
        for i in range(num_of_sentences):
            for j in range(num_of_sentences):
                reinforced_walk_sum = sum([organic_matrix[i][k] * n_visit[k] for k in range(num_of_sentences)])
                reinforced_walk = (organic_matrix[i][j] * n_visit[j]) / reinforced_walk_sum
                transition_matrix[i, j] = ((1 - lambda_value) * prior_distribution[j]) + (
                        lambda_value * reinforced_walk)
                if cosine_matrix[i][j]:
                    n_visit[j] += 1
        initial_state = np.full(shape=num_of_sentences, fill_value=1 / num_of_sentences)

        def generate_divrank(old_state):
            return np.dot(transition_matrix, old_state)

        divrank_scores = TextSummarizer.__power_method(initial_state, generate_divrank, settings.threshold)
        for i in range(num_of_sentences):
            self._sentences_score[i][settings.ranking_mode["name"]] = float("{0:.6f}".format(divrank_scores[i]))

    def __grasshopper(self):
        """
        Graph Random-walk with Absorbing StateS that HOPs among PEaks for Ranking: A partial implementation of the
        Grasshopper, a novel ranking algorithm based on random walks in an absorbing Markov chain which ranks items
        with an emphasis on diversity.

        Improving Diversity in Ranking using Absorbing Random Walks
            Xiaojin Zhu          jerryzhu@cs.wisc.edu
            Andrew B. Goldberg   goldberg@cs.wisc.edu
            Jurgen Van Gaelj     vangael@cs.wisc.edu
            David Andrzejewski   andrzeje@cs.wisc.edu
            Department of Computer Sciences
            University of Wisconsin, Madison, Madison, WI 53705
        You can check the algorithm at http://pages.cs.wisc.edu/~jerryzhu/pub/grasshopper.pdf
        """

        settings = self._settings
        params = settings.parameters_of(settings.Rank.GRASSHOPPER)
        num_of_sentences = len(self._sentences_score)

        alpha_value = params["alpha_value"]
        lambda_value = params["lambda_value"]
        cos_threshold = params["cos_threshold"]

        cosine_matrix = self._cosine_matrix[:]
        cosine_matrix = [[(
            1 if np.dot(cosine_matrix[i], cosine_matrix[j]) / (np.linalg.norm(cosine_matrix[i]) *
                                                               np.linalg.norm(cosine_matrix[j])) > cos_threshold
            else 0) for j in range(num_of_sentences)] for i in range(num_of_sentences)]
        cosine_matrix = [np.divide(cosine_matrix[i], sum(cosine_matrix[i])) for i in range(num_of_sentences)]

        if settings.reranking_mode:
            ranked_sentences = self._sentences_score
            # pprint(ranked_sentences)
            prior_distribution = [ranked_sentences[i][settings.ranking_mode["name"]] for i in range(num_of_sentences)]
            prior_distribution = np.divide(prior_distribution, sum(prior_distribution))
        else:
            prior_distribution = [np.power(i + 1, alpha_value * -1) for i in range(num_of_sentences)]

        all_one_vector = np.full(shape=num_of_sentences, fill_value=1)
        teleporting_random_walk = np.multiply(lambda_value, cosine_matrix)
        teleporting_random_walk += np.multiply((1 - lambda_value), np.outer(all_one_vector, prior_distribution))
        markov_chain_tracker = [i for i in range(num_of_sentences)]

        def absorb_state(index, num_of_ranked):
            sentence_index = markov_chain_tracker.pop(index)
            markov_chain_tracker.insert(0, sentence_index)

            cosine_matrix[index] = [1 if index == i else 0 for i in range(num_of_sentences)]
            sentence_vector = cosine_matrix.pop(index)
            cosine_matrix.insert(0, sentence_vector)

            submatrix_q = [[cosine_matrix[i][j] for j in range(num_of_ranked, num_of_sentences)]
                           for i in range(num_of_ranked, num_of_sentences)]
            identity_matrix = np.diag([1 for _ in range(num_of_ranked, num_of_sentences)])
            fundamental_matrix = np.linalg.inv((identity_matrix - submatrix_q))
            all_one_vector = np.full(shape=num_of_sentences - num_of_ranked, fill_value=1)
            n_visit = np.dot(fundamental_matrix, all_one_vector) / (num_of_sentences - num_of_ranked)
            return n_visit.tolist()

        stationary_distribution = self.__power_method(prior_distribution,
                                                      lambda state: np.dot(teleporting_random_walk, state),
                                                      settings.threshold)

        # pprint(cosine_matrix)
        g1 = stationary_distribution.index(max(stationary_distribution))
        num_of_ranked = 1
        visit_n = absorb_state(g1, num_of_ranked)
        for i in range(1, num_of_sentences):
            # print("Iteration: " + i)
            num_of_ranked += 1
            sentence_index = visit_n.index(max(visit_n))
            sentence_index += num_of_ranked - 1
            visit_n = absorb_state(sentence_index, num_of_ranked)
        markov_chain_tracker = [num_of_sentences - markov_chain_tracker[i] - 1 for i in range(num_of_sentences)]

        for i in range(num_of_sentences):
            self._sentences_score[i][settings.Rank.GRASSHOPPER.name] = (float("{0:.6f}".format(
                    markov_chain_tracker.index(i) / max(markov_chain_tracker)
            )))

    def __maximal_marginal_relevance(self):
        """
        Maximal Marginal Relevance: A diversity based ranking technique used to maximize the relevance and novelty
        in finally retrieved top-ranked items.

        The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries
            Jaime Carbonell jgc@cs.cmu.edu
            Jade Goldstein  jade@cs.cmu.edu
            Language Technologies Institute, Carnegie Mellon University
        You can check the algorithm at http://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
        """

        def reconstruct_cosine_matrix():
            sentences.extend(TextNormalizer(params["query"])().normalized_text)
            tf = TextSummarizer.__build_term_frequency(sentences)
            idf = self.__build_inverse_document_frequency(tf[0], tf[1])
            return self.__build_cosine_matrix(tf[0], idf)

        def compute_mmr(index):
            similarity_scores = [cosine_matrix[index][sentence["index"]] for sentence in mmr_scores]
            maximum_similarity = max(similarity_scores) if similarity_scores else 0
            mmr = params["lambda_value"]*(cosine_matrix[index][-1] - (1 - params["lambda_value"]) * maximum_similarity)
            return mmr

        def normalize_mmr_scores():
            min_sent = min(mmr_scores, key=lambda sentence: sentence[mmr_name])
            max_sent = max(mmr_scores, key=lambda sentence: sentence[mmr_name])
            min_mmr = min_sent[mmr_name]
            max_mmr = max_sent[mmr_name]
            for sentence in mmr_scores:
                sentence[mmr_name] = float("{0:.6f}".format((sentence[mmr_name] - min_mmr) / (max_mmr - min_mmr)))
                ranked_sentences.append(sentence)

        settings = self._settings
        params = settings.parameters_of(settings.Rerank.MAXIMAL_MARGINAL_RELEVANCE)
        sentences = self._normalized_text[:]
        ranked_sentences = self._sentences_score[:]

        cosine_matrix = reconstruct_cosine_matrix()
        mmr_name = self._settings.reranking_mode["name"]
        mmr_scores = list()

        while ranked_sentences:
            sentence = max(ranked_sentences, key=lambda sentence: sentence[settings.ranking_mode["name"]])
            sentence[mmr_name] = compute_mmr(sentence["index"])
            mmr_scores.append(sentence)
            ranked_sentences.remove(sentence)

        normalize_mmr_scores()
        self._sentences_score = ranked_sentences

    @staticmethod
    def __build_term_frequency(sentences: List[NormalizedSentence], n_gram: int = 1) -> Tuple[WordVector, WordDictionary]:
        # initialize word_dictionary set
        word_dictionary = set()
        for sentence in sentences:
            for i in range(len(sentence) - n_gram + 1):
                word = ""
                for j in range(n_gram):
                    word += sentence[i + j]
                    if j != n_gram - 1:
                        word += " "
                word_dictionary.add(word)

        # count the frequency with the corresponding n-gram model
        word_vector = {i: {word: 0 for word in word_dictionary} for i in range(len(sentences))}
        for i in range(len(sentences)):
            for j in range(len(sentences[i]) - n_gram + 1):
                word = ""
                for k in range(n_gram):
                    word += sentences[i][j + k]
                    if k != n_gram - 1:
                        word += " "
                word_vector[i][word] += 1
        return word_vector, word_dictionary

    @staticmethod
    def __build_inverse_document_frequency(tf: WordVector, dictionary: WordDictionary) -> Dict[Word, float]:
        total_docs = len(tf)
        doc_frequency = {word: 0.0 for word in dictionary}
        for row in tf:
            for word in tf[row]:
                if tf[row][word] != 0:
                    doc_frequency[word] += 1

        inv_frequency = {word: np.log(float(total_docs) / doc_frequency[word]) for word in dictionary}
        return inv_frequency

    @staticmethod
    def __build_cosine_matrix(tf: WordVector, idf: Dict[Word, float]) -> List[List[float]]:

        # TODO PERFORM AN MATRIX OPERATION FOR OPTIMIZATION
        def idf_modified_cosine(x, y):
            # numerator = 0
            # summation_x = summation_y = 0
            dictionary = tf[x]

            numerator = np.sum([tf[x][word] * tf[y][word] * np.square(idf[word]) for word in dictionary])
            summation_x = np.sum([np.square(tf[x][word] * idf[word]) for word in dictionary])
            summation_y = np.sum([np.square(tf[y][word] * idf[word]) for word in dictionary])
            # for word in dictionary:
            #     numerator += tf[x][word] * tf[y][word] * np.square(idf[word])
            #     summation_x += np.square(tf[x][word] * idf[word])
            #     summation_y += np.square(tf[y][word] * idf[word])

            denominator = np.sqrt(summation_x) * np.sqrt(summation_y)
            idf_cosine = numerator / denominator
            idf_cosine = float("{0:.6f}".format(idf_cosine))
            return idf_cosine

        num_of_sentences = len(tf)
        cosine_matrix = [
            [
                float("{0:.6f}".format(
                    np.sum([tf[i][word] * tf[j][word] * np.square(idf[word]) for word in tf[i]]) /
                    (np.sqrt(np.sum([np.square(tf[i][word] * idf[word]) for word in tf[i]])) *
                     np.sqrt(np.sum([np.square(tf[j][word] * idf[word]) for word in tf[i]])))
                ))
                # idf_modified_cosine(i, j)
             for j in range(num_of_sentences)
             ]
            for i in range(num_of_sentences)
        ]
        return cosine_matrix

    @staticmethod
    def __power_method(initial_state: np.ndarray, generate_state_function: Callable[[np.ndarray], np.ndarray],
                       threshold: float) -> List[float]:
        """
        Computes for the stationary distribution of a given Markov chain or transition matrix

        :param initial_state: the first state for initializing the power method
        :param generate_state_function: generate the next distribution of the transition matrix
        :param threshold: [0, 1] the value in which the method will converge
        :return: the stationary distribution of the matrix
        """

        new_state = generate_state_function(initial_state)
        new_state = np.divide(new_state, max(new_state))
        norm = np.linalg.norm(np.subtract(new_state, initial_state))

        while norm > threshold:
            old_state = new_state
            new_state = generate_state_function(old_state)
            new_state = np.divide(new_state, max(new_state))
            norm = np.linalg.norm(np.subtract(new_state, old_state))
        return new_state.tolist()

    @property
    def word_vector(self):
        return self._word_vector

    @property
    def word_dictionary(self):
        return self._word_dictionary

    @property
    def inverse_document_frequency(self):
        return self._idf

    @property
    def cosine_similarity_matrix(self):
        return self._cosine_matrix

    @property
    def sentences_score(self):
        return self._sentences_score

    @property
    def summary_text(self):
        return self._summary_text

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value: "TextSummarizer.Settings"):
        self._settings = value

    class Settings:
        class Rank(Enum):
            DIVRANK = "D"
            LEXRANK = "L"
            GRASSHOPPER = "G"

        class Rerank(Enum):
            GRASSHOPPER = "G"
            MAXIMAL_MARGINAL_RELEVANCE = "M"

        def __init__(self, ranking_mode: Rank = Rank.DIVRANK, reranking_mode: Rerank = None, query: str = None):
            Rank = self.__class__.Rank
            Rerank = self.__class__.Rerank

            self._reranking_mode = None
            self._ranking_mode = ranking_mode
            self._ranking_map = {
                "threshold": 0.001,
                Rank.DIVRANK.name: {
                    "constant_parameters": None,
                    "setter": self.set_divrank_parameters
                },
                Rank.LEXRANK.name: {
                    "constant_parameters": None,
                    "setter": self.set_lexrank_parameters
                },
                Rank.GRASSHOPPER.name: {
                    "constant_parameters": None,
                    "setter": self.set_grasshopper_parameters
                },
                Rerank.MAXIMAL_MARGINAL_RELEVANCE.name: {
                    "constant_parameters": None,
                    "setter": self.set_maximal_marginal_relevance_parameters
                }
            }
            # setting parameters for the divrank algorithm as a "secondary default"
            # since the default grasshopper tends to have some internal errors in rare cases, e.g. Singular Matrix
            self._ranking_map[Rank.DIVRANK.name]["setter"]()
            self._ranking_map[ranking_mode.name]["setter"]()

            if reranking_mode:
                if reranking_mode.name == ranking_mode.name == Rerank.GRASSHOPPER.name:
                    return
                self._reranking_mode = reranking_mode
                if query and reranking_mode.name == Rerank.MAXIMAL_MARGINAL_RELEVANCE.name:
                    self._ranking_map[reranking_mode.name]["setter"](query)
                else:
                    self._ranking_map[reranking_mode.name]["setter"]()

        def __str__(self):
            return ("Rank Mode: " + self._ranking_mode.name
                    + "\nParameters: "
                    + str([str(key) + " => " + str(value)
                           for key, value in self.parameters_of(self._ranking_mode).items()])
                    + "\nRerank Mode: " + (self._reranking_mode.name if self._reranking_mode else "None")
                    + "\nParameters: "
                    + str([str(key) + " => " + str(value)
                           for key, value in self.parameters_of(self._reranking_mode).items()]
                          if self._reranking_mode else None))

        def set_divrank_parameters(self, lambda_value: float = 0.9, alpha_value: float = 0.25, beta_value: float = None,
                                   cos_threshold: float = 0.1) -> type(None):
            self._ranking_map[self.__class__.Rank.DIVRANK.name]["constant_parameters"] = {
                "lambda_value": lambda_value,
                "alpha_value": alpha_value,
                "beta_value": beta_value,
                "cos_threshold": cos_threshold
            }

        def set_lexrank_parameters(self, damping_factor: float = 0.85, cos_threshold: float = 0.1) -> type(None):
            self._ranking_map[self.__class__.Rank.LEXRANK.name]["constant_parameters"] = {
                "damping_factor": damping_factor,
                "cos_threshold": cos_threshold
            }

        def set_grasshopper_parameters(self, lambda_value: float = 0.5, alpha_value: float = 0.25,
                                       cos_threshold: float = 0.1) -> type(None):
            self._ranking_map[self.__class__.Rank.GRASSHOPPER.name]["constant_parameters"] = {
                "lambda_value": lambda_value,
                "alpha_value": alpha_value,
                "cos_threshold": cos_threshold
            }

        def set_maximal_marginal_relevance_parameters(self, query: str, lambda_value: float = 0.7) -> type(None):
            self._ranking_map[self.__class__.Rerank.MAXIMAL_MARGINAL_RELEVANCE.name]["constant_parameters"] = {
                "lambda_value": lambda_value,
                "query": query
            }

        def parameters_of(self, rank: Union[Rank, Rerank]) -> Optional[Dict[str, Union[float, int]]]:
            return self._ranking_map[rank.name]["constant_parameters"]

        def __get_rank(self, name: str) -> Dict[str, Union[str, Dict]]:
            return {
                "name": name,
                "constant_parameters": self._ranking_map[name]["constant_parameters"]
            }

        @property
        def threshold(self):
            return self._ranking_map["threshold"]

        @property
        def ranking_mode(self):
            return self.__get_rank(self._ranking_mode.name)

        @property
        def reranking_mode(self):
            if self._reranking_mode:
                return self.__get_rank(self._reranking_mode.name)
            return None

        @threshold.setter
        def threshold(self, value: float):
            self._ranking_map["threshold"] = value

        @ranking_mode.setter
        def ranking_mode(self, value: Rank):
            self._ranking_mode = value
            self._ranking_map[self._ranking_mode.name]["setter"]()

        @reranking_mode.setter
        def reranking_mode(self, value: Rerank):
            self._reranking_mode = value
            self._ranking_map[self._reranking_mode.name]["setter"]()


Rank = TextSummarizer.Settings.Rank
Rerank = TextSummarizer.Settings.Rerank

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


document2 = """
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
"""

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

# tn = TextNormalizer(document3)
# tn = tn()
# pprint(tn.normalized_text)
# tsSettings = TextSummarizer.Settings(Rank.GRASSHOPPER, Rerank.MAXIMAL_MARGINAL_RELEVANCE, "iphone")
# summarizer = TextSummarizer(tn, tsSettings)

# cProfile.run("summarizer(summary_length=5)", "summarizer")
# p = pstats.Stats("summarizer")
# p.strip_dirs().sort_stats("cumulative").print_stats(10)
# p.sort_stats('time').print_stats(10)
