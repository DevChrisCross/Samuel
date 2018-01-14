import re
import numpy as np
import string
import nltk
from pprint import pprint
import warnings
from typing import Callable, Tuple, List, Dict, Set
from Normalize import TextNormalizer
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Summarizer:
    def __init__(self, normalized_text: "TextNormalizer", settings: "Summarizer.Settings"):
        self._text = normalized_text.original_text
        self._raw_text = normalized_text.raw_text
        self._normalized_text = normalized_text.normalized_text

        self._word_vector, self._word_dictionary = Summarizer.__build_term_frequency(self._normalized_text)
        self._idf = Summarizer.__build_inverse_document_frequency(self._word_vector, self._word_dictionary)
        self._cosine_matrix = Summarizer.__build_cosine_matrix(self._word_vector, self._idf)

        self._summary_text = None
        self._summary_scores = [{
            "index": i,
            "raw_text": self._raw_text[i],
            "norm_text": self._normalized_text[i]
        } for i in range(len(self._raw_text))]

        self._settings = settings

    def summarizer(self, summary_length: int, threshold: float = 0.001, rank_mode: str = "D", rerank: bool = False,
                   query: str = None, sort_by_score: bool = False) -> "Summarizer":
        """
        Summarizes a document using the specified ranking algorithm set by the user.

        :param summary_length: the number of sentences needed in the summary
        :param threshold: the threshold value for the stationary distribution of the algorithm
        :param rank_mode: ['D','L','G'] namely DivRank, LexRank, Grasshopper
            decides which ranking algorithm will be used for the summarization
        :param rerank: boolean. if enabled, the module will used a reranking algorithm, Grasshopper by default
        :param query: string. supplying a query automatically uses a reranking algorithm, the the Maximal Marginal Relevance
            and overrides the Grasshopper if rerank is enabled
        :param sort_by_score: boolean. if the sentences should be sorted by appearance or score

        """

        rank_map = {
            "D": {"name": "divrank_score", "function": self.__divrank},
            "L": {"name": "lexrank_score", "function": self.__lexrank},
            "G": {"name": "ghopper_score", "function": self.__grasshopper},
            "M": {"name": "mmr_score", "function": self.__maximal_marginal_relevance}
        }

        scorebase = rank_map[rank_mode]["name"]
        invoke_function = rank_map[rank_mode]["function"]
        invoke_function(threshold)

        if rerank:
            if query and not scorebase == "ghopper_score":
                self.__maximal_marginal_relevance(query=query, scorebase=scorebase)
                scorebase = rank_map["M"]["name"]
            else:
                if not scorebase == "ghopper_score":
                    self.__grasshopper(scorebase=scorebase)
                    scorebase = rank_map["G"]["name"]
        self._summary_scores.sort(key=lambda sentence: sentence[scorebase], reverse=True)

        _summary = self._summary_scores[:summary_length]
        scorebase = scorebase if sort_by_score else "index"
        _summary.sort(key=lambda sentence: sentence[scorebase], reverse=sort_by_score)
        summary_text = " ".join([sentence["raw_text"] for sentence in _summary])
        self._summary_text = summary_text
        return self

    @classmethod
    def create_summarizer(cls, normalized_text: "TextNormalizer") -> "Summarizer":
        return cls(normalized_text, Summarizer.Settings())

    def __lexrank(self, threshold: float = 0.001, damping_factor: float = 0.85):
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

        :param sentences: the array in which the lexrank score of each sentence will be stored
        :param cosine_matrix: the cosine matrix to be ranked by Lexical PageRank
        :param threshold: [0, 1]. the threshold value for the algorithm
        :param damping_factor: [0, 1] the convergence value for the algorithm
        :return: array of sentences and their corresponding lexrank score
        """

        num_of_sentences = len(self._summary_scores)
        cosine_matrix = self._cosine_matrix[:]
        initial_state = np.full(shape=num_of_sentences, fill_value=1 / num_of_sentences)

        def generate_lexrank(old_state):
            __length = len(old_state)
            new_state = np.zeros(shape=__length)
            for i in range(__length):
                summation_j = 0
                for j in range(__length):
                    summation_k = sum(cosine_matrix[j])
                    summation_j += (old_state[j] * cosine_matrix[i][j]) / summation_k
                new_state[i] = (damping_factor / __length) + ((1 - damping_factor) * summation_j)
            return new_state

        lexrank_scores = Summarizer.__power_method(initial_state, generate_lexrank, threshold)
        for i in range(num_of_sentences):
            self._summary_scores[i]["lexrank_score"] = float("{0:.3f}".format(lexrank_scores[i]))

    def __divrank(self, threshold: float = 0.001, lambda_value: float = 0.9, alpha_value: float = 0.25,
                  beta_value: float = None, cos_threshold: float = 0.1):
        """
        Diverse Rank
        Computes the divrank for the corresponding given cosine matrix.
        A novel ranking algorithm based on a reinforced random walk in an information network.

        DivRank: the Interplay of Prestige and Diversity in Information Networks
            Qiaozhu Mei     qmei@umich.edu
            Jian Guo        guojian@umich.edu
            Dragomir Radev  radev@umich.edu
            School of Information, Department of EECS, Department of Statistics
            University of Michigan, Ann Arbor, MI 48109 USA
        You can check the algorithm at https://pdfs.semanticscholar.org/0ba4/ab185a116f11286460463357720853fa81a7.pdf

        :param threshold: [0, 1]. the threshold value for the algorithm
        :param lambda_value: [0, 1] the convergence value for the algorithm
        :param alpha_value: [0, 1] the value from which the organic transition probability is produced
        :param beta_value: [0, 1] the value from which the prior transition probability is produced
        :param cos_threshold: the cosine threshold in which the cosine matrix becomes a binary cosine matrix
        :return: the sentences with their corresponding divrank scores
        """

        num_of_sentences = len(self._summary_scores)
        cosine_matrix = self._cosine_matrix[:]

        cosine_matrix = [[(1 if cosine_matrix[i][j] > cos_threshold else 0) for j in range(num_of_sentences)]
                         for i in range(num_of_sentences)]
        cosine_matrix = [np.divide(cosine_matrix[i], np.sum(cosine_matrix[i])) for i in range(num_of_sentences)]
        initial_state = np.full(shape=num_of_sentences, fill_value=1 / num_of_sentences)

        def generate_divrank(old_state):

            def organic_value(x, y):
                return (1 - alpha_value) if x == y else (alpha_value * cosine_matrix[x][y])

            __length = len(old_state)
            new_state = np.zeros(shape=__length)
            visited_n = np.full(shape=__length, fill_value=1)

            for i in range(__length):
                summation_j = 0
                for j in range(__length):
                    summation_k = 0
                    for k in range(__length):
                        summation_k += (organic_value(j, k) * old_state[k] * visited_n[k])
                        if organic_value(j, k):
                            visited_n[k] += 1
                    summation_j += (old_state[j] * ((organic_value(j, i) * visited_n[i]) / summation_k))
                    if organic_value(j, i):
                        visited_n[i] += 1
                prior_distribution = np.power(i+1, beta_value*-1) if beta_value else (1/__length)
                new_state[i] = ((1 - lambda_value) * prior_distribution) + (lambda_value * summation_j)
            return new_state

        divrank_scores = Summarizer.__power_method(initial_state, generate_divrank, threshold)
        for i in range(num_of_sentences):
            self._summary_scores[i]["divrank_score"] = float("{0:.3f}".format(divrank_scores[i]))

    def __grasshopper(self, threshold: float = 0.001, scorebase=None, lambda_value=0.5, alpha_value=0.25,
                      cos_threshold=0.1):
        """
        Graph Random-walk with Absorbing StateS that HOPs among PEaks for Ranking
        A partial implementation of the Grasshopper, a novel ranking algorithm based on random walks in an absorbing
        Markov chain which ranks items with an emphasis on diversity.

        Improving Diversity in Ranking using Absorbing Random Walks
            Xiaojin Zhu          jerryzhu@cs.wisc.edu
            Andrew B. Goldberg   goldberg@cs.wisc.edu
            Jurgen Van Gaelj     vangael@cs.wisc.edu
            David Andrzejewski   andrzeje@cs.wisc.edu
            Department of Computer Sciences
            University of Wisconsin, Madison, Madison, WI 53705
        You can check the algorithm at http://pages.cs.wisc.edu/~jerryzhu/pub/grasshopper.pdf

        :param scorebase: string. if scorebase has a value, the sentences will be considered ranked as input
        :param lambda_value: [0, 1] the value to balance the interpolation in the cosine matrix
        :param alpha_value: [0, 1] the value from which the prior distribution is produced if the sentences happens
            to be unranked
        :param cos_threshold: the cosine threshold in which the cosine matrix becomes a binary cosine matrix
        :return: the sentences with their corresponding rank on the grasshopper
        """

        num_of_sentences = len(self._summary_scores)
        ranked_sentences = self._summary_scores
        cosine_matrix = self._cosine_matrix[:]

        prior_distribution = ([ranked_sentences[i][scorebase] for i in range(num_of_sentences)] if scorebase
                              else [np.power(i+1, alpha_value*-1) for i in range(num_of_sentences)])
        vector_ones = np.full(shape=num_of_sentences, fill_value=1)
        distribution_matrix = np.outer(vector_ones, prior_distribution)
        distribution_matrix = np.multiply(distribution_matrix, (1-lambda_value))

        cosine_matrix = [[(1 if cosine_matrix[i][j] > cos_threshold else 0) for j in range(num_of_sentences)]
                         for i in range(num_of_sentences)]
        cosine_matrix = [np.divide(cosine_matrix[i], np.sum(cosine_matrix[i])) for i in range(num_of_sentences)]
        cosine_matrix = np.multiply(cosine_matrix, lambda_value)
        cosine_matrix = np.add(cosine_matrix, distribution_matrix).tolist()

        grasshopper_scores = list()
        rank_iteration = int(np.ceil(num_of_sentences/2))
        distribution = np.full(shape=num_of_sentences, fill_value=1/num_of_sentences)

        def generate_grank(old_state):
            return np.dot(self._cosine_matrix, old_state)

        for i in range(rank_iteration):
            stationary_distribution = self.__power_method(distribution, generate_grank, threshold)
            highest_score = stationary_distribution.index(max(stationary_distribution))
            grasshopper_scores.append(highest_score)
            cosine_matrix.pop(grasshopper_scores[i])

            absorbing_markov = [(1 if j == grasshopper_scores[i] else 0) for j in range(num_of_sentences)]
            cosine_matrix.insert(grasshopper_scores[i], absorbing_markov)
            distribution = stationary_distribution

        for i in range(num_of_sentences):
            self._summary_scores[i]["ghopper_score"] = (len(grasshopper_scores) - grasshopper_scores.index(i)
                                                        if i in grasshopper_scores else -1)

    def __maximal_marginal_relevance(self, query, scorebase, lambda_value=0.7):
        """
        Maximal Marginal Relevance
        A diversity based ranking technique used to maximize the relevance
        and novelty in finally retrieved top-ranked items.

        The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries
            Jaime Carbonell jgc@cs.cmu.edu
            Jade Goldstein  jade@cs.cmu.edu
            Language Technologies Institute, Carnegie Mellon University
        You can check the algorithm at http://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

        :param query: string. a text where mmr would be referenced to
        :param scorebase: string. determines what scoring system shall be used for mmr
        :param lambda_value: a decimal value ranging [0, 1]
            Users wishing to sample the information space around the query, should set this at a smaller value, and
            those wishing to focus in on multiple potentially overlapping or reinforcing relevant documents,
            should set this to a larger value
        :return: {mmr} the ranked sentences with an additional mmr key attribute
        """
        sentences = self._normalized_text[:]
        ranked_sentences = self._summary_scores[:]

        def reconstruct_cosine_matrix():
            sentences.extend(TextNormalizer.create_normalizer(query)().normalized_text)
            tf = Summarizer.__build_term_frequency(sentences)
            idf = self.__build_inverse_document_frequency(tf[0], tf[1])
            return self.__build_cosine_matrix(tf[0], idf)

        cosine_matrix = reconstruct_cosine_matrix()

        def compute_mmr(x, mmr_scores):
            """Computes the mmr of a sentence in respect to the existing subset mmr_scores

            :param x: the index of the sentences with the maximum score
            :param mmr_scores: the subset containing the processed sentences
            :return: the maximal marginal relevance value of the sentence
            """

            similarity_scores = [cosine_matrix[x][sentence["index"]] for sentence in mmr_scores]
            maximum_similarity = max(similarity_scores) if similarity_scores else 0
            mmr = lambda_value * (cosine_matrix[x][-1] - (1 - lambda_value) * maximum_similarity)

            return mmr

        mmr_scores = list()
        while ranked_sentences:
            sentence = max(ranked_sentences, key=lambda sentence: sentence[scorebase])
            sentence["mmr_score"] = compute_mmr(sentence["index"], mmr_scores)
            mmr_scores.append(sentence)
            ranked_sentences.remove(sentence)

        min_sent = min(mmr_scores, key=lambda sentence: sentence["mmr_score"])
        max_sent = max(mmr_scores, key=lambda sentence: sentence["mmr_score"])
        min_mmr = min_sent["mmr_score"]
        max_mmr = max_sent["mmr_score"]
        for sentence in mmr_scores:
            sentence["mmr_score"] = float("{0:.3f}".format((sentence["mmr_score"] - min_mmr) / (max_mmr - min_mmr)))
            ranked_sentences.append(sentence)
        self._summary_scores = ranked_sentences

    @staticmethod
    def __build_term_frequency(sentences, n_gram: int = 1) -> Tuple[dict, set]:
        """
        Feature extractor using the Bag-of-Words model
        """
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
    def __build_inverse_document_frequency(tf, dictionary) -> dict:
        """
        Feature Extraction partially from the known TF-IDF modely
        """

        total_docs = len(tf)
        doc_frequency = {word: 0.0 for word in dictionary}
        for row in tf:
            for word in tf[row]:
                if tf[row][word] != 0:
                    doc_frequency[word] += 1

        inv_frequency = {word: np.log(float(total_docs) / doc_frequency[word]) for word in dictionary}
        return inv_frequency

    @staticmethod
    def __build_cosine_matrix(tf, idf) -> list:
        """
        Constructs a idf modified cosine similarity matrix
        """

        num_of_sentences = len(tf)

        def idf_modified_cosine(x, y):
            """
            Computes idf modified cosine similarity value of two sentences
            """

            numerator = 0
            summation_x = summation_y = 0
            dictionary = tf[x]

            for word in dictionary:
                numerator += tf[x][word] * tf[y][word] * np.square(idf[word])
                summation_x += np.square(tf[x][word] * idf[word])
                summation_y += np.square(tf[y][word] * idf[word])

            denominator = np.sqrt(summation_x) * np.sqrt(summation_y)
            idf_cosine = numerator / denominator
            idf_cosine = float("{0:.3f}".format(idf_cosine))
            return idf_cosine

        cosine_matrix = [[idf_modified_cosine(i, j) for j in range(num_of_sentences)] for i in range(num_of_sentences)]
        return cosine_matrix

    @staticmethod
    def __power_method(initial_state: np.ndarray, generate_state_function: Callable[[np.ndarray], np.ndarray],
                       threshold: float) -> list:
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
    def summary_text(self):
        return self._summary_text

    class Settings:
        def __init__(self, ranking_mode: str = "D"):
            self._ranking_mode = ranking_mode
            self._ranking_map = {
                "D": ("divrank", self.set_divrank_parameters),
                "L": ("lexrank", self.set_lexrank_parameters),
                "G": ("ghopper", self.set_grasshopper_parameters),
                "M": ("mmr", self.set_maximal_marginal_relevance_parameters)
            }

            self._constant_parameters = {
                "threshold": 0.001,
                "divrank": None,
                "lexrank": None,
                "ghopper": None,
                "mmr": None
            }
            setup_default_parameters = self._ranking_map[self._ranking_mode][1]
            setup_default_parameters()

        def set_divrank_parameters(self, lambda_value: float = 0.9, alpha_value: float = 0.25, beta_value: float = None,
                                   cos_threshold: float = 0.1) -> type(None):
            self._constant_parameters["divrank"] = {
                "lambda_value": lambda_value,
                "alpha_value": alpha_value,
                "beta_value": beta_value,
                "cos_threshold": cos_threshold
            }

        def set_lexrank_parameters(self, damping_factor: float = 0.85) -> type(None):
            self._constant_parameters["lexrank"] = {
                "damping_factor": damping_factor
            }

        def set_grasshopper_parameters(self, lambda_value: float = 0.5, alpha_value: float = 0.25,
                                       cos_threshold: float = 0.1) -> type(None):
            self._constant_parameters["ghopper"] = {
                "lambda_value": lambda_value,
                "alpha_value": alpha_value,
                "cos_threshold": cos_threshold
            }

        def set_maximal_marginal_relevance_parameters(self, lambda_value: float = 0.7) -> type(None):
            self._constant_parameters["mmr"] = {
                "lambda_value": lambda_value
            }

        @property
        def ranking_mode(self):
            return self._ranking_map[self._ranking_mode][0]

        @property
        def constant_parameters(self):
            return self._constant_parameters

        @ranking_mode.setter
        def ranking_mode(self, value: str):
            self._ranking_mode = value
            self._ranking_map[self._ranking_mode][1]()

document1 = [
    '''iraqi vice president taha yassin ramadan announced today, sunday, that iraq refuses to back down from its decision to stop cooperating with disarmament inspectors before its demands are met.''',
    '''iraqi vice president taha yassin ramadan announced today, thursday, that iraq rejects cooperating with the united nations except on the issue of lifting the blockade imposed upon it since the year 1990.''',
    '''ramadan told reporters in baghdad that "iraq cannot deal positively with whoever represents the security council unless there was a clear stance on the issue of lifting the blockade off of it.''',
    '''baghdad had decided late last october to completely cease cooperating with the inspectors of the united nations special commision (unscom), in charge of disarming iraq's weapons, and whose work became very limited since the fifth of august, and announced it will not resume its cooperation with the commission even if it were subjected to a military operation.''',
    '''the russian foreign minister, igor ivanov, warned today, wednesday against using force against iraq, which will destroy, according to him, seven years of difficult diplomatic work and will complicate the regional situation in the area.''',
    '''ivanov contended that carrying out air strikes against iraq, who refuses to cooperate with the united nations inspectors, "will end the tremendous work achieved by the international group during the past seven years and will complicate the situation in the region."''',
    '''nevertheless, ivanov stressed that baghdad must resume working with the special commission in charge of disarming the iraqi weapons of mass destruction (unscom).''',
    '''the special representative of the united nations secretary-general in baghdad, prakash shah, announced today, wednesday, after meeting with the iraqi deputy prime minister tariq aziz, that iraq refuses to back down from its decision to cut off cooperation with the disarmament inspectors.''',
    '''british prime minister tony blair said today, sunday, that the crisis between the international community and iraq "did not end" and that britain is still ready, prepared, and able to strike iraq."''',
    '''in a gathering with the press held at the prime minister's office, blair contended that the crisis with iraq " will not end until iraq has absolutely and unconditionally respected its commitments" towards the united nations.''',
    '''a spokesman for tony blair had indicated that the british prime minister gave permission to british air force tornado planes stationed to kuwait to join the aerial bombardment against iraq.'''
]

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

tn = TextNormalizer.create_normalizer(document2)
pprint(Summarizer(tn()).summarizer(summary_length=5, rank_mode="D", rerank=True, query="game engine").summary_text)
# pprint(summarizer(document1, summary_length=3, query="War against Iraq", tokenize_sent=False, sort_score=True, drank=True))
# pprint(extract_keyphrase(document2))
# print(numpy.add([5,2,3], [5,3,2]))

from gensim.summarization import summarize
# print(summarize("""The Elder Scrolls V: Skyrim is an open world action role-playing video game developed by Bethesda Game Studios and published by Bethesda Softworks.
# It is the fifth installment in The Elder Scrolls series, following The Elder Scrolls IV: Oblivion.
# Skyrim's main story revolves around the player character and their effort to defeat Alduin the World-Eater, a dragon who is prophesied to destroy the world.
# The game is set two hundred years after the events of Oblivion and takes place in the fictional province of Skyrim.
# The player completes quests and develops the character by improving skills.
# Skyrim continues the open world tradition of its predecessors by allowing the player to travel anywhere in the game world at any time, and to ignore or postpone the main storyline indefinitely.
# The player may freely roam over the land of Skyrim, which is an open world environment consisting of wilderness expanses, dungeons, cities, towns, fortresses and villages.
# Players may navigate the game world more quickly by riding horses, or by utilizing a fast-travel system which allows them to warp to previously Players have the option to develop their character.
# At the beginning of the game, players create their character by selecting one of several races, including humans, orcs, elves and anthropomorphic cat or lizard-like creatures, and then customizing their character's appearance, discovered locations.
# Over the course of the game, players improve their character's skills, which are numerical representations of their ability in certain areas.
# There are eighteen skills divided evenly among the three schools of combat, magic, and stealth.
# Skyrim is the first entry in The Elder Scrolls to include Dragons in the game's wilderness.
# Like other creatures, Dragons are generated randomly in the world and will engage in combat.""",ratio=0.4))
