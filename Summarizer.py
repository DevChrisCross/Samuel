import re
import numpy as np
# import numpy
import string
import nltk
from pprint import pprint
import warnings
import Normalize
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def term_frequency(sentences, n_gram=1):
    """
    Feature extractor using the Bag-of-Words model

    :param sentences: array. the normalized array of sentences which contains tokenized words.
    :param n_gram: the gram model in which the feature extraction will be performed.
        (e.g. Unigram, Bigram, Trigram)
    :return: {word vector, word dictionary} dictionary. the word vector that contains
        the features of the sentences, and the dictionary built for the feature extraction
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

    return {
        "word_vector": word_vector,
        "word_dictionary": word_dictionary
    }


def inverse_document_frequency(tf, dictionary):
    """
    Feature Extraction partially from the known TF-IDF model

    :param tf: the term frequency matrix used and referenced for the idf
    :param dictionary: the dictionary used by the term frequency matrix
    :return: array. the inverse document frequency of each word in the dictionary
    """

    total_docs = len(tf)
    doc_frequency = {word: 0.0 for word in dictionary}
    for row in tf:
        for word in tf[row]:
            if tf[row][word] != 0:
                doc_frequency[word] += 1

    inv_frequency = {word: np.log(float(total_docs) / doc_frequency[word]) for word in dictionary}
    return inv_frequency


def build_cosine_matrix(sent_length, tf, idf):
    """
    Constructs a idf modified cosine similarity matrix

    :param sent_length: the number of the sentences involved
    :param tf: the term frequency matrix of the sentences involved
    :param idf: the inverse document frequency array of the sentences involved
    :return: the idf modified cosine similarity matrix
    """

    def idf_modified_cosine(x, y, tf, idf):
        """
        Computes idf modified cosine similarity value of two sentences

        :param x: the index of the first sentence relevant to tf and idf
        :param y: the index of the second sentence relevant to tf and idf
        :param tf: the term frequency matrix
        :param idf: the inverse document frequency
        :return: the idf modified cosine similarity value
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

    cosine_matrix = [[idf_modified_cosine(i, j, tf, idf) for j in range(sent_length)] for i in range(sent_length)]
    return cosine_matrix


def power_method(transition_matrix, initial_state, generate_state=None, threshold=0.001):
    """
    Computes for the stationary distribution of a given Markov chain or transition matrix

    :param transition_matrix: the matrix to be computed to
    :param initial_state: the first state for initializing the power method
    :param generate_state: if a function is supplied, it will be used instead to generate the next distribution
        of the transition matrix
    :param threshold: [0, 1] the value in which the method will converge
    :return: the stationary distribution of the matrix
    """

    new_state = (generate_state(initial_state, transition_matrix) if generate_state
                 else np.dot(transition_matrix, initial_state))
    new_state = np.divide(new_state, max(new_state))
    norm = np.linalg.norm(np.subtract(new_state, initial_state))

    while norm > threshold:
        old_state = new_state
        new_state = (generate_state(old_state, transition_matrix) if generate_state
                     else np.dot(transition_matrix, old_state))
        new_state = np.divide(new_state, max(new_state))
        norm = np.linalg.norm(np.subtract(new_state, old_state))
    return new_state.tolist()


def lexrank(sentences, cosine_matrix, damping_factor=0.85, threshold=0.001):
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

    def generate_lexrank(old_state, cosine_matrix):
        __length = len(old_state)
        new_state = np.zeros(shape=__length)
        for i in range(__length):
            summation_j = 0
            for j in range(__length):
                summation_k = sum(cosine_matrix[j])
                summation_j += (old_state[j] * cosine_matrix[i][j]) / summation_k
            new_state[i] = (damping_factor / __length) + ((1 - damping_factor) * summation_j)
        return new_state

    __length = len(sentences)
    initial_state = np.full(shape=__length, fill_value=1/__length)
    lexrank_scores = power_method(cosine_matrix, initial_state, generate_state=generate_lexrank, threshold=threshold)
    for i in range(__length):
        sentences[i]["lexrank_score"] = float("{0:.3f}".format(lexrank_scores[i]))

    return sentences


def divrank(sentences, cosine_matrix, threshold=0.001, lambda_value=0.9, alpha_value=0.25, beta_value=None,
            cos_threshold=0.1):
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

    :param sentences: the normalized array of sentences
    :param cosine_matrix: the cosine matrix to be ranked by Diverse Rank
    :param threshold: [0, 1]. the threshold value for the algorithm
    :param lambda_value: [0, 1] the convergence value for the algorithm
    :param alpha_value: [0, 1] the value from which the organic transition probability is produced
    :param beta_value: [0, 1] the value from which the prior transition probability is produced
    :param cos_threshold: the cosine threshold in which the cosine matrix becomes a binary cosine matrix
    :return: the sentences with their corresponding divrank scores
    """

    def generate_divrank(old_state, cosine_matrix):

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

    __length = len(sentences)
    cosine_matrix = [[(1 if cosine_matrix[i][j] > cos_threshold else 0) for j in range(__length)]
                     for i in range(__length)]
    cosine_matrix = [np.divide(cosine_matrix[i], np.sum(cosine_matrix[i])) for i in range(__length)]

    initial_state = np.full(shape=__length, fill_value=1/__length)
    divrank_scores = power_method(cosine_matrix, initial_state, generate_state=generate_divrank, threshold=threshold)
    for i in range(__length):
        sentences[i]["divrank_score"] = float("{0:.3f}".format(divrank_scores[i]))

    return sentences


def grasshopper(ranked_sentences, cosine_matrix, scorebase=None, lambda_value=0.5, alpha_value=0.25, cos_threshold=0.1):
    """
    Graph Random-walk with Absorbing StateS that HOPs among PEaks for Ranking
    A partial implementation of the Grasshopper, a novel ranking algorithm based on random walks in an absorbing Markov
    chain which ranks items with an emphasis on diversity.

    Improving Diversity in Ranking using Absorbing Random Walks
        Xiaojin Zhu          jerryzhu@cs.wisc.edu
        Andrew B. Goldberg   goldberg@cs.wisc.edu
        Jurgen Van Gaelj     vangael@cs.wisc.edu
        David Andrzejewski   andrzeje@cs.wisc.edu
        Department of Computer Sciences
        University of Wisconsin, Madison, Madison, WI 53705
    You can check the algorithm at http://pages.cs.wisc.edu/~jerryzhu/pub/grasshopper.pdf

    :param ranked_sentences: an array of sentences that can be either ranked or unranked
    :param cosine_matrix: the cosine matrix to be ranked by Grasshopper
    :param scorebase: string. if scorebase has a value, the sentences will be considered ranked as input
    :param lambda_value: [0, 1] the value to balance the interpolation in the cosine matrix
    :param alpha_value: [0, 1] the value from which the prior distribution is produced if the sentences happens
        to be unranked
    :param cos_threshold: the cosine threshold in which the cosine matrix becomes a binary cosine matrix
    :return: the sentences with their corresponding rank on the grasshopper
    """

    __length = len(ranked_sentences)
    prior_distribution = ([ranked_sentences[i][scorebase] for i in range(__length)] if scorebase
                          else [np.power(i+1, alpha_value*-1) for i in range(__length)])
    vector_ones = np.full(shape=__length, fill_value=1)
    distribution_matrix = np.outer(vector_ones, prior_distribution)
    distribution_matrix = np.multiply(distribution_matrix, (1-lambda_value))

    cosine_matrix = [[(1 if cosine_matrix[i][j] > cos_threshold else 0) for j in range(__length)]
                     for i in range(__length)]
    cosine_matrix = [np.divide(cosine_matrix[i], np.sum(cosine_matrix[i])) for i in range(__length)]
    cosine_matrix = np.multiply(cosine_matrix, lambda_value)
    cosine_matrix = np.add(cosine_matrix, distribution_matrix).tolist()

    grasshopper_scores = list()
    rank_iteration = int(np.ceil(__length/2))
    distribution = np.full(shape=__length, fill_value=1/__length)

    for i in range(rank_iteration):
        stationary_distribution = power_method(cosine_matrix, distribution)
        highest_score = stationary_distribution.index(max(stationary_distribution))
        grasshopper_scores.append(highest_score)
        cosine_matrix.pop(grasshopper_scores[i])

        absorbing_markov = [(1 if j == grasshopper_scores[i] else 0) for j in range(__length)]
        cosine_matrix.insert(grasshopper_scores[i], absorbing_markov)
        distribution = stationary_distribution

    for i in range(__length):
        ranked_sentences[i]["ghopper_score"] = (len(grasshopper_scores) - grasshopper_scores.index(i)
                                        if i in grasshopper_scores else -1)
    ranked_sentences = sorted(ranked_sentences, key=lambda sentence: sentence["ghopper_score"], reverse=True)
    return ranked_sentences


def maximal_marginal_relevance(sentences, ranked_sentences, query, scorebase, lambda_value=0.7):
    """
    Maximal Marginal Relevance
    A diversity based ranking technique used to maximize the relevance
    and novelty in finally retrieved top-ranked items.

    The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries
        Jaime Carbonell jgc@cs.cmu.edu
        Jade Goldstein  jade@cs.cmu.edu
        Language Technologies Institute, Carnegie Mellon University
    You can check the algorithm at http://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

    :param sentences: the normalized sentences
    :param ranked_sentences: array. {score, index} a dictionary of ranked list performed by an IR system.
    :param query: string. a text where mmr would be referenced to
    :param scorebase: string. determines what scoring system shall be used for mmr
    :param lambda_value: a decimal value ranging [0, 1]
        Users wishing to sample the information space around the query, should set this at a smaller value, and
        those wishing to focus in on multiple potentially overlapping or reinforcing relevant documents,
        should set this to a larger value
    :return: {mmr} the ranked sentences with an additional mmr key attribute
    """

    def compute_mmr(x, mmr_scores, cosine_matrix):
        """Computes the mmr of a sentence in respect to the existing subset mmr_scores

        :param x: the index of the sentences with the maximum score
        :param mmr_scores: the subset containg the processed sentences
        :param cosine_matrix: the idf modified cosine similarity matrix
        :return: the maximal marginal relevance value of the sentence
        """

        similarity_scores = [cosine_matrix[x][sentence["index"]] for sentence in mmr_scores]
        maximum_similarity = max(similarity_scores) if similarity_scores else 0
        mmr = lambda_value * (cosine_matrix[x][-1] - (1 - lambda_value) * maximum_similarity)

        return mmr

    query = Normalize.normalize_text(query)
    sentences.append(query["normalized"][0])
    tf = term_frequency(sentences)
    idf = inverse_document_frequency(tf["word_vector"], tf["word_dictionary"])
    cosine_matrix = build_cosine_matrix(len(sentences), tf["word_vector"], idf)

    mmr_scores = list()
    while ranked_sentences:
        sentence = max(ranked_sentences, key=lambda sentence: sentence[scorebase])
        sentence["mmr_score"] = compute_mmr(sentence["index"], mmr_scores, cosine_matrix)
        mmr_scores.append(sentence)
        ranked_sentences.remove(sentence)

    min_sent = min(mmr_scores, key=lambda sentence: sentence["mmr_score"])
    max_sent = max(mmr_scores, key=lambda sentence: sentence["mmr_score"])
    min_mmr = min_sent["mmr_score"]
    max_mmr = max_sent["mmr_score"]
    for sentence in mmr_scores:
        sentence["mmr_score"] = float("{0:.3f}".format((sentence["mmr_score"] - min_mmr) / (max_mmr - min_mmr)))
        ranked_sentences.append(sentence)

    return ranked_sentences


def extract_keyphrase(text, n_gram=2, keywords=4, correct_sent=False, tokenize_sent=True):
    """Extracts significant keywords or keyphrases that represents the idea of the entire text

    :param text: string. the text to bextracted the info from
    :param n_gram: the gram model to be used for collocations
    :param keywords: the top number of keywords to return
    :param correct_sent: parameters for the normalize module
    :param tokenize_sent: parameters for the normalize module
    :return: an array of keyphrases
    """

    def word_similarity(words1, words2):
        words1 = [letter.lower() for letter in words1 if letter != " "]
        words2 = [letter.lower() for letter in words2 if letter != " "]

        words = words1 + words2
        character_vector = {i: {letter.lower(): 0 for letter in words} for i in range(2)}
        for index, word in enumerate([words1, words2]):
            for letter in word:
                character_vector[index][letter] += 1

        c_vector = [[character_vector[i][letter] for letter in character_vector[i]] for i in range(2)]
        similarity_n = np.dot(c_vector[0], c_vector[1])
        similarity_d = np.sqrt(sum(np.square(c_vector[0]))) * np.sqrt(sum(np.square(c_vector[1])))
        similarity_score = similarity_n / similarity_d

        return similarity_score

    sentences = Normalize.normalize_text(text, None, tokenize_sent, correct_sent)
    tf = term_frequency(sentences["normalized"], n_gram)
    word_dict = {word: 0 for word in tf["word_dictionary"]}
    for sentence in tf["word_vector"]:
        for word in tf["word_vector"][sentence]:
            word_dict[word] += tf["word_vector"][sentence][word]
    collocations = sorted(word_dict.items(), key=lambda word: word[1], reverse=True)
    collocations = collocations[:keywords]

    raw_sentences = nltk.sent_tokenize(text) if tokenize_sent else text
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in raw_sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    chunked_sentences = list(nltk.ne_chunk_sents(tagged_sentences))

    sc_regex_string = "[{}]".format(re.escape(string.punctuation))
    sc_regex_compiled = re.compile(pattern=sc_regex_string)
    word = 0
    tag = 1
    formed_noun = ""

    phrase_tag_list = ["DT", "JJ", "NN", "NNS", "NNP", "NNPS"]
    noun_phrases = list()
    for i in range(len(chunked_sentences)):
        for j in range(len(chunked_sentences[i])):
            chunked_word = chunked_sentences[i][j]
            if hasattr(chunked_word, "label"):
                formed_noun += (" " if formed_noun else "") + (" ".join([child[0] for child in chunked_word]))
            else:
                is_special_character = sc_regex_compiled.sub(string=chunked_word[word], repl="") == ""
                if not is_special_character or chunked_word[word] == "." or chunked_word[word] == ",":
                    if chunked_word[tag] in phrase_tag_list:
                        formed_noun += (" " if formed_noun else "") + chunked_word[word]
                    else:
                        if formed_noun:
                            noun_phrases.append(formed_noun)
                            formed_noun = ""

    formed_keyphrases = dict()
    for keyphrase in collocations:
        top_phrases = {phrase: word_similarity(keyphrase[0], phrase) for phrase in noun_phrases}
        top_phrases = sorted(top_phrases.items(), key=lambda word: word[1], reverse=True)
        formed_keyphrases[keyphrase[0]] = top_phrases[0]

    return formed_keyphrases


def summarizer(corpus, summary_length, threshold=0.001, rank="D", rerank=False, query=None, sort_score=False,
               split_sent=False, correct_sent=False, tokenize_sent=True):
    """
    Summarizes a document using the specified ranking algorithm set by the user.

    :param corpus: the document to be summarized
    :param summary_length: the number of sentences needed in the summary
    :param threshold: the threshold value for the stationary distribution of the algorithm
    :param rank: ['D','L','G'] namely DivRank, LexRank, Grasshopper
        decides which ranking algorithm will be used for the summarization
    :param rerank: boolean. if enabled, the module will used a reranking algorithm, Grasshopper by default
    :param query: string. supplying a query automatically uses a reranking algorithm, the the Maximal Marginal Relevance
        and overrides the Grasshopper if rerank is enabled
    :param sort_score: boolean. if the sentences should be sorted by appearance or score
    :param split_sent: boolean. if the output should be an array of sentences or an entire string
    :param correct_sent: boolean. if the text normalization module should perform a word correcting
    :param tokenize_sent: boolean. if the text input should be tokenize into sentences
        It should be set to false if the text input is an array of sentences
    :return: {text, score} a dictionary that returns the text summary, and the corresponding scores of the sentences
    """

    keywords = extract_keyphrase(corpus, correct_sent=correct_sent, tokenize_sent=tokenize_sent)
    sentences = Normalize.normalize_text(corpus, None, tokenize_sent, correct_sent)
    tf = term_frequency(sentences["normalized"])
    idf = inverse_document_frequency(tf["word_vector"], tf["word_dictionary"])
    cosine_matrix = build_cosine_matrix(len(sentences["normalized"]), tf["word_vector"], idf)

    summary_scores = [{
        "index": i,
        "raw_text": sentences["raw"][i].replace('\n', "").capitalize(),
        "norm_text": ",".join(sentences["normalized"][i])
    } for i in range(len(sentences["raw"]))]

    rank_map = {
        "D": {"system": "divrank_score", "process": divrank(summary_scores, cosine_matrix, threshold)},
        "L": {"system": "lexrank_score", "process": lexrank(summary_scores, cosine_matrix, threshold)},
        "G": {"system": "ghopper_score", "process": grasshopper(summary_scores, cosine_matrix)}
    }
    scorebase = rank_map[rank]["system"]
    summary_scores = rank_map[rank]["process"]
    summary_scores = (grasshopper(summary_scores, cosine_matrix, scorebase)
                      if rerank and rank_map[rank] != "G" and not query
                      else summary_scores)
    summary_scores = (maximal_marginal_relevance(sentences["normalized"], summary_scores, query, scorebase)
                      if query and rank_map[rank] != "G"
                      else summary_scores)

    sort_criteria = "mmr_score" if query else scorebase
    summary_scores = sorted(summary_scores, key=lambda sentence: sentence[sort_criteria], reverse=True)
    summary_scores = summary_scores[:summary_length]
    summary_scores = sorted(summary_scores, key=lambda sentence: sentence[sort_criteria if sort_score else "index"],
                            reverse=sort_score)
    summary_text = [sentence["raw_text"] for sentence in summary_scores]
    summary_text = " ".join(summary_text) if not split_sent else summary_text

    return {
        "text": summary_text,
        "score": summary_scores,
        "keywords": keywords
    }


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

pprint(summarizer(document2, summary_length=5, rank="G"))
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
