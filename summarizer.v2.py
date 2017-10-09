import re
import numpy
import string
import nltk
import enchant
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pprint import pprint
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def normalize_text(text, tokenize_sentence=True, correct_spelling=False):
    """A function that performs text normalization.

    Tokenizes sentences and words, removes stopwords and special characters
     performs lemmatization on words, and further remove words which does not
     qualify in the part-of-speech tag map.

    :param text: string. the text to be normalized
    :param tokenize_sentence: boolean. if the text input should be tokenize into sentences
        It should be set to false if the text input is an array of sentences
    :param correct_spelling: boolean. if the spelling of words should be checked and corrected
    :return: {normalized, raw} dictionary. contains the normalized and raw sentences
    """

    def correct_word(word):
        """Corrects the misspelled word

        Corrects the word by removing irregular repeated letters,
        and further corrects it by suggesting possible intended words to be used
        using the PyEnchant library. You can check it at http://pythonhosted.org/pyenchant/api/enchant.html

        :param word: string. the word to be corrected
        :return: string. the corrected word
        """

        def check_word(old_word):
            if wordnet.synsets(old_word):
                return old_word
            else:
                new_word = repeat_regex_compiled.sub(string=old_word, repl=match_substitution)
                new_word = new_word if new_word == old_word else check_word(new_word)
                return new_word

        enchant_dict = enchant.Dict("en_US")
        match_substitution = r'\1\2\3'
        repeat_regex_string = r'(\w*)(\w)\2(\w*)'
        repeat_regex_compiled = re.compile(pattern=repeat_regex_string)

        initial_correct_word = check_word(word)
        word_suggestions = enchant_dict.suggest(initial_correct_word)
        is_word_correct = enchant_dict.check(initial_correct_word)

        if is_word_correct:
            return initial_correct_word
        else:
            final_correct_word = word_suggestions[0] if word_suggestions else initial_correct_word
            return final_correct_word

    raw_sentences = nltk.sent_tokenize(text) if tokenize_sentence else text
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in raw_sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences, tagset="universal")

    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_en.extend(["n't", "'s", "'d", "'t", "'ve", "'ll"])

    sc_regex_string = "[{}]".format(re.escape(string.punctuation))
    sc_regex_compiled = re.compile(pattern=sc_regex_string)

    wordnet_lemmatizer = WordNetLemmatizer()
    word = 0; tag = 1
    pos_tag_map = {
        "ADJ": wordnet.ADJ,
        "VERB": wordnet.VERB,
        "NOUN": wordnet.NOUN,
        "ADV": wordnet.ADV
    }

    normalized_sentences = list()
    for sentence in tagged_sentences:
        new_sentence = list()
        for tagged_word in sentence:
            if tagged_word[word] not in stopwords_en:
                is_special_character = sc_regex_compiled.sub(string=tagged_word[word], repl="") == ""
                tagged_word = None if is_special_character else tagged_word
                if tagged_word and tagged_word[tag] in pos_tag_map:
                    wordnet_tag = pos_tag_map[tagged_word[tag]]
                    lemmatized_word = wordnet_lemmatizer.lemmatize(tagged_word[word], wordnet_tag)
                    lemmatized_word = correct_word(lemmatized_word) if correct_spelling else lemmatized_word
                    new_sentence.append(lemmatized_word.lower())
        normalized_sentences.append(new_sentence)

    return {
        "normalized": normalized_sentences,
        "raw": raw_sentences
    }


def term_frequency(sentences, n_gram=1):
    """Feature extractor using the Bag-of-Words model

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
                word += sentence[i+j]
                if j != n_gram-1:
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
    """Feature Extraction partially from the known TF-IDF model

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

    inv_frequency = {word: numpy.log(float(total_docs) / doc_frequency[word]) for word in dictionary}
    return inv_frequency


def build_cosine_matrix(sent_length, tf, idf):
    """Constructs a idf modified cosine similarity matrix

    :param sent_length: the number of the sentences involved
    :param tf: the term frequency matrix of the sentences involved
    :param idf: the inverse document frequency array of the sentences involved
    :return: the idf modified cosine similarity matrix
    """

    def idf_modified_cosine(x, y, tf, idf):
        """Computes idf modified cosine similarity value of two sentences

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
            numerator += tf[x][word] * tf[y][word] * numpy.square(idf[word])
            summation_x += numpy.square(tf[x][word] * idf[word])
            summation_y += numpy.square(tf[y][word] * idf[word])

        denominator = numpy.sqrt(summation_x) * numpy.sqrt(summation_y)
        idf_cosine = numerator / denominator
        idf_cosine = float("{0:.3f}".format(idf_cosine))

        return idf_cosine

    cosine_matrix = [[idf_modified_cosine(i, j, tf, idf) for j in range(sent_length)] for i in range(sent_length)]
    return cosine_matrix


def lexrank(cosine_matrix, threshold, damping_factor = 0.85):
    """Computes the Lexrank for the corresponding given cosine matrix. A ranking algorithm which involves
       computing sentence importance based on the concept of eigenvector centrality in a graph representation of sentences.

    LexRank: Graph-based Lexical Centrality as Salience in Text Summarization
        Güneş Erkan         gerkan@umich.edu
        Dragomir R. Radev   radev@umich.edu
        Department of EECS, School of Information
        University of Michigan, Ann Arbor, MI 48109 USA
    You can check the algorithm at https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html

    :param cosine_matrix: the cosine matrix to be ranked by Lexical PageRank
    :param threshold: a decimal value ranging [0, 1]. the threshold value for the algorithm
    :param damping_factor: a decimal value ranging [0, 1] the convergence value for the algorithm
    :return: (score, index) a tuple list of the lexrank score and its index in the cosine matrix
    """

    def generate_lexrank(old_lexrank, cosine_matrix):
        """Generates a new array of lexrank score based on the old lexrank

        :param old_lexrank: array. the cuurent list of lexrank scores
        :param cosine_matrix: the cosine matrix
        :return: an array of lexrank scores
        """

        lexrank_length = len(old_lexrank)
        new_lexrank = numpy.zeros(shape=lexrank_length)

        for i in range(lexrank_length):
            summation_j = 0
            for j in range(lexrank_length):
                summation_k = sum(cosine_matrix[j])
                summation_j += old_lexrank[j] * (cosine_matrix[i][j] / summation_k)
            new_lexrank[i] = (damping_factor / lexrank_length) + ((1 - damping_factor) * summation_j)

        return new_lexrank

    initial_lexrank = numpy.zeros(shape=len(cosine_matrix))
    initial_lexrank.fill(1/len(initial_lexrank))
    new_lexrank = generate_lexrank(initial_lexrank, cosine_matrix)
    lexrank_vector = new_lexrank - initial_lexrank
    delta_value = numpy.linalg.norm(lexrank_vector)

    while delta_value > threshold:
        old_lexrank = new_lexrank
        new_lexrank = generate_lexrank(old_lexrank, cosine_matrix)
        lexrank_vector = new_lexrank - old_lexrank
        delta_value = numpy.linalg.norm(lexrank_vector)

    lexrank_scores = [{
        "score": float("{0:.3f}".format(new_lexrank[i] / max(new_lexrank))),
        "index": i
    } for i in range(len(new_lexrank))]
    lexrank_scores = sorted(lexrank_scores, key=lambda sentence: sentence["score"], reverse=True)

    return lexrank_scores


def maximal_marginal_relevance(sentences, ranked_sentences, query, lambda_value=0.7):
    """A diversity based ranking technique used to maximize the relevance and novelty in finally retrieved top-ranked items.

    The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries
        Jaime Carbonell jgc@cs.cmu.edu
        Jade Goldstein  jade@cs.cmu.edu
        Language Technologies Institute, Carnegie Mellon University
    You can check the algorithm at http://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

    :param sentences: the normalized sentences
    :param ranked_sentences: array. {score, index} a dictionary of ranked list performed by an IR system.
    :param query: string. a text where mmr would be referenced to
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

    query = normalize_text(query)
    sentences.append(query["normalized"][0])
    tf = term_frequency(sentences)
    idf = inverse_document_frequency(tf["word_vector"], tf["word_dictionary"])
    cosine_matrix = build_cosine_matrix(len(sentences), tf["word_vector"], idf)

    mmr_scores = list()
    while ranked_sentences:
        sentence = max(ranked_sentences, key=lambda sentence: sentence["score"])
        sentence["mmr"] = compute_mmr(sentence["index"], mmr_scores, cosine_matrix)
        mmr_scores.append(sentence)
        ranked_sentences.remove(sentence)

    min_sent = min(mmr_scores, key=lambda sentence: sentence["mmr"])
    max_sent = max(mmr_scores, key=lambda sentence: sentence["mmr"])
    min_mmr = min_sent["mmr"]
    max_mmr = max_sent["mmr"]
    for sentence in mmr_scores:
        sentence["mmr"] = float("{0:.3f}".format((sentence["mmr"] - min_mmr) / (max_mmr - min_mmr)))
    mmr_scores = sorted(mmr_scores, key=lambda sentence: sentence["mmr"], reverse=True)

    return mmr_scores


def divrank(cosine_matrix, threshold=0.1, damping_factor=0.9, alpha_value=0.25, beta_value=0.3, cos_threshold = 0.1):

    def organic_value(x, y, cosine_matrix):
        return (1 - alpha_value) if x == y else (alpha_value*cosine_matrix[x][y])

    def generate_divrank(old_divrank, cosine_matrix):
        divrank_length = len(old_divrank)
        new_divrank = numpy.zeros(shape=divrank_length)
        visited_n = numpy.zeros(shape=divrank_length)
        visited_n.fill(1)

        for i in range(divrank_length):
            summation_j = 0
            for j in range(divrank_length):
                summation_k = 0
                for k in range(divrank_length):
                    summation_k += (organic_value(j, k, cosine_matrix) * old_divrank[k] * visited_n[k])
                    if organic_value(j, k, cosine_matrix):
                        visited_n[k] += 1
                summation_j += (old_divrank[j] * ((organic_value(j, i, cosine_matrix) * visited_n[i]) / summation_k))
                if organic_value(j, i, cosine_matrix):
                    visited_n[i] += 1
            new_divrank[i] = ((1 - damping_factor) * numpy.power(i + 1, beta_value * -1)) + (damping_factor * summation_j)
        return new_divrank

    divrank_length = len(cosine_matrix)
    node_degree = numpy.zeros(shape=divrank_length)
    for i in range(divrank_length):
        for j in range(divrank_length):
            if cosine_matrix[i][j] > cos_threshold:
                cosine_matrix[i][j] = 1
                node_degree[i] += 1
            else:
                cosine_matrix[i][j] = 0
    cosine_matrix = [[cosine_matrix[i][j]/node_degree[i] for j in range(divrank_length)] for i in range(divrank_length)]

    initial_divrank = numpy.zeros(shape=divrank_length)
    initial_divrank.fill(1 / divrank_length)
    new_divrank = generate_divrank(initial_divrank, cosine_matrix)
    lexrank_vector = new_divrank - initial_divrank
    delta_value = numpy.linalg.norm(lexrank_vector)

    while delta_value > threshold:
        old_divrank = new_divrank
        new_divrank = generate_divrank(old_divrank, cosine_matrix)
        lexrank_vector = new_divrank - old_divrank
        delta_value = numpy.linalg.norm(lexrank_vector)

    divrank_scores = [{
        "score": float("{0:.3f}".format(new_divrank[i] / max(new_divrank))),
        "index": i
    } for i in range(len(new_divrank))]
    divrank_scores = sorted(divrank_scores, key=lambda sentence: sentence["score"], reverse=True)

    return divrank_scores


def initialize_lexrank(corpus, summary_length, threshold=0.1, mmr=False, query=None, orderby_score=False, split_sent=False, correct_sent=False, tokenize_sent=True):
    """Summarizes a document using the the Lexical PageRank Algorithm

        The documentation and option for using the DivRank Algorithm is not yet set.

    :param corpus: the document to be summarized
    :param summary_length: the number of sentences needed in the summary
    :param threshold: the threshold value of the LexRank Algorithm
    :param mmr: boolean. enables the Maximal Marginal Relevance Algorithm
    :param query: string. the query needed to enable mmr
    :param orderby_score: boolean. if the sentences should be sorted by appearance or score
    :param split_sent: boolean. if the output should be an array of sentences or an entire string
    :param correct_sent: boolean. if the text normalization module should perform a word correcting
    :param tokenize_sent: boolean. if the text input should be tokenize into sentences
        It should be set to false if the text input is an array of sentences
    :return: {text, score} a dictionary that returns the text summary, and the corresponding scores of the sentences
    """

    if mmr and not query:
        raise ValueError("You need to provide a query to enable mmr.")

    sentences = normalize_text(corpus, tokenize_sent, correct_sent)
    tf = term_frequency(sentences["normalized"])
    idf = inverse_document_frequency(tf["word_vector"], tf["word_dictionary"])
    cosine_matrix = build_cosine_matrix(len(sentences["normalized"]), tf["word_vector"], idf)

    lexrank_scores = lexrank(cosine_matrix, threshold)
    lexrank_scores = divrank(cosine_matrix, threshold)
    sentence_scores = maximal_marginal_relevance(sentences["normalized"], lexrank_scores, query) if mmr and query else lexrank_scores
    summary_scores = sentence_scores[:summary_length]
    summary_scores = summary_scores if orderby_score else sorted(summary_scores, key=lambda sentence: sentence["index"])

    summary_text = list() if split_sent else ""
    for sentence in summary_scores:
        sentence["text"] = sentences["raw"][sentence["index"]].capitalize().replace('\n', "")
        if split_sent:
            summary_text.append(sentence["text"])
        else:
            summary_text += sentence["text"] + " "

    return {
        "text": summary_text,
        "score": summary_scores
    }


def extract_keyphrase(text, n_gram=2, keywords=4, correct_sent=False, tokenize_sent=True):
    """

    :param text:
    :param n_gram:
    :param keywords:
    :param correct_sent:
    :param tokenize_sent:
    :return:
    """

    def word_similarity(words1, words2):
        """

        :param words1:
        :param words2:
        :return:
        """

        words1 = [letter.lower() for letter in words1 if letter != " "]
        words2 = [letter.lower() for letter in words2 if letter != " "]

        words = words1 + words2
        character_vector = {i: {letter.lower(): 0 for letter in words} for i in range(2)}
        for index, word in enumerate([words1, words2]):
            for letter in word:
                character_vector[index][letter] += 1

        c_vector = [[character_vector[i][letter] for letter in character_vector[i]] for i in range(2)]
        similarity_n = numpy.dot(c_vector[0], c_vector[1])
        similarity_d = numpy.sqrt(sum(numpy.square(c_vector[0]))) * numpy.sqrt(sum(numpy.square(c_vector[1])))
        similarity_score = similarity_n / similarity_d

        return similarity_score

    sentences = normalize_text(text, tokenize_sent, correct_sent)
    tf = term_frequency(sentences["normalized"], n_gram)
    word_dict = {word: 0 for word in tf["word_dictionary"]}
    for sentence in tf["word_vector"]:
        for word in tf["word_vector"][sentence]:
            word_dict[word] += tf["word_vector"][sentence][word]
    collocations = sorted(word_dict.items(), key=lambda word: word[1], reverse=True)
    collocations = collocations[:keywords]

    raw_sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in raw_sentences]
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    chunked_sentences = list(nltk.ne_chunk_sents(tagged_sentences))

    sc_regex_string = "[{}]".format(re.escape(string.punctuation))
    sc_regex_compiled = re.compile(pattern=sc_regex_string)
    word = 0; tag = 1
    formed_noun = ""

    phrase_tag_list = ["DT", "JJ", "NN", "NNS", "NNP", "NNPS"]
    noun_phrases = list()
    for i in range(len(chunked_sentences)):
        # phrases = list()
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
        formed_keyphrases[keyphrase] = top_phrases[0]

    return formed_keyphrases

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
The Elder Scrolls V: Skyrim is an open world action role-playing video game developed by Bethesda Game Studios and published by Bethesda Softworks.
"""

pprint(initialize_lexrank(document2, summary_length=3, mmr=False, query="Elder Scrolls Online"))
pprint(initialize_lexrank(document1, summary_length=3, mmr=False, query="War against Iraq", tokenize_sent=False, orderby_score=True))
# pprint(extract_keyphrase(document2))


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
