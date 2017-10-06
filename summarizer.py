'''
    Continuous LexRank (Lexical PageRank)
    Description: A improved revised algorithm for PageRank & TextRank for handling text summarization
    Reference:
        LexRank: Graph-based Lexical Centrality as Salience in Text Summarization
            Güneş Erkan         -- gerkan@umich.edu
            Dragomir R. Radev   -- radev@umich.edu
            Department of EECS, School of Information: University of Michigan, Ann Arbor, MI 48109 USA
        https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html
'''

import re
import math
import numpy
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from pprint import pprint

def term_frequency(sent_array, n_gram = 1):
    tf_vector = dict()
    dictionary_set = set()

    # establish dictionary set
    for sentence in sent_array:
        for i in range(len(sentence) - n_gram + 1):
            word = ""
            for j in range(n_gram):
                word += sentence[i+j]
                if j != n_gram-1:
                    word += " "
            dictionary_set.add(word)

    # initialize tf_vector with empty frequencies
    for i in range(len(sent_array)):
        tf_vector[i] = dict()
        for word in dictionary_set:
            tf_vector[i][word] = 0

    # count the frequency with the N-gram model
    for i in range(len(sent_array)):
        for j in range(len(sent_array[i]) - n_gram + 1):
            word = ""
            for k in range(n_gram):
                word += sent_array[i][j + k]
                if k != n_gram - 1:
                    word += " "
            tf_vector[i][word] += 1

    return tf_vector, dictionary_set

def document_frequency(tf_matrix, dictionary, smoothing = True):
    doc_frequency = {}
    inv_frequency = {}

    # document frequency
    for word in dictionary:
        doc_frequency[word] = (1 if smoothing else 0)
        inv_frequency[word] = 0.0

    for row in range(len(tf_matrix)):
        for word in tf_matrix[row]:
            if tf_matrix[row].get(word) != 0:
                doc_frequency[word] += 1

    # inverse document frequency
    total_docs = len(tf_matrix) + (1 if smoothing else 0)
    print("total docs:", total_docs)
    for word in dictionary:
        inv_frequency[word] = (1.0 if smoothing else 0.0) + numpy.log(float(total_docs) / doc_frequency[word])
        # print(word, inv_frequency[word])
        # print(doc_frequency[word])

    return doc_frequency, inv_frequency

def idf_modified_cosine(sentX_index, sentY_index, tf_matrix, idf):
    numerator = 0
    summation_X = summation_Y = 0
    dictionary = tf_matrix[sentX_index]

    for word in dictionary:
        if tf_matrix[sentX_index][word] and tf_matrix[sentY_index][word]:
            numerator += tf_matrix[sentX_index][word] * tf_matrix[sentY_index][word] * (idf[word] ** 2)
        summation_X += (tf_matrix[sentX_index][word] * (idf[word])) ** 2
        summation_Y += (tf_matrix[sentY_index][word] * (idf[word])) ** 2
    # numerator = float("{0:.2f}".format(numerator))
    denominator = math.sqrt(summation_X) * math.sqrt(summation_Y)
    # denominator = float("{0:.2f}".format(denominator))
    idf_cosine = numerator / denominator
    # print("_____________")
    # print(numerator)
    # print(summation_X)
    # print(summation_Y)
    # print(denominator)
    idf_cosine = float("{0:.2f}".format(idf_cosine))
    return idf_cosine

def init_cosine_matrix(sent_array, tf_matrix, idf, cos_threshold = 0.2):
    sent_length = len(sent_array)
    sent_degree = numpy.zeros(shape=(sent_length))
    cosine_matrix = numpy.zeros(shape=(sent_length, sent_length))

    # establish cosine matrix with the indicated threshold
    for i in range(sent_length):
        for j in range(sent_length):
            cosine_matrix[i][j] = idf_modified_cosine(i, j, tf_matrix, idf)
            # if cosine_matrix[i][j] > cos_threshold:
            #     # continue
            #     cosine_matrix[i][j] = 1
            #     sent_degree[i] += 1
            # else:
            #     cosine_matrix[i][j] = 0

    # divide the value by degrees for vertex voting
    # for i in range(sent_length):
    #     for j in range(sent_length):
    #         cosine_matrix[i][j] /= sent_degree[i]
    # for i in range(sent_length):
    #     norm1 = sum(cosine_matrix[i])
    #     for j in range(sent_length):
    #         cosine_matrix[i][j] /= norm1

    return cosine_matrix

def power_method(cosine_matrix, matrix_size, threshold = 0.2, damping_factor = 0.85):
    print("initiated stoch")
    def generate_eigen_vector(eigen_vector):
        print(eigen_vector)
        eigen_vectorN = numpy.zeros(shape=matrix_size)
        for i in range(matrix_size):
            summation_j = 0
            for j in range(matrix_size):
                summation_k = 0
                for k in range(matrix_size):
                    summation_k += cosine_matrix[j][k]
                summation_j += eigen_vector[j] * (cosine_matrix[i][j] / summation_k)
            eigen_vectorN[i] = ((1 - damping_factor) / matrix_size) + (damping_factor * (summation_j))

        return eigen_vectorN

    ctr = 0
    eigen_vector = []
    eigen_vector.append(numpy.zeros(shape=matrix_size))
    for i in range(matrix_size):
        eigen_vector[ctr][i] = 1/matrix_size
    ctr += 1
    eigen_vector.append(generate_eigen_vector(eigen_vector[ctr-1]))
    delta = numpy.linalg.norm(numpy.subtract(eigen_vector[ctr], eigen_vector[ctr-1]), ord=2)

    print(delta, threshold)
    # while delta > threshold:
    while delta != 0:
        ctr += 1
        eigen_vector[ctr] = generate_eigen_vector(eigen_vector[ctr-1])
        delta = numpy.linalg.norm(numpy.subtract(eigen_vector[ctr], eigen_vector[ctr-1]), ord=2)
        print(delta, threshold)
    return eigen_vector[ctr]

def normalize_sent(corpus):
    corpus = corpus.lower()
    sentences = nltk.sent_tokenize(corpus)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_en.extend(["n't", "'s", "'d", "'t", "'ve", "'ll"])

    filtered_sentences = []
    sc_regex_string = "[{}]".format(re.escape(string.punctuation))
    sc_regex_compiled = re.compile(pattern=sc_regex_string)
    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            if word in stopwords_en:
                continue
            word = sc_regex_compiled.sub(string=word, repl="")
            if word != "":
                new_sentence.append(word)
        filtered_sentences.append(new_sentence)

    # wordnet_lemmatizer = WordNetLemmatizer()
    # annotated_sentences = [nltk.pos_tag(sentence, tagset="universal") for sentence in filtered_sentences]

    return filtered_sentences

def continuous_lexrank(sent_array, sent_limit):
    # feature extraction: bag of words
    tf_matrix, dictionary = term_frequency(sent_array)
    # for tf in tf_matrix:
    #     print(tf_matrix.get(tf))

    # feature extraction: df, idf
    doc_frequency, inv_frequency = document_frequency(tf_matrix, dictionary)
    print(doc_frequency)
    # print(inv_frequency)

    # document similarity: idf-modified-cosine
    cosine_matrix = init_cosine_matrix(sent_array, tf_matrix, inv_frequency)
    for i in range(len(cosine_matrix)):
        print(cosine_matrix[i])
        # print(numpy.round(cosine_matrix[i], 2))

    # initiate lexrank score for each sentence
    sent_length = len(sent_array)
    lexrank = power_method(cosine_matrix, sent_length)
    print("LexrANkL ")
    print(lexrank)
    # lexrank_score = [compute_lexrank(cosine_matrix, nodeU) for nodeU in range(sent_length)]
    # lexrank_score = sorted(iterable=lexrank_score, reverse=True)
    return lexrank


document = """
The Elder Scrolls V: Skyrim is an open world action role-playing video game developed by Bethesda Game Studios and published by Bethesda Softworks.
It is the fifth installment in The Elder Scrolls series, following The Elder Scrolls IV: Oblivion. 
Skyrim's main story revolves around the player character and their effort to defeat Alduin the World-Eater, a dragon who is prophesied to destroy the world.
The game is set two hundred years after the events of Oblivion and takes place in the fictional province of Skyrim.
The player completes quests and develops the character by improving skills.
Skyrim continues the open world tradition of its predecessors by allowing the player to travel anywhere in the game world at any time, and to ignore or postpone the main storyline indefinitely. 
The player may freely roam over the land of Skyrim, which is an open world environment consisting of wilderness expanses, dungeons, cities, towns, fortresses and villages. 
Players may navigate the game world more quickly by riding horses, or by utilizing a fast-travel system which allows them to warp to previously Players have the option to develop their character. 
At the beginning of the game, players create their character by selecting one of several races, including humans, orcs, elves and anthropomorphic cat or lizard-like creatures, and then customizing their character's appearance, discovered locations. 
Over the course of the game, players improve their character's skills, which are numerical representations of their ability in certain areas. 
There are eighteen skills divided evenly among the three schools of combat, magic, and stealth.
Skyrim is the first entry in The Elder Scrolls to include Dragons in the game's wilderness. 
Like other creatures, Dragons are generated randomly in the world and will engage in combat.
"""
# 1, 12, 6

document = """
    iraqi vice president taha yassi ramadan announced today, sunday, that iraq refuses to back down from its decision to stop cooperating with disarmament inspectors before its demands are met.
iraqi vice president taha yassin ramadan announced today, thursday, that iraq rejects cooperating with the united nations except on the issue of lifting the blockade imposed upon it since the year 1990.
ramadan told reporters in baghdad that "iraq cannot deal positively with whoever represents the security council unless there was a clear stance on the issue of lifting the blockade off of it.
baghdad had decided late last october to completely cease cooperating with the inspectors of the united nations special commision (unscom), in charge of disarming iraq's weapons, and whose work became very limited since the fifth of august, and announced it will not resume its cooperation with the commission even if it were subjected to a military operation.
the russian foreign minister, igor ivanov, warned today, wednesday against using force against iraq, which will destroy, according to him, seven years of difficult diplomatic work and will complicate the regional situation in the area.
ivanov contended that carrying out air strikes against irq, who refuses to cooperate with the united nations inspectors, "will end the tremendous work achieved by the international group during the past seven years and will complicate the situation in the region."
nevertheless, ivanov stressed that baghdad must resume working with the special commission in charge of disarming the iraqi weapons of mass destruction (unscom).
the special representative of the united nations secretary-general in baghdad, prakash shah, announced today, wednesday, after meeting with the iraqi deputy prime minister tariq aziz, that iraq refuses to back down from its decision to cut off cooperation with the disarmament inspectors.
british prime minister tony blair said today, sunday, that the crisis between the international community and iraq "did not end" and that britain is still ready, prepared, and able to strike iraq."
in a gathering with the press held at the prime minister's office, blair contended that the crisis with iraq " will not end until iraq has absolutely and unconditionally respected its commitments" towards the united nations.
a spokesman for tony blair had indicated that the british prime minister gave permission to british air force tornado planes stationed to kuwait to join the aerial bombardment against iraq.
"""
# print(nltk.corpus.stopwords.words('english'))
# pprint(normalize_sent(document))
text = normalize_sent(document)

print(text)
continuous_lexrank(text, 3)


