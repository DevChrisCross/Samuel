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

import math
import numpy
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

def document_frequency(sent_array, dictionary, smoothing = True):
    doc_frequency = {}
    inv_frequency = {}

    # document frequency
    for word in dictionary:
        doc_frequency[word] = (1 if smoothing else 0)
        inv_frequency[word] = 0.0

    for sentence in sent_array:
        for word in sentence:
            if word in dictionary:
                doc_frequency[word] += 1
                break

    # inverse document frequency
    total_docs = len(sent_array) + (1 if smoothing else 0)
    for word in dictionary:
        inv_frequency[word] = (1.0 if smoothing else 0.0) + numpy.log(float(total_docs) / doc_frequency[word])

    return doc_frequency, inv_frequency

def idf_modified_cosine(sentX_index, sentY_index, tf_matrix, idf):
    numerator = denominator = 0
    summation_X = summation_Y = 0
    dictionary = tf_matrix[sentX_index]

    for word in dictionary:
        numerator += tf_matrix[sentX_index][word] * tf_matrix[sentY_index][word] * (idf[word]) ** 2
        summation_X += (tf_matrix[sentX_index][word] * (idf[word])) ** 2
        summation_Y += (tf_matrix[sentY_index][word] * (idf[word])) ** 2

    denominator = math.sqrt(summation_X) * math.sqrt(summation_Y)
    idf_cosine = numerator / denominator

    return idf_cosine

def init_cosine_matrix(sent_array, cos_threshold, tf_matrix, idf):
    sent_length = len(sent_array)
    sent_degree = numpy.zeros(shape=(sent_length))
    cosine_matrix = numpy.zeros(shape=(sent_length, sent_length))

    # establish cosine matrix with the indicated threshold
    for i in range(sent_length):
        for j in range(sent_length):
            cosine_matrix[i][j] = idf_modified_cosine(sent_array[i], sent_array[j], tf_matrix, idf)
            if cosine_matrix[i][j] > cos_threshold:
                cosine_matrix[i][j] = 1
                sent_degree[i] += 1
            else:
                cosine_matrix[i][j] = 0

    # divide the value by degrees for vertex voting
    for i in range(sent_length):
        for j in range(sent_length):
            cosine_matrix[i][j] /= sent_degree[i]

    return cosine_matrix

def compute_lexrank(cosine_matrix, nodeU, damping_factor = 0.85):
    sent_length = len(cosine_matrix)
    lexrank_value = 0
    summation_adjnodeU = 0

    # compute the needed summation of vertices adjacent to the node U
    for nodeV in range(sent_length):
        summation_adjnodeV = 0
        for nodeZ in range(sent_length):
            if cosine_matrix[nodeV][nodeZ] > 0:
                summation_adjnodeV += cosine_matrix[nodeV][nodeZ]
        summation_adjnodeU += ((cosine_matrix[nodeU][nodeV] / summation_adjnodeV) * compute_lexrank(cosine_matrix, nodeV))

    lexrank_value = (damping_factor / sent_length) + (1 - damping_factor) * summation_adjnodeU
    return lexrank_value

def continuous_lexrank(sent_array, sent_limit, cos_threshold = 0.1):
    # feature extraction: bag of words
    tf_matrix, dictionary = term_frequency(sent_array)

    # feature extraction: df, idf
    doc_frequency, inv_frequency = document_frequency(sent_array, dictionary)

    # document similarity: idf-modified-cosine
    cosine_matrix = init_cosine_matrix(sent_array, cos_threshold, tf_matrix, inv_frequency)

    # initiate lexrank score for each sentence
    sent_length = len(sent_array)
    lexrank_score = numpy.zeros(shape=(sent_length))
    lexrank_score = [compute_lexrank(cosine_matrix, nodeU) for nodeU in range(sent_length)]
    # lexrank_score = sorted(iterable=lexrank_score, reverse=True)
    return lexrank_score

sent_array = [
    ["The", "main", "operations", "on", "a", "dictionary", "are", "storing", "a", "value"],
    ["with", "some", "key", "and", "extracting", "the", "value", "given", "the", "key"]
]
