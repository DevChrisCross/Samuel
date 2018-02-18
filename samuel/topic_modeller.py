import json
import pyLDAvis.gensim
from gensim.models import LdaModel, Phrases
from gensim.corpora import Dictionary
from warnings import filterwarnings
filterwarnings(action='ignore', category=UserWarning, module='gensim')


class TextTopicModeller:
    def __init__(self, normalized_text, visualize=False):
        _id = id(self)
        _name = self.__class__.__name__

        print(_name, _id, "Establishing bigrams")
        bi_gram = Phrases(normalized_text)
        normalized_text = [bi_gram[line] for line in normalized_text]

        print(_name, _id, "Establishing word dictionary")
        dictionary = Dictionary(normalized_text)
        dictionary_words = [dictionary.doc2bow(text) for text in normalized_text]

        print(_name, _id, "Constructing LDA model")
        lda_model = LdaModel(corpus=dictionary_words, num_topics=10, id2word=dictionary)

        if visualize:
            filename = 'visualization.json'
            visualization = pyLDAvis.gensim.prepare(lda_model, dictionary_words, dictionary)
            pyLDAvis.save_json(visualization, filename)
            with open(filename) as json_data:
                visual = json.load(json_data)
                self._topics = visual
        else:
            self._topics = lda_model.get_topics()
        print(_name, _id, "Topic modelling done")

    @property
    def topics(self):
        return self._topics


if __name__ == "__main__":
    pass
# def __extract_keyphrase(text, n_gram=2, keywords=4):
#
#     def word_similarity(words1, words2):
#         words1 = [letter.lower() for letter in words1 if letter != " "]
#         words2 = [letter.lower() for letter in words2 if letter != " "]
#
#         words = words1 + words2
#         character_vector = {i: {letter.lower(): 0 for letter in words} for i in range(2)}
#         for index, word in enumerate([words1, words2]):
#             for letter in word:
#                 character_vector[index][letter] += 1
#
#         c_vector = [[character_vector[i][letter] for letter in character_vector[i]] for i in range(2)]
#         similarity_n = np.dot(c_vector[0], c_vector[1])
#         similarity_d = np.sqrt(sum(np.square(c_vector[0]))) * np.sqrt(sum(np.square(c_vector[1])))
#         similarity_score = similarity_n / similarity_d
#
#         return similarity_score
#
#     sentences = Normalize.normalize_text(text, None, tokenize_sent, correct_sent)
#     tf = __term_frequency(sentences["normalized"], n_gram)
#     word_dict = {word: 0 for word in tf["word_dictionary"]}
#     for sentence in tf["word_vector"]:
#         for word in tf["word_vector"][sentence]:
#             word_dict[word] += tf["word_vector"][sentence][word]
#     collocations = sorted(word_dict.items(), key=lambda word: word[1], reverse=True)
#     collocations = collocations[:keywords]
#
#     raw_sentences = nltk.sent_tokenize(text) if tokenize_sent else text
#     tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in raw_sentences]
#     tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
#     chunked_sentences = list(nltk.ne_chunk_sents(tagged_sentences))
#
#     sc_regex_string = "[{}]".format(re.escape(string.punctuation))
#     sc_regex_compiled = re.compile(pattern=sc_regex_string)
#     word = 0
#     tag = 1
#     formed_noun = ""
#
#     phrase_tag_list = ["DT", "JJ", "NN", "NNS", "NNP", "NNPS"]
#     noun_phrases = list()
#     for i in range(len(chunked_sentences)):
#         for j in range(len(chunked_sentences[i])):
#             chunked_word = chunked_sentences[i][j]
#             if hasattr(chunked_word, "label"):
#                 formed_noun += (" " if formed_noun else "") + (" ".join([child[0] for child in chunked_word]))
#             else:
#                 is_special_character = sc_regex_compiled.sub(string=chunked_word[word], repl="") == ""
#                 if not is_special_character or chunked_word[word] == "." or chunked_word[word] == ",":
#                     if chunked_word[tag] in phrase_tag_list:
#                         formed_noun += (" " if formed_noun else "") + chunked_word[word]
#                     else:
#                         if formed_noun:
#                             noun_phrases.append(formed_noun)
#                             formed_noun = ""
#
#     formed_keyphrases = dict()
#     for keyphrase in collocations:
#         top_phrases = {phrase: word_similarity(keyphrase[0], phrase) for phrase in noun_phrases}
#         top_phrases = sorted(top_phrases.items(), key=lambda word: word[1], reverse=True)
#         formed_keyphrases[keyphrase[0]] = top_phrases[0]
#
#     return formed_keyphrases
