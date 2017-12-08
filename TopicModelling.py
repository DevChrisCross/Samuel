import warnings
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim
import json
import gensim
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def topic_modelling(normalized_text, visualize=False):
    bi_gram = gensim.models.Phrases(normalized_text)
    normalized_text = [bi_gram[line] for line in normalized_text]

    dictionary = Dictionary(normalized_text)
    dictionary_words = [dictionary.doc2bow(text) for text in normalized_text]

    lda_model = LdaModel(corpus=dictionary_words, num_topics=10, id2word=dictionary)

    if visualize:
        filename = 'visualization.json'
        __py_lda_vis_visualization(filename, lda_model, dictionary_words, dictionary)
        with open(filename) as json_data:
            visual = json.load(json_data)
            return visual
    return lda_model.get_topics()


def __py_lda_vis_visualization(filename, lda_model, dictionary_words, dictionary):
    if __name__ == '__main__':
        visualization = pyLDAvis.gensim.prepare(lda_model, dictionary_words, dictionary)
        pyLDAvis.save_json(visualization, filename)

# lda = topic_modelling(corpus['normalized'], True)
