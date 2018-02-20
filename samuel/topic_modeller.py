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
        _length = len(normalized_text)

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