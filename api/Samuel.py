import Normalize
import Translator
import summarizer_v2


def process(raw_corpus, summary_length):
    corpus = ""

    for cor in raw_corpus:
        corpus += cor + " "

    corpus = Translator.translate(corpus)

    corpus, tokens, sentence_n = Normalize.normalize_corpus(corpus)
    normalized_corpus = {'corpus': corpus, 'tokens': tokens, 'sentence_n': sentence_n}
    summarized_corpus = summarizer_v2.summarize(raw_corpus, summary_length)

    return summarized_corpus


kingsman = "Another witty and fun, action film by Matthew Vaughn. " \
           "Taron Egerton and Colin Firth come together once more to recreate their fantastic spy duo. " \
           "Though the movie lacks in parts of what it promised in the trailers and details of what many fans may have hoped. " \
           "It begrudgingly makes up for though its eye-popping visuals, " \
           "great new characters, and a clear believable storyline of Harry Hart's revival. Kingsman: " \
           "The Golden Circle is a near pitch perfect sequel to relive the entertainment of Kingsman: The Secret Service."

kingsman_2 = u"Just an overall fun experience. " \
             u"I still favor the first one a bit, but this was definitely a worthy successor in my opinion."

# print(process(kingsman, 2))

# print(summarizer_v2.summarize(kingsman, 3))
