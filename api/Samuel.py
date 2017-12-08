from Normalize import normalize_text
from Translator import translate
from Summarizer import summarizer
from extractor import corpus_classfication


def process(raw_corpus, summary_length,query=""):
    # corpus = translate(raw_corpus)
    corpus = raw_corpus
    normalized_corpus = normalize_text(corpus)
    summarized_corpus = summarizer(corpus, summary_length,query=query)

    polarity = corpus_classfication(normalized_corpus['tokens'])

    return {
        'summarized_text': summarized_corpus['text'],
        'polarity': polarity
    }

#
# kingsman = "Another witty and fun, action film by Matthew Vaughn. " \
#            "Taron Egerton and Colin Firth come together once more to recreate their fantastic spy duo. " \
#            "Though the movie lacks in parts of what it promised in the trailers and details of what many fans may have hoped. " \
#            "It begrudgingly makes up for though its eye-popping visuals, " \
#            "great new characters, and a clear believable storyline of Harry Hart's revival. Kingsman: " \
#            "The Golden Circle is a near pitch perfect sequel to relive the entertainment of Kingsman: The Secret Service."
#
# kingsman2 = u"Just an overall fun experience. " \
#             u"I still favor the first one a bit, but this was definitely a worthy successor in my opinion."
#
# print(process(kingsman, 3))
