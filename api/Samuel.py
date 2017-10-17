from Normalize import normalize_text
from Translator import translate
from Summarizer import summarizer
from pprint import pprint


def process(raw_corpus, summary_length):
    corpus = translate(raw_corpus)
    normalized_corpus = normalize_text(corpus)
    summarized_corpus = summarizer(corpus, summary_length)

    return {
        'summarized_text': summarized_corpus['text']
    }


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

# print(process("The food is great but the service is not good. But I enjoy it.", 2))
#
# # print(testing())
