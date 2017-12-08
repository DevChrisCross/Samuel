from Normalize import normalize_text
from Translator import translate
from Summarizer import summarizer
from extractor import corpus_classfication


def api(data):
    corpus = data['corpus']
    # corpus = translate(corpus)

    # For Summarizer
    summary_length = data['summary_length']
    threshold = 0.001 if 'threshold' not in data else data['threshold']
    rank = "D" if 'rank' not in data else data['rank']
    rerank = False if 'rerank' not in data else data['rerank']
    query = None if 'query' not in data else data['query']
    sort_score = False if 'sort_score' not in data else data['sort_score']
    split_sent = False if 'split_sent' not in data else data['split_sent']
    correct_sent = False if 'correct_sent' not in data else data['correct_sent']
    tokenize_sent = True if 'tokenize_sent' not in data else data['tokenize_sent']

    normalized_corpus = normalize_text(corpus)
    summarized_corpus = summarizer(corpus, summary_length, threshold, rank, rerank, query, sort_score, split_sent,
                                   correct_sent, tokenize_sent)

    polarity = corpus_classfication(normalized_corpus['tokens'])

    return {
        'summarized_text': summarized_corpus['text'],
        'polarity': polarity
    }

# kingsman = "Another witty and fun, action film by Matthew Vaughn. " \
#            "Taron Egerton and Colin Firth come together once more to recreate their fantastic spy duo. " \
#            "Though the movie lacks in parts of what it promised in the trailers and details of what " \
#            "many fans may have hoped. " \
#            "It begrudgingly makes up for though its eye-popping visuals, " \
#            "great new characters, and a clear believable storyline of Harry Hart's revival. Kingsman: " \
#            "The Golden Circle is a near pitch perfect sequel to relive the entertainment of Kingsman: " \
#            "The Secret Service."
#
# kingsman2 = u"Just an overall fun experience. " \
#             u"I still favor the first one a bit, but this was definitely a worthy successor in my opinion."
#
# print(process(kingsman, 3))
