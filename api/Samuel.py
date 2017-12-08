from Normalize import normalize_text
from Translator import translate
from Summarizer import summarizer
from extractor import corpus_classfication
import TopicModelling
from pprint import pprint
from warnings import filterwarnings

filterwarnings(action='ignore')


def api(data):
    corpus = data['corpus']
    # corpus = translate(corpus)
    normalized_corpus = normalize_text(corpus)

    # For Topic Modelling
    visualize = False if 'visualize' not in data else data['visualize']

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

    dashboard = build_dashboard(TopicModelling.topic_modelling(normalized_corpus['normalized'], visualize))
    summarized_corpus = summarizer(corpus, summary_length, threshold, rank, rerank, query, sort_score, split_sent,
                                   correct_sent, tokenize_sent)
    polarity = corpus_classfication(normalized_corpus['tokens'])
    return {
        'summarized_text': summarized_corpus['text'],
        'polarity': polarity,
        'dashboard': dashboard
    }


def build_dashboard(data):
    dashboard_head = '<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">\n' \
                     '<div id="ldavis_el70603330337764279074628"></div>\n'
    dashboard_script1 = '<script type="text/javascript">\n' \
                        'var ldavis_el70603330337764279074628_data = ' + str(data) + ';\n'
    dashboard_script2 = 'function LDAvis_load_lib(url, callback){' \
                        'var s = document.createElement("script");' \
                        's.src = url;' \
                        's.async = true;' \
                        's.onreadystatechange = s.onload = callback;' \
                        's.onerror = function(){console.warn("failed to load library " + url);};' \
                        'document.getElementsByTagName("head")[0].appendChild(s);' \
                        '}' \
                        'if(typeof(LDAvis) !== "undefined"){' \
                        '!function(LDAvis){' \
                        '}(LDAvis);' \
                        '}else if(typeof define === "function" && define.amd){' \
                        'require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});' \
                        'require(["d3"], function(d3){' \
                        'window.d3 = d3;' \
                        'LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){' \
                        'new LDAvis("#" + "ldavis_el70603330337764279074628", ldavis_el70603330337764279074628_data);' \
                        '});' \
                        '});' \
                        '}else{' \
                        'LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){' \
                        'LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){' \
                        'new LDAvis("#" + "ldavis_el70603330337764279074628", ldavis_el70603330337764279074628_data);' \
                        '})' \
                        '});' \
                        '}'
    return dashboard_head + dashboard_script1 + dashboard_script2

# kingsman = "Another witty and fun, action film by Matthew Vaughn. " \
#            "Taron Egerton and Colin Firth come together once more to recreate their fantastic spy duo. " \
#            "Though the movie lacks in parts of what it promised in the trailers and details of what " \
#            "many fans may have hoped. " \
#            "It begrudgingly makes up for though its eye-popping visuals, " \
#            "great new characters, and a clear believable storyline of Harry Hart's revival. Kingsman: " \
#            "The Golden Circle is a near pitch perfect sequel to relive the entertainment of Kingsman: " \
#            "The Secret Service."
#
# data = {
#     'corpus': kingsman,
#     'summary_length': 3,
#     'visualize': True
# }
#
# api(data)

#
# kingsman2 = u"Just an overall fun experience. " \
#             u"I still favor the first one a bit, but this was definitely a worthy successor in my opinion."
#
# print(process(kingsman, 3))
