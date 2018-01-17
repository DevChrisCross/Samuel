from TextNormalizer import TextNormalizer
from TextTranslator import translate, Language
from TextSummarizer import TextSummarizer
from TextTopicModeller import topic_modelling
from TextSentimentClassifier import unsupervised_extractor
from warnings import filterwarnings
from enum import Enum
from typing import Type

from pprint import pprint

filterwarnings(action='ignore')

def init():
    Rank = TextSummarizer.Settings.Rank
    Rerank = TextSummarizer.Settings.Rerank

    def parse_enum(enumeration: Type[Enum]):
        return str({en.name: en.name for en in enumeration})

    return{
        'Rank': parse_enum(Rank),
        'Rerank': parse_enum(Rerank),
        'Language': parse_enum(Language)
    }


def api(data):

    def check_param(default,param,value):
        return default if param not in data else value

    corpus = data['corpus']

    translate_from = check_param(Language.TAGALOG,"translate_from",data['translate_from'])
    translate_to = check_param(Language.ENGLISH, "translate_to", data['translate_to'])
    corpus = translate(corpus, translate_from, translate_to)

    # For Unsupervised Extractor
    threshold_extractor = 0.1 if 'threshold_extractor' not in data else data['threshold_extractor']
    verbose = False if 'verbose' not in data else data['verbose']

    polarity = unsupervised_extractor(corpus, threshold_extractor, verbose)

    # For Topic Modelling
    visualize = False if 'visualize' not in data else data['visualize']

    dashboard = build_dashboard(topic_modelling(normalized_corpus['normalized'], visualize))

    # For Summarizer
    summary_length = data['summary_length']
    threshold_summarizer = 0.001 if 'threshold_summarizer' not in data else data['threshold_summarizer']
    rank = "D" if 'rank' not in data else data['rank']
    rerank = False if 'rerank' not in data else data['rerank']
    query = None if 'query' not in data else data['query']
    sort_score = False if 'sort_score' not in data else data['sort_score']
    #split_sent = False if 'split_sent' not in data else data['split_sent']
    #correct_sent = False if 'correct_sent' not in data else data['correct_sent']
    #tokenize_sent = True if 'tokenize_sent' not in data else data['tokenize_sent']

    summarized_corpus = summarizer(corpus, summary_length, threshold_summarizer, rank, rerank, query, sort_score,
                                   split_sent,
                                   correct_sent, tokenize_sent)

    # TEXT NORMALIZER
    request_tokens = False
    preserve_lettercase = False
    minimum_word_length = 1

    expand_word_contraction = False
    contraction_map = TextNormalizer.DEFAULT_CONTRACTION_MAP
    preserve_stopword = False

    enable_pos_tag_filter = True
    pos_tag_map = TextNormalizer.DEFAULT_POS_TAG_MAP
    correct_spelling = False
    preserve_wordform = False

    preserve_special_character = False
    preserve_punctuation_emphasis = False
    punctuation_emphasis_list = TextNormalizer.DEFAULT_PUNCTUATION_EMPHASIS
    punctuation_emphasis_level = 1

    settings = (TextNormalizer.Settings()
        .set_independent_properties(minimum_word_length, request_tokens, preserve_lettercase)
        .set_word_contraction_properties(expand_word_contraction, preserve_stopword, contraction_map)
        .set_special_character_properties(preserve_special_character, preserve_punctuation_emphasis,
                                          punctuation_emphasis_level, punctuation_emphasis_list)
        .set_pos_tag_properties(preserve_wordform, correct_spelling, enable_pos_tag_filter, pos_tag_map))
    normalized_corpus = TextNormalizer(corpus, settings)

    # TEXT SUMMARIZER
    Rank = TextSummarizer.Settings.Rank
    Rerank = TextSummarizer.Settings.Rerank

    summary_length = data['summary_length']
    sort_by_score = False if 'sort_score' not in data else data['sort_score']
    threshold_summarizer = 0.001 if 'threshold_summarizer' not in data else data['threshold_summarizer']
    rank = "D" if 'rank' not in data else data['rank']
    rerank = False if 'rerank' not in data else data['rerank']
    query = None if 'query' not in data else data['query']
    RankType =
    RerankType =
    tsSettings = TextSummarizer.Settings(RankType, RerankType)


    rank_map = {
        Rank.DIVRANK.name: {}
    }

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
