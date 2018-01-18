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


def init(KEY: str):
    def parse_enum(enumeration: Type[Enum]):
        return {en.name: en.value for en in enumeration}

    return {
        'Rank': parse_enum(TextSummarizer.Settings.Rank),
        'Rerank': parse_enum(TextSummarizer.Settings.Rerank),
        'Language': parse_enum(Language),
        'KEY': KEY
    }


def api(data):
    def check_param(default, param: str):
        return default if param not in data else data[param]

    text = data['text']

    # TEXT TRANSLATOR
    translate_from = check_param(Language.TAGALOG.value, "translate_from")
    translate_to = check_param(Language.ENGLISH.value, "translate_to")
    text = translate(text, translate_from, translate_to)

    # TEXT NORMALIZER
    normalizer_settings = (TextNormalizer.Settings()
        .set_independent_properties(minimum_word_length=2, request_tokens=True, preserve_lettercase=True)
        .set_special_character_properties(punctuation_emphasis_level=4)
        .set_word_contraction_properties())
    normalized_text = TextNormalizer(text, normalizer_settings)

    # TEXT SENTIMENT CLASSIFIER
    threshold_classifier = check_param(0.1, "threshold_classifier")
    verbose = check_param(False, "verbose")
    polarity = unsupervised_extractor(text, threshold_classifier, verbose)

    # TEXT TOPIC MODELLER
    visualize = check_param(False, "visualize")
    dashboard = build_dashboard(topic_modelling(normalized_text().normalized_text, visualize))

    # TEXT SUMMARIZER
    summary_length = data['summary_length']
    sort_by_score = check_param(False, "sort_by_score")
    rank = check_param("D", "rank")
    query = check_param(None, "query")

    def summarizer_settings():
        Rank = TextSummarizer.Settings.Rank
        Rerank = TextSummarizer.Settings.Rerank
        _rank = Rank.DIVRANK
        _rerank = Rerank.GRASSHOPPER

        if rank == Rank.DIVRANK.value:
            _rank = Rank.DIVRANK
        elif rank == Rank.LEXRANK.value:
            _rank = Rank.LEXRANK
        elif rank == Rank.GRASSHOPPER.value:
            _rank = Rank.GRASSHOPPER

        if query is not None:
            return TextSummarizer.Settings(_rank, Rerank.MAXIMAL_MARGINAL_RELEVANCE, query)
        else:
            return TextSummarizer.Settings(_rank, _rerank)

    summarize_text = TextSummarizer(normalized_text, summarizer_settings())

    return {
        'summarized_text': summarize_text(summary_length, sort_by_score).summary_text,
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
