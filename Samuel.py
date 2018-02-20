from samuel.normalizer import TextNormalizer, Property, NormalizerManager
from samuel.translator import TranslatorManager, Language
from samuel.summarizer import TextSummarizer
from samuel.topic_modeller import TextTopicModeller
from samuel.sentiment_classifier import TextSentimentClassifier
from functools import partial
from numpy import round
from warnings import filterwarnings
from time import time
from enum import Enum
from typing import Type, Dict, Any, List
from multiprocessing import Pool


def init(key: str) -> Dict[str, Any]:
    def parse_enum(enumeration: Type[Enum]):
        return {en.name: en.value for en in enumeration}

    return {
        'Language': parse_enum(Language),
        'KEY': key
    }


def api(data: Dict) -> Dict[str, Any]:
    def check_param(default: Any, param: str):
        return default if param not in data else data[param]

    text = data['text']
    query = data['query']

    # TEXT TRANSLATOR
    # translate_from = check_param(Language.TAGALOG.value, "translate_from")
    # translate_to = check_param(Language.ENGLISH.value, "translate_to")
    # text = TranslatorManager(text, translate_from = translate_from, translate_to=translate_to).translated_text

    # TEXT NORMALIZER
    token_count = 0
    raw_sents = list()
    sentences = list()
    start_time = time()

    partitions = NormalizerManager.partitioned_docs(text)
    if len(partitions) > 15:
        t_normalizer = NormalizerManager(partitions)
        raw_sents = t_normalizer.raw_sents
        sentences = t_normalizer.sentences
        token_count = len(t_normalizer.tokens)
    else:
        for partition in partitions:
            tn = TextNormalizer(partition, query=query)
            raw_sents.extend(tn.raw_sents)
            sentences.extend(tn.sentences)
            token_count += len(tn.tokens)

    # TEXT SENTIMENT CLASSIFIER
    neu_threshold = check_param(0.1, "threshold_classifier")

    # TEXT TOPIC MODELLER
    visualize = check_param(False, "visualize")
    dashboard_style = check_param(True, "dashboard_style")

    # TEXT SUMMARIZER
    summary_length = data['summary_length']
    sort_by_score = check_param(False, "sort_by_score")

    options = {
        "raw_sents": raw_sents,
        "sents": sentences,
        "summary": summary_length,
        "sort_by_score": sort_by_score,
        "visualize": visualize,
        "query": query,
        "style": dashboard_style,
        "partitions": partitions,
        "neu_threshold": neu_threshold
    }

    samuel_data = dict()
    print("Preparing API Process Pool")
    pool = Pool()
    print("Mapping API Processes")
    result = pool.map_async(partial(api_processor, options=options), list(range(3)))
    for data in result.get():
        samuel_data.update(data)
    end_time = time()
    samuel_data.update({"polarity": samuel_data["sc"]["final_sentiment"],
                        "percentage": samuel_data["sc"]["percentage"]})
    samuel_data.pop("sc")
    print("API Pooling Done")
    print("Data processed in", round(end_time - start_time, 2), "secs. with over",
          len(raw_sents), "sentences consisted of",
          token_count, "tokens (excluding sentences and tokens below normalization threshold)")
    return samuel_data


def api_processor(func_id: int, options: Dict[str, Any]) -> Dict[str, Any]:
    if func_id == 0:
        return exec_sentiment_classifier(options["partitions"], options["neu_threshold"])
    if func_id == 1:
        return exec_topic_modeller(options["sents"], options["visualize"], options["style"])
    if func_id == 2:
        return exec_summarizer(options["raw_sents"], options["sents"], options["summary"], options["sort_by_score"],
                               options["query"])


def exec_sentiment_classifier(partitions: List[str], neu_threshold: float) -> Dict[str, Any]:
    tokens = list()
    for partition in partitions:
        tokens.extend(
            TextNormalizer(partition, {Property.Letter_Case, Property.Stop_Word, Property.Special_Char}).tokens
        )
    sentiment_classifier = TextSentimentClassifier(" ".join(partitions), tokens,
                                                   neutrality_threshold=neu_threshold).sentiment_score
    return {"sc": sentiment_classifier}


def exec_summarizer(raw_sents, sents, summary_length: int, sort_by_score: bool, query: str) -> Dict[str, Any]:
    summarizer = TextSummarizer(raw_sents, sents, summary_length=summary_length, sort_by_score=sort_by_score)
    summary, scores = summarizer.continuous_lexrank()
    if query:
        try:
            summary, scores = summarizer.mmr(query, scores)
        except ValueError as ve:
            pass
    return {"summarized_text": summary}


def exec_topic_modeller(sents: List[str], visualize: bool, style: bool) -> Dict[str, Any]:
    def build_dashboard(data, dashboard_style=True):
        dashboard_head = '<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">\n'
        dashboard_script1 = '<div id="ldavis_el70603330337764279074628"></div>\n' \
                            '<script type="text/javascript">\n' \
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
        return ((dashboard_head + dashboard_script1 + dashboard_script2) if dashboard_style
        else (dashboard_script1 + dashboard_script2))

    dashboard = build_dashboard(TextTopicModeller(sents, visualize=visualize).topics, dashboard_style=style)
    return {"dashboard": dashboard}


if __name__ == "__main__":
    from samuel.test.test_document import single_test_document, document3

    api({
        "text": single_test_document,
        "summary_length": 10,
        "visualize": True,
        "query": "I see Joe Ragan commenting on me."
    })
    pass
