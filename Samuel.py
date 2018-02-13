from samuel.normalizer import TextNormalizer, Property
from samuel.translator import TranslatorManager, Language
from samuel.summarizer import TextSummarizer
from samuel.topic_modeller import TextTopicModeller
from samuel.sentiment_classifier import TextSentimentClassifier
from warnings import filterwarnings
from enum import Enum
from typing import Type, Dict, Any
filterwarnings(action='ignore')


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

    # TEXT TRANSLATOR
    translate_from = check_param(Language.TAGALOG.value, "translate_from")
    translate_to = check_param(Language.ENGLISH.value, "translate_to")
    text = TranslatorManager(text, translate_from = translate_from, translate_to=translate_to).translated_text

    # TEXT NORMALIZER
    t_normalizer = TextNormalizer(text)

    # TEXT SENTIMENT CLASSIFIER
    neu_threshold = check_param(0.1, "threshold_classifier")
    t_normalizer2 = TextNormalizer(text, {Property.Letter_Case, Property.Stop_Word, Property.Special_Char},
                                   min_word_length=2)
    sentiment_classifier = TextSentimentClassifier(text, t_normalizer2.tokens,
                                                   neutrality_threshold=neu_threshold).sentiment_score

    # TEXT TOPIC MODELLER
    visualize = check_param(False, "visualize")
    dashboard_style = check_param(True, "dashboard_style")
    dashboard = build_dashboard(TextTopicModeller(t_normalizer.sentences, visualize),dashboard_style)

    # TEXT SUMMARIZER
    summary_length = data['summary_length']
    sort_by_score = check_param(False, "sort_by_score")
    query = check_param(None, "query")

    summarizer = TextSummarizer(t_normalizer.raw_sents, t_normalizer.sentences,
                                summary_length=summary_length, sort_by_score=sort_by_score)
    summary, scores = summarizer.grasshopper()
    if query:
        summary, scores = summarizer.mmr(query, scores)

    return {
        'summarized_text': summary,
        'polarity': sentiment_classifier['final_sentiment'],
        'percentage': sentiment_classifier['percentage'],
        'dashboard': dashboard
    }


def build_dashboard(data, dashboard_style = True):
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
    return (dashboard_head + dashboard_script1 + dashboard_script2) if dashboard_style else (dashboard_script1 + dashboard_script2)


kingsman = "Another witty and fun, action film by Matthew Vaughn. " \
           "Taron Egerton and Colin Firth come together once more to recreate their fantastic spy duo. " \
           "Though the movie lacks in parts of what it promised in the trailers and details of what " \
           "many fans may have hoped. " \
           "It begrudgingly makes up for though its eye-popping visuals, " \
           "great new characters, and a clear believable storyline of Harry Hart's revival. Kingsman: " \
           "The Golden Circle is a near pitch perfect sequel to relive the entertainment of Kingsman: " \
           "The Secret Service."

text1 = "Hands down, of all the smartphones I have used so far, iPhone 8 Plus got the best battery life. I am not a " \
        "heavy user. All I do is make few quick calls, check emails, quick update of social media and maps and " \
        "navigation once in a while. On average with light use (excluding maps and navigation), iPhone 8 Plus lasts " \
        "for 4 full days! You heard it right, 4 full days! At the end of the 4th day, I am usually left with 5-10% of " \
        "battery and that's about the time I charge the phone. The heaviest I used it was once when I had to rely on " \
        "GPS for a full day. I started with 100% on the day I was travelling and by the end of the day, I had around " \
        "70% left. And I was able get through the next two days without any issues (light use only). "
text2 = "The last iPhone I used was an iPhone 5 and it is very clear that the smartphone cameras have come a long way. " \
        "iPhone 8 Plus produces very crisp photos without any over saturation, which is what I really appreciate. Even " \
        "though I got used to Samsung's over saturated photos over the last 3 years, whenever I see a photo true to " \
        "real life colours, it really appeals me. When buying this phone, my main concern with camera was its " \
        "performance in low light as I was used to pretty awesome performance on my Note 4. iPhone 8 Plus did not " \
        "disappoint me. I was able to capture some shots at a work function and they looked truly amazing. Auto HDR " \
        "seems very on point in my opinion. You will see these in the link below. Portrait mode has been somewhat " \
        "consistent. I felt that it does not perform as well on a very bright day. But overall, given that it is still " \
        "in beta, it works quite well (See Camaro SS photo). Video recording wise, it is pretty good at the standard " \
        "1080p 30fps. I am yet to try any 4k 60fps shots. But based on what I have seen from tech reviewers, " \
        "it is pretty awesome. "
text3 = "I have missed iOS a lot over the last 3 years and it feels good to be back. Super smooth and no hiccups. I " \
        "know few people have experienced some bugs recently. I guess I was one of the lucky ones not to have any " \
        "issues. Maybe it was already patched when I bought the phone. However, my only complaint is the fact that " \
        "iOS still does not let you clear all notifications from a particular app at once. I really would like to see " \
        "fixed in a future update. Customisation wise, I do not have any issues because I hardly customised my Note " \
        "4. Only widgets I had running were weather and calendar. Even then, I would still open up the actual app to " \
        "seek more detail. However, I still do not use iCloud. I really wish Apple would have given us more online " \
        "storage for backup. 5GB is hardly enough these days to backup everything on your phone. One of my mates " \
        "suggested that Apple should ship the phone with iCloud storage same as the phone. It surely would be " \
        "awesome. But business wise, I cannot see Apple going ahead with such a decision. But in my opinion, " \
        "iCloud users should get at least 15GB free. Coming from an Android, I thought it would make sense to keep " \
        "using my Google account to sync contacts and photos as it would take away the hassle of setting everything " \
        "up from scratch. Only issue is sometimes I feel like iOS restricting the background app refresh of Google " \
        "apps such as Google photos. For example, I always have to keep Google Photos running in order to allow " \
        "background upload, which makes no sense. Same goes for OneDrive. Overall, navigation around the OS is easy " \
        "and convenient. "
