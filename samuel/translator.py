from googletrans import Translator
from enum import Enum

translator = Translator(service_urls=[
    'translate.google.com',
])


class Language(Enum):
    TAGALOG = "tl"
    ENGLISH = "en"


def translate(corpus: str, enabled=False, translate_from=Language.TAGALOG.value, translate_to=Language.ENGLISH.value):
    return corpus if not enabled else translator.translate(corpus, translate_to, translate_from).text
