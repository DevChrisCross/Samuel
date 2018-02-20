from googletrans import Translator
from enum import Enum
from functools import partial
from typing import Union, List, Optional
import multiprocessing as mp


class Language(Enum):
    TAGALOG = "tl"
    ENGLISH = "en"


class TextTranslator:
    def __init__(self):
        self._translator = Translator(service_urls=[
            'translate.google.com',
        ])

    def is_language(self, text: str, lang: Language) -> bool:
        detection_rate = self._translator.detect(text)
        return detection_rate.lang == lang.value

    def translate_to(self, text: str,
                     _from: Language = Language.TAGALOG,
                     _to: Language = Language.ENGLISH) -> str:
        return self._translator.translate(text, src=_from.value, dest=_to.value).text

    def translate_if(self, text: str, lang: Language, _to: Language = Language.ENGLISH) -> Optional:
        if self.is_language(text, lang):
            return self.translate_to(text, _from=lang, _to=_to)
        else:
            return None


class TranslatorManager:
    def __init__(self, text: str, character_threshold: int = 4500,
                 translate_from: Language = Language.TAGALOG.value, translate_to: Language = Language.ENGLISH.value):
        tokens = text.split()

        def partition_text():
            batch_string = ""
            for token in tokens:
                if len(batch_string) > character_threshold:
                    yield batch_string
                    batch_string = token
                else:
                    batch_string += " " + token
            yield batch_string

        print("Preparing document batches: object", (id(self)))
        text_batches = list(partition_text())

        print("Preparing process pool: object", (id(self)))
        pool = mp.Pool()
        print("Mapping document batches: object", (id(self)))
        result = pool.map_async(partial(translate, translate_from=translate_from, translate_to=translate_to),
                                text_batches)

        self._text = " ".join(result.get())
        pool.close()
        pool.join()
        print("Translation pooling done: object", (id(self)))

    @property
    def translated_text(self):
        return self._text


if __name__ == "__main__":
    translate_if("panget", Language.TAGALOG)
    # from samuel.test.test_document import single_test_document
    # print(TranslatorManager(single_test_document).translated_text)
    pass
