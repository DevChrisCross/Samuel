from googletrans import Translator

translator = Translator(service_urls=[
      'translate.google.com',
    ])


def translate(corpus):
    translated_corpus = translator.translate(corpus, 'en', 'tl')
    return translated_corpus.text


# movie_review = u"eto ung isa sa movie na hinding hindi ko makakalimutan, every scene ay talagang maganda. Siguradong numero uno to sa takilya"
# hugot = u"Minsan may mga bagay talaga na mawawala at hindi mo ito mapipigilan. " \
#         u"Pero, when the time is right, babalik ito! " \
#         u"At siguradong mas matatag at mas matamis pa kesa nung dati."
# hugot_pa = u"Masakit ang magmahal, pero hindi mo masasabing nagmamahal ka kung di ka pa nasasaktan." \
#            u"If there is pain, ibig sabihin tunay kang nagmamahal."
# hugot_na_hugot = u"Sabi nila kung mahal mo hindi mo siya iiwan, " \
#                  u"pero minsan kailangan mong iwanan kasi nga mahal mo siya, " \
#                  u"and you know na ayun ung magpapaligaya sa kanya. Let go mo nalang siya, and know that the best is yet to come." \
#                  u"May nakalaang pag-ibig para sayo." \
#                  u"Kailangan mo lang matututong maghintay"
#
# print(translate(hugot))
