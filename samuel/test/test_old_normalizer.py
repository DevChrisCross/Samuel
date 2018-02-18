from samuel.old.old_normalizer import TextNormalizer


def test_old_normalizer():
    settings = (TextNormalizer.Settings()
                .set_independent_properties(minimum_word_length=2, request_tokens=True, preserve_lettercase=True)
                .set_special_character_properties(punctuation_emphasis_level=4)
                .set_word_contraction_properties()
                .set_pos_tag_properties(enable_pos_tag_filter=False))
    text_normalizer = TextNormalizer(
        "I hope this group of film-makers!!!! never re-unites. ever again. IT SUCKS????  >:(", settings)
    text_normalizer().append("Here you go! :)")
    assert text_normalizer.normalized_text == [['hope', 'this', 'group', 'of', 'film-makers!!!!'],
                                               ['never', 're-unites'], ['ever', 'again'], ['IT', 'SUCKS????'], ['>:('],
                                               ['Here', 'you', 'go'], [':)']]
