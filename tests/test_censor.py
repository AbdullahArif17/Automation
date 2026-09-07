from app.utils.censor import censor_word, censor_text, censor_word_tuples

def test_censor_word_basic():
    assert censor_word("fuck") == "f*ck"
    assert censor_word("FUCKING") == "F*CKING"
    assert censor_word("Bitch") == "B*tch"
    assert censor_word("shit!") == "sh*t!"
    assert censor_word('"asshole?"') == '"a**hole?"'
    assert censor_word("clean") == "clean"

def test_censor_text():
    raw = "What the fuck is this bullshit, you bitch?"
    expected = "What the f*ck is this bullsh*t, you b*tch?"
    assert censor_text(raw) == expected

def test_censor_text_casing_and_punctuation():
    raw = "SHIT! That was fucking wild, dumbass."
    expected = "SH*T! That was f*cking wild, dumba**."
    assert censor_text(raw) == expected

def test_censor_word_tuples():
    tuples = [
        ("What", 0.0, 0.5),
        ("the", 0.5, 0.8),
        ("fuck", 0.8, 1.2),
        ("man?", 1.2, 1.6),
    ]
    censored = censor_word_tuples(tuples)
    assert censored[2] == ("f*ck", 0.8, 1.2)
    assert censored[0] == ("What", 0.0, 0.5)
