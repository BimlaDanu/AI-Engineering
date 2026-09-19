from src.analysis.quality import assess, fault_rates, readability, style_faults


def test_style_faults_detects_a_truncated_summary():
    assert style_faults("The council approved the budget and then")["truncated"]


def test_style_faults_accepts_a_properly_ended_summary():
    assert not style_faults("The council approved the budget.")["truncated"]


def test_style_faults_detects_a_repeated_sentence():
    assert style_faults("It rained. It rained.")["repeated_sentence"]


def test_style_faults_detects_an_unresolved_opening_pronoun():
    assert style_faults("They approved the budget.")["opens_with_pronoun"]
    assert not style_faults("Councillors approved the budget.")["opens_with_pronoun"]


def test_style_faults_detects_a_model_preamble():
    assert style_faults("Here is a summary of the article: the budget passed.")["has_preamble"]
    assert not style_faults("The budget passed.")["has_preamble"]


def test_style_faults_detects_an_overlong_sentence():
    long_sentence = " ".join(["word"] * 45) + "."
    assert style_faults(long_sentence)["overlong_sentence"]


def test_readability_reports_words_per_sentence():
    stats = readability("One two three. Four five six.", "en")
    assert stats["words_per_sentence"] == 3.0


def test_readability_is_not_attempted_for_an_unsupported_language():
    stats = readability("ஒரு சில சொற்கள்.", "ta")
    assert stats["flesch"] != stats["flesch"], (
        "unsupported languages should give NaN, not a wrong number"
    )


def test_assess_produces_one_row_per_summary(summaries):
    assessed = assess(summaries)
    assert len(assessed) == len(summaries)
    assert {"flesch", "truncated", "has_preamble"} <= set(assessed.columns)


def test_fault_rates_reports_a_percentage_per_method(summaries):
    rates = fault_rates(assess(summaries))
    assert set(rates.index) == {"extractive", "abstractive"}
    assert (rates >= 0).all().all() and (rates <= 100).all().all()
