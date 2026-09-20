"""Unit tests for the LLM orchestration layer, using fake API responses.

Covers: parameter pass-through from RequestSettings, few-shot message
layout, structured-output request shape, malformed JSON, truncated output,
empty replies, and provider errors — all without network access.
"""

import json

import pytest

from categories import TECHNIQUES
from core import (
    COACHING_FORMAT,
    JUDGE_FORMAT,
    QUESTION_SET_FORMAT,
    CoachingResponse,
    JudgeFeedback,
    RequestSettings,
    ask_llm,
    estimate_cost,
    evaluate_category_answer,
    generate_category_questions,
    generate_concept_image,
    generate_cover_letter,
    generate_prep,
    run_feature,
    summarize_cv,
)

SETTINGS = RequestSettings(
    model="openai/gpt-5-mini",
    temperature=0.4,
    top_p=0.9,
    frequency_penalty=0.5,
    max_tokens=1234,
)


# ---------------------------------------------------------------------------
# ask_llm — the single shared API path
# ---------------------------------------------------------------------------


def test_ask_llm_passes_all_settings(fake_client_factory):
    client = fake_client_factory("hello")
    ask_llm(client, SETTINGS, "system", "user question")
    call = client.calls[0]
    assert call["model"] == "openai/gpt-5-mini"
    assert call["temperature"] == 0.4
    assert call["top_p"] == 0.9
    assert call["frequency_penalty"] == 0.5
    assert call["max_tokens"] == 1234
    assert "response_format" not in call


def test_ask_llm_message_roles(fake_client_factory):
    client = fake_client_factory("ok")
    ask_llm(client, SETTINGS, "SYS", "USER")
    messages = client.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "SYS"
    assert messages[1]["content"] == "USER"


def test_ask_llm_few_shot_examples_alternate_roles(fake_client_factory):
    client = fake_client_factory("ok")
    examples = TECHNIQUES["Few-shot"]["examples"]
    ask_llm(client, SETTINGS, "SYS", "real question", examples=examples)
    roles = [m["role"] for m in client.calls[0]["messages"]]
    # system, then user/assistant per example, then the real user message
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert client.calls[0]["messages"][-1]["content"] == "real question"


def test_ask_llm_sends_response_format(fake_client_factory):
    client = fake_client_factory('{"questions": []}')
    ask_llm(client, SETTINGS, "s", "u", response_format=QUESTION_SET_FORMAT)
    rf = client.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert "questions" in rf["json_schema"]["schema"]["properties"]


def test_ask_llm_empty_reply_returns_empty_string(fake_client_factory):
    client = fake_client_factory("")
    assert ask_llm(client, SETTINGS, "s", "u") == ""
    client = fake_client_factory(None)
    assert ask_llm(client, SETTINGS, "s", "u") == ""


def test_ask_llm_returns_truncated_content(fake_client_factory):
    client = fake_client_factory("partial answ", finish_reason="length")
    assert ask_llm(client, SETTINGS, "s", "u") == "partial answ"


# ---------------------------------------------------------------------------
# run_feature / generate_prep
# ---------------------------------------------------------------------------


def test_run_feature_puts_cv_in_user_message_not_system(fake_client_factory):
    client = fake_client_factory("advice text")
    run_feature(
        client,
        SETTINGS,
        feature_name="STAR answer coach",
        user_text="my draft answer",
        technique=TECHNIQUES["Zero-shot"],
        cv_summary="Python dev at Acme, 5 years.",
    )
    messages = client.calls[0]["messages"]
    system = messages[0]["content"]
    user = messages[-1]["content"]
    assert "Acme" not in system
    assert "Acme" in user
    assert "my draft answer" in user


def test_run_feature_structured_returns_typed_coaching(fake_client_factory):
    payload = json.dumps(
        {
            "advice": "Lead with results.",
            "action_items": ["Quantify impact"],
            "common_mistakes": ["Being vague"],
        }
    )
    client = fake_client_factory(payload)
    result = run_feature(
        client,
        SETTINGS,
        feature_name="STAR answer coach",
        user_text="draft",
        technique=TECHNIQUES["Structured output"],
        structured=True,
    )
    assert isinstance(result, CoachingResponse)
    assert result.advice == "Lead with results."
    assert client.calls[0]["response_format"] == COACHING_FORMAT


def test_run_feature_structured_falls_back_to_text_on_bad_json(fake_client_factory):
    client = fake_client_factory("not json at all")
    result = run_feature(
        client,
        SETTINGS,
        feature_name="STAR answer coach",
        user_text="draft",
        technique=TECHNIQUES["Structured output"],
        structured=True,
    )
    assert result == "not json at all"


def test_run_feature_provider_error_returns_friendly_message(fake_client_factory):
    client = fake_client_factory(error=RuntimeError("429 rate limit exceeded"))
    result = run_feature(
        client,
        SETTINGS,
        feature_name="STAR answer coach",
        user_text="draft",
        technique=TECHNIQUES["Zero-shot"],
    )
    assert isinstance(result, str) and result.startswith("⚠️")
    assert "rate limit" in result


def test_generate_prep_uses_technique_and_role_context(fake_client_factory):
    client = fake_client_factory("answer")
    generate_prep(
        client,
        SETTINGS,
        technique=TECHNIQUES["Chain-of-thought"],
        role="Data Analyst",
        seniority="Junior",
        difficulty="Easy",
        user_question="What SQL should I know?",
    )
    messages = client.calls[0]["messages"]
    assert messages[0]["content"] == TECHNIQUES["Chain-of-thought"]["system"]
    user = messages[-1]["content"]
    assert "Junior-level Data Analyst" in user
    assert "What SQL should I know?" in user


def test_generate_prep_empty_reply_yields_warning_message(fake_client_factory):
    client = fake_client_factory("")
    result = generate_prep(
        client,
        SETTINGS,
        technique=TECHNIQUES["Zero-shot"],
        role="Data Analyst",
        seniority="Mid",
        difficulty="Medium",
        user_question="anything",
    )
    assert isinstance(result, str) and result.startswith("⚠️")


# ---------------------------------------------------------------------------
# generate_category_questions
# ---------------------------------------------------------------------------


def _questions_payload(n=10):
    return json.dumps({"questions": [f"Question number {i} — long enough?" for i in range(n)]})


def test_generate_questions_happy_path(fake_client_factory):
    client = fake_client_factory(_questions_payload())
    questions = generate_category_questions(
        client,
        SETTINGS,
        category_name="Python Coding",
        difficulty="Medium",
        seniority="Mid",
    )
    assert len(questions) == 10
    assert client.calls[0]["response_format"] == QUESTION_SET_FORMAT


def test_generate_questions_accepts_bare_array(fake_client_factory):
    client = fake_client_factory(
        json.dumps(["A first question long enough?", "A second question long enough?"])
    )
    questions = generate_category_questions(
        client,
        SETTINGS,
        category_name="Python Coding",
        difficulty="Easy",
        seniority="Junior",
    )
    assert len(questions) == 2


def test_generate_questions_falls_back_to_numbered_lines(fake_client_factory):
    raw = (
        "1. What is a generator in Python and when would you use one?\n"
        "2. Explain how dictionaries handle hash collisions internally."
    )
    client = fake_client_factory(raw)
    questions = generate_category_questions(
        client,
        SETTINGS,
        category_name="Python Coding",
        difficulty="Hard",
        seniority="Senior",
    )
    assert len(questions) == 2
    assert questions[0].startswith("What is a generator")


def test_generate_questions_provider_error_returns_empty(fake_client_factory):
    client = fake_client_factory(error=RuntimeError("connection timeout"))
    assert (
        generate_category_questions(
            client,
            SETTINGS,
            category_name="Python Coding",
            difficulty="Easy",
            seniority="Junior",
        )
        == []
    )


def test_generate_questions_empty_reply_returns_empty(fake_client_factory):
    client = fake_client_factory("")
    assert (
        generate_category_questions(
            client,
            SETTINGS,
            category_name="Python Coding",
            difficulty="Easy",
            seniority="Junior",
        )
        == []
    )


# ---------------------------------------------------------------------------
# evaluate_category_answer (LLM-as-a-judge)
# ---------------------------------------------------------------------------

JUDGE_PAYLOAD = json.dumps(
    {
        "score": 4,
        "verdict": "Good but generic.",
        "strengths": ["Correct definition"],
        "improvements": ["Add an example", "Mention complexity"],
        "model_answer": "A strong answer would define it and give an example.",
    }
)


def test_evaluate_answer_returns_typed_feedback(fake_client_factory):
    client = fake_client_factory(JUDGE_PAYLOAD)
    feedback = evaluate_category_answer(
        client,
        SETTINGS,
        category_name="Python Coding",
        question="What is a list comprehension?",
        answer="It builds lists concisely.",
    )
    assert isinstance(feedback, JudgeFeedback)
    assert feedback.score == 4
    assert client.calls[0]["response_format"] == JUDGE_FORMAT


def test_evaluate_answer_malformed_json_returns_raw_text(fake_client_factory):
    client = fake_client_factory("Score: 4/5. Nice answer!")
    feedback = evaluate_category_answer(
        client,
        SETTINGS,
        category_name="Python Coding",
        question="q",
        answer="a",
    )
    assert feedback == "Score: 4/5. Nice answer!"


def test_evaluate_answer_provider_error_returns_message(fake_client_factory):
    client = fake_client_factory(error=RuntimeError("401 unauthorized"))
    feedback = evaluate_category_answer(
        client,
        SETTINGS,
        category_name="Python Coding",
        question="q",
        answer="a",
    )
    assert isinstance(feedback, str) and "authentication" in feedback


# ---------------------------------------------------------------------------
# summarize_cv
# ---------------------------------------------------------------------------


def test_summarize_cv_empty_text_makes_no_api_call(fake_client_factory):
    client = fake_client_factory()
    assert summarize_cv(client, SETTINGS, "") == ""
    assert summarize_cv(client, SETTINGS, "   ") == ""
    assert client.calls == []


def test_summarize_cv_sends_cv_in_user_message(fake_client_factory):
    client = fake_client_factory("- Skills: Python, SQL")
    summary = summarize_cv(client, SETTINGS, "John Doe, Data Analyst at Acme")
    assert summary == "- Skills: Python, SQL"
    messages = client.calls[0]["messages"]
    assert "Acme" not in messages[0]["content"]  # not in the system message
    assert "Acme" in messages[-1]["content"]  # in the user message


def test_summarize_cv_api_error_returns_empty(fake_client_factory):
    client = fake_client_factory(error=RuntimeError("boom"))
    assert summarize_cv(client, SETTINGS, "some cv text") == ""


# ---------------------------------------------------------------------------
# estimate_cost (per-request pricing)
# ---------------------------------------------------------------------------

PRICING = {"openai/gpt-5-mini": {"prompt": 0.000001, "completion": 0.000002}}


def test_estimate_cost_uses_injected_pricing():
    cost = estimate_cost("openai/gpt-5-mini", 1000, 500, pricing=PRICING)
    # 1000 * 1e-6 + 500 * 2e-6 = 0.001 + 0.001 = 0.002
    assert cost == pytest.approx(0.002)


def test_estimate_cost_unknown_model_returns_none():
    assert estimate_cost("who/knows", 10, 10, pricing=PRICING) is None


# ---------------------------------------------------------------------------
# Job-description threading (RAG-style grounding, user message only)
# ---------------------------------------------------------------------------


def test_run_feature_puts_job_description_in_user_message(fake_client_factory):
    client = fake_client_factory("advice")
    run_feature(
        client,
        SETTINGS,
        feature_name="STAR answer coach",
        user_text="my draft",
        technique=TECHNIQUES["Zero-shot"],
        job_description="Backend role at Globex requiring Kafka.",
    )
    messages = client.calls[0]["messages"]
    assert "Globex" not in messages[0]["content"]  # not in system message
    assert "Globex" in messages[-1]["content"]  # in user message


# ---------------------------------------------------------------------------
# generate_cover_letter
# ---------------------------------------------------------------------------


def test_cover_letter_puts_cv_and_jd_in_user_message_not_system(fake_client_factory):
    client = fake_client_factory("Dear [Hiring Manager], ...")
    result = generate_cover_letter(
        client,
        SETTINGS,
        cv_summary="Data engineer at Acme, 6 years, built Kafka pipelines.",
        job_description="Senior data engineer at Globex, Kafka + Spark.",
    )
    assert result.startswith("Dear")
    messages = client.calls[0]["messages"]
    system = messages[0]["content"]
    user = messages[-1]["content"]
    # Untrusted CV / JD content must live in the user message only.
    assert "Acme" not in system and "Globex" not in system
    assert "Acme" in user and "Globex" in user
    # No structured-output envelope — a cover letter is plain prose.
    assert "response_format" not in client.calls[0]


def test_cover_letter_includes_extra_notes_in_user_message(fake_client_factory):
    client = fake_client_factory("letter")
    generate_cover_letter(
        client,
        SETTINGS,
        cv_summary="Backend dev.",
        job_description="Backend role.",
        extra_notes="Emphasise my open-source leadership.",
    )
    assert "open-source leadership" in client.calls[0]["messages"][-1]["content"]


def test_cover_letter_empty_reply_returns_warning(fake_client_factory):
    client = fake_client_factory("")
    result = generate_cover_letter(client, SETTINGS, cv_summary="x", job_description="y")
    assert isinstance(result, str) and result.startswith("⚠️")


def test_cover_letter_provider_error_returns_friendly_message(fake_client_factory):
    client = fake_client_factory(error=RuntimeError("429 rate limit exceeded"))
    result = generate_cover_letter(client, SETTINGS, cv_summary="x", job_description="y")
    assert isinstance(result, str) and result.startswith("⚠️")
    assert "rate limit" in result


# ---------------------------------------------------------------------------
# generate_concept_image (multimodal image generation)
# ---------------------------------------------------------------------------


class _ImgMessage:
    def __init__(self, images):
        self.content = ""
        self.images = images


class _ImgChoice:
    def __init__(self, message):
        self.message = message


class _ImgResponse:
    def __init__(self, images):
        self.choices = [_ImgChoice(_ImgMessage(images))]


class _ImgClient:
    """Minimal fake exposing chat.completions.create for image tests."""

    def __init__(self, images=None, error=None):
        self._images = images
        self._error = error
        self.calls = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer._error is not None:
                    raise outer._error
                return _ImgResponse(outer._images)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# 1x1 transparent PNG, base64 — a valid decodable payload
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def test_generate_concept_image_returns_bytes():
    images = [{"image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}}]
    client = _ImgClient(images=images)
    result = generate_concept_image(client, "Explain RAG")
    assert isinstance(result, bytes) and len(result) > 0
    # requests the image modality
    assert client.calls[0]["extra_body"]["modalities"] == ["image", "text"]


def test_generate_concept_image_no_images_returns_none():
    client = _ImgClient(images=[])
    assert generate_concept_image(client, "Explain RAG") is None


def test_generate_concept_image_provider_error_returns_none():
    client = _ImgClient(error=RuntimeError("boom"))
    assert generate_concept_image(client, "Explain RAG") is None
