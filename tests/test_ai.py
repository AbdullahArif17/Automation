"""Phase 3 tests: provider abstraction, prompts, evaluator, script generator."""
import json
from pathlib import Path

import pytest

from app.ai.provider import LLMProvider, MockProvider, extract_json
from app.ai.prompts import render, load
from app.ai.evaluator import ScriptEvaluator
from app.ai.gemini import GeminiProvider
from app.content.script_generator import ScriptGenerator


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    txt = "Here you go:\n```json\n{\"a\": 2}\n```\nthanks"
    assert extract_json(txt) == {"a": 2}


def test_extract_json_with_surrounding_text():
    txt = "blah {\"a\": 3} trailing"
    assert extract_json(txt) == {"a": 3}


def test_mock_provider_passthrough():
    p = MockProvider()
    out = p.generate_json("any")
    assert "script" in out


def test_prompt_render_substitutes_vars():
    tmpl = load("script")
    rendered = render("script", topic="Local LLMs", summary="s", facts=["f1", "f2"])
    assert "Local LLMs" in rendered
    assert "f1, f2" in rendered
    assert "{topic}" not in rendered


def test_gemini_requires_api_key():
    with pytest.raises(RuntimeError):
        GeminiProvider(api_key="")


def test_evaluator_pass_and_fail():
    # Pass case
    quality_pass = json.dumps({
        "hook": 9, "accuracy": 9, "clarity": 9, "retention": 9, "novelty": 9,
        "pacing": 9, "visual_potential": 9, "naturalness": 9, "policy_risk": 0,
        "total": 9.0, "verdict": "pass", "notes": "good"
    })
    p = MockProvider([quality_pass])
    ev = ScriptEvaluator(p, min_quality=7.5)
    res = ev.evaluate("some script", ["fact"])
    assert res.passed
    assert res.total >= 7.5


def test_script_generator_selects_best_and_passes():
    # Call order: 3 scripts, then 3 evals (per attempt).
    responses = [
        json.dumps({"script": "S1 weak", "hook": "h", "duration_estimate_seconds": 30}),
        json.dumps({"script": "S2 okay", "hook": "h", "duration_estimate_seconds": 30}),
        json.dumps({"script": "S3 strong", "hook": "h", "duration_estimate_seconds": 30}),
        json.dumps({"hook": 7, "accuracy": 7, "clarity": 7, "retention": 7, "novelty": 7,
                    "pacing": 7, "visual_potential": 7, "naturalness": 7, "policy_risk": 1,
                    "total": 7.0, "verdict": "fail"}),
        json.dumps({"hook": 7, "accuracy": 7, "clarity": 7, "retention": 7, "novelty": 7,
                    "pacing": 7, "visual_potential": 7, "naturalness": 7, "policy_risk": 1,
                    "total": 7.2, "verdict": "fail"}),
        json.dumps({"hook": 9, "accuracy": 9, "clarity": 9, "retention": 9, "novelty": 9,
                    "pacing": 9, "visual_potential": 9, "naturalness": 9, "policy_risk": 0,
                    "total": 8.5, "verdict": "pass"}),
    ]
    gen = ScriptGenerator(MockProvider(responses), num_candidates=3, max_attempts=3)
    best = gen.generate("AI topic", summary="s", facts=["f"])
    assert best.text == "S3 strong"
    assert best.evaluation.passed


def test_script_generator_regenerates_on_failure():
    # All attempts fail -> keep best-effort, but it should still return something.
    fail_q = json.dumps({"hook": 5, "accuracy": 5, "clarity": 5, "retention": 5,
                         "novelty": 5, "pacing": 5, "visual_potential": 5,
                         "naturalness": 5, "policy_risk": 0, "total": 5.0, "verdict": "fail"})
    # Mock queue order == call order: per attempt -> 3 scripts then 3 evals.
    responses = []
    for _ in range(3):
        responses += [json.dumps({"script": "g", "hook": "h", "duration_estimate_seconds": 30})
                      for _ in range(3)]
        responses += [fail_q for _ in range(3)]
    gen = ScriptGenerator(MockProvider(responses), num_candidates=3, max_attempts=3)
    best = gen.generate("topic")
    assert best is not None
    assert not best.evaluation.passed  # never passed threshold
