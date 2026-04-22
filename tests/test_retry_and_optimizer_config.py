"""
Tests for retry logic and optimizer prompt configuration.
"""

import pytest
from unittest.mock import MagicMock, call, patch

from prompt_forge._retry import call_with_retry
from prompt_forge.inference.agent import InferenceAgent
from prompt_forge.training.optimizer import PromptOptimizer
from prompt_forge.training.pipeline import TrainingConfig
from prompt_forge.training.prompt import DEFAULT_OPTIMIZER_PROMPT, DEFAULT_CONSOLIDATION_PROMPT
from prompt_forge.llm.client import LLMResponse, LLMMessage


# ── call_with_retry ───────────────────────────────────────────────────────────

def test_retry_succeeds_on_first_attempt():
    fn = MagicMock(return_value="ok")
    result = call_with_retry(fn, max_retries=3, delay=0)
    assert result == "ok"
    assert fn.call_count == 1


def test_retry_succeeds_after_one_failure():
    fn = MagicMock(side_effect=[RuntimeError("boom"), "ok"])
    result = call_with_retry(fn, max_retries=3, delay=0)
    assert result == "ok"
    assert fn.call_count == 2


def test_retry_exhausts_and_raises():
    fn = MagicMock(side_effect=RuntimeError("always fails"))
    with pytest.raises(RuntimeError, match="always fails"):
        call_with_retry(fn, max_retries=2, delay=0)
    assert fn.call_count == 3  # initial + 2 retries


def test_retry_zero_retries_raises_immediately():
    fn = MagicMock(side_effect=ValueError("instant fail"))
    with pytest.raises(ValueError):
        call_with_retry(fn, max_retries=0, delay=0)
    assert fn.call_count == 1


def test_retry_sleeps_with_backoff():
    fn = MagicMock(side_effect=[RuntimeError(), RuntimeError(), "ok"])
    with patch("prompt_forge._retry.time.sleep") as mock_sleep:
        call_with_retry(fn, max_retries=3, delay=1.0, backoff=2.0)
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0] == call(1.0)   # delay * 2^0
    assert mock_sleep.call_args_list[1] == call(2.0)   # delay * 2^1


# ── InferenceAgent retry ──────────────────────────────────────────────────────

def _make_llm(responses):
    llm = MagicMock()
    llm.complete.side_effect = responses
    return llm


def test_inference_agent_retries_on_failure():
    llm = _make_llm([RuntimeError("rate limit"), LLMResponse(text="ok", usage={})])
    agent = InferenceAgent(llm=llm, prompt_text="sys", max_retries=2, retry_delay=0)
    result = agent.run(input_text="hello")
    assert result == "ok"
    assert llm.complete.call_count == 2


def test_inference_agent_raises_after_exhausting_retries():
    llm = _make_llm([RuntimeError("down")] * 4)
    agent = InferenceAgent(llm=llm, prompt_text="sys", max_retries=3, retry_delay=0)
    with pytest.raises(RuntimeError, match="down"):
        agent.run(input_text="hello")
    assert llm.complete.call_count == 4


def test_inference_agent_default_max_retries():
    agent = InferenceAgent(llm=MagicMock(), prompt_text="sys")
    assert agent.max_retries == 3
    assert agent.retry_delay == 1.0


# ── PromptOptimizer retry ─────────────────────────────────────────────────────

def _make_bundle(tmp_path, name="b"):
    from prompt_forge.bundle import ExampleBundle
    from pathlib import Path
    p = tmp_path / f"{name}_input.txt"
    p.write_text("hello")
    ep = tmp_path / f"{name}_expected.txt"
    ep.write_text("world")
    return ExampleBundle(bundle_id=name, files={"input": p, "expected_output": ep})


def test_optimizer_retries_on_failure(tmp_path):
    good_response = MagicMock()
    good_response.text = (
        "<optimized_prompt>better</optimized_prompt>"
        "<learnings>learned</learnings><issues>None</issues>"
    )
    good_response.usage = {}
    llm = _make_llm([RuntimeError("timeout"), good_response])
    optimizer = PromptOptimizer(llm=llm, max_retries=2, retry_delay=0)
    result = optimizer.optimize("seed", [_make_bundle(tmp_path)])
    assert result.new_prompt == "better"
    assert llm.complete.call_count == 2


def test_optimizer_default_prompts():
    optimizer = PromptOptimizer(llm=MagicMock())
    assert optimizer.optimizer_prompt == DEFAULT_OPTIMIZER_PROMPT
    assert optimizer.consolidation_prompt == DEFAULT_CONSOLIDATION_PROMPT


def test_optimizer_custom_prompts():
    optimizer = PromptOptimizer(
        llm=MagicMock(),
        optimizer_prompt="my optimizer prompt",
        consolidation_prompt="my consolidation prompt",
    )
    assert optimizer.optimizer_prompt == "my optimizer prompt"
    assert optimizer.consolidation_prompt == "my consolidation prompt"


# ── TrainingConfig ────────────────────────────────────────────────────────────

def test_training_config_retry_defaults():
    config = TrainingConfig()
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert not hasattr(config, "optimizer_temperature")
    assert config.eval_train is False


def test_training_config_custom_retry():
    config = TrainingConfig(max_retries=5, retry_delay=2.0)
    assert config.max_retries == 5
    assert config.retry_delay == 2.0


# ── Public exports ────────────────────────────────────────────────────────────

def test_default_prompts_are_public():
    import prompt_forge
    assert hasattr(prompt_forge, "DEFAULT_OPTIMIZER_PROMPT")
    assert hasattr(prompt_forge, "DEFAULT_CONSOLIDATION_PROMPT")
    assert "optimized_prompt" in prompt_forge.DEFAULT_OPTIMIZER_PROMPT
    assert "consolidate" in prompt_forge.DEFAULT_CONSOLIDATION_PROMPT.lower()
