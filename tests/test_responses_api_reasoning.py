from openai_harmony import ReasoningEffort

from gpt_oss.responses_api.api_server import get_reasoning_effort
from gpt_oss.responses_api.types import ReasoningConfig


def test_reasoning_config_default_uses_string_literal():
    config = ReasoningConfig()
    assert isinstance(config.effort, str)
    assert config.effort == "low"


def test_get_reasoning_effort_accepts_string_literal():
    result = get_reasoning_effort("medium")
    assert result is ReasoningEffort.MEDIUM


def test_get_reasoning_effort_accepts_enum_instance():
    result = get_reasoning_effort(ReasoningEffort.HIGH)
    assert result is ReasoningEffort.HIGH
