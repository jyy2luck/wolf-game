"""生成配置的旧格式兼容、校验和持久化。"""

import json
import os

import pytest

import config
from config import LLMConfig


def test_old_config_uses_new_defaults_without_rewriting(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    original = json.dumps({"base_url": "https://x/v1/", "api_key": "test", "model": "m"})
    path.write_text(original)
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    loaded = config.load_config()
    assert loaded.base_url == "https://x/v1"
    assert loaded.max_tokens == 4096
    assert loaded.max_tokens_limit == 16384
    assert loaded.thinking is None and loaded.json_mode is None
    assert path.read_text() == original
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_generation_options_roundtrip(tmp_path, monkeypatch):
    folder = tmp_path / "llm-config"
    monkeypatch.setattr(config, "CONFIG_DIR", folder)
    monkeypatch.setattr(config, "CONFIG_PATH", folder / "config.json")
    expected = LLMConfig("https://x/v1", "test", "m", 1024, 8192, "disabled", True)
    config.save_config(expected)
    assert config.load_config() == expected
    if os.name != "nt":
        assert config.CONFIG_PATH.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("options", [
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": True},
    {"max_tokens": "4096"}, {"max_tokens": 1.5},
    {"max_tokens_limit": 0}, {"max_tokens_limit": True},
    {"max_tokens_limit": 2048}, {"thinking": "auto"},
    {"json_mode": "false"}, {"json_mode": 1},
])
def test_invalid_generation_options_fail_when_loading(tmp_path, monkeypatch, options):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"base_url": "https://x", "api_key": "test", "model": "m", **options}))
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    with pytest.raises(ValueError):
        config.load_config()


@pytest.mark.parametrize('options', [
    {'max_input_tokens': 0}, {'max_input_tokens': True},
    {'context_window_tokens': 4500}, {'context_window_tokens': True},
    {'action_timeout_seconds': 0}, {'action_timeout_seconds': True},
    {'action_timeout_seconds': float('nan')}, {'action_timeout_seconds': float('inf')},
])
def test_invalid_action_budgets(options):
    with pytest.raises(ValueError):
        LLMConfig('https://x', 'test', 'm', **options)


def test_action_budgets_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CONFIG_DIR', tmp_path)
    monkeypatch.setattr(config, 'CONFIG_PATH', tmp_path/'config.json')
    expected = LLMConfig('https://x', 'test', 'm', max_input_tokens=9000,
                         context_window_tokens=32000, action_timeout_seconds=45)
    config.save_config(expected)
    assert config.load_config() == expected


@pytest.mark.parametrize('endpoint,thinking,disabled', [
    ('https://api.deepseek.com/v1',None,True),
    ('https://api.deepseek.com/v1','enabled',True),
    ('https://x/v1','enabled',True),
    ('https://x/v1',None,False),
])
def test_connectivity_check_never_enables_thinking(monkeypatch,endpoint,thinking,disabled):
    from types import SimpleNamespace
    requests=[]
    sdk=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:requests.append(kw))))
    monkeypatch.setattr(config,'OpenAI',lambda **kw:sdk)
    assert config.verify_config(LLMConfig(endpoint,'test','m',thinking=thinking)) is None
    assert ('extra_body' in requests[0]) == disabled
    if disabled:
        assert requests[0]['extra_body']=={'thinking':{'type':'disabled'}}
