import os

import httpx
import pytest

from ss14_agent import keystore, providers
from ss14_agent.config import load_config
from ss14_agent.providers import DECISION_PROVIDERS, ORCHESTRATOR_PROVIDERS


def test_config_follows_provider():
    cfg = load_config()
    assert cfg.decision.url == "https://openrouter.ai/api/alpha/decisions"
    assert cfg.decision.key_var == "OPENROUTER_API_KEY" and cfg.decision.layout == "parts"
    cfg.decision.provider = "typesafe"
    assert cfg.decision.url.endswith("/v1/systemone") and cfg.decision.layout == "assets"
    cfg.orchestrator.provider = "ollama"
    assert cfg.orchestrator.key_var == "" and cfg.orchestrator.base_url == "http://localhost:11434/v1"
    cfg.orchestrator.provider = "nope"
    with pytest.raises(ValueError):
        _ = cfg.orchestrator.provider_info


def test_check_reports_bad_and_good_keys(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        req = httpx.Request("GET", url)
        if headers.get("Authorization") == "Bearer good" or headers.get("x-api-key") == "good":
            body = {"data": {"label": "test"}} if url.endswith("/key") else {"data": [{"id": "m1"}, {"id": "m2"}]}
            return httpx.Response(200, json=body, request=req)
        return httpx.Response(401, json={"error": {"message": "Invalid key"}}, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    orp = ORCHESTRATOR_PROVIDERS["openrouter"]
    assert "OK" in providers.check(orp, "good")
    with pytest.raises(providers.ConnectionFailed, match="401"):
        providers.check(orp, "bad")
    with pytest.raises(providers.ConnectionFailed, match="no key"):
        providers.check(orp, "")
    anth = ORCHESTRATOR_PROVIDERS["anthropic"]
    assert providers.check(anth, "good") == "Anthropic OK, 2 models"
    assert providers.list_models(anth, "good") == ["m1", "m2"]
    assert DECISION_PROVIDERS["typesafe"].headers["TypeSafe-Danger-Zone-Images"] == "true"


def test_save_key_to_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(keystore, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(keystore, "_keychain", lambda: False)
    (tmp_path / ".env").write_text("OTHER=1\nGROQ_API_KEY=old\n")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert keystore.save_key("GROQ_API_KEY", " new-key ") == ".env"
    assert os.environ["GROQ_API_KEY"] == "new-key"
    assert (tmp_path / ".env").read_text() == "OTHER=1\nGROQ_API_KEY=new-key\n"
    monkeypatch.delenv("GROQ_API_KEY")
