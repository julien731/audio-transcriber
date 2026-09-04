from __future__ import annotations

import sys
import types

import pytest

from backend.services import align_models


class TestAlignReposFor:
    def test_thai_maps_to_app_override_repo(self):
        assert align_models.align_repos_for(["th"]) == ["airesearch/wav2vec2-large-xlsr-53-th"]

    def test_torch_native_languages_contribute_nothing(self):
        assert align_models.align_repos_for(["en", "fr", "de", "es", "it"]) == []

    def test_unknown_codes_are_skipped(self):
        assert align_models.align_repos_for(["xx", "zz"]) == []

    def test_mixed_set_keeps_only_hf_repos(self):
        # en is torch-native (skipped), th + ja are HF-backed.
        assert align_models.align_repos_for(["en", "th", "ja"]) == [
            "airesearch/wav2vec2-large-xlsr-53-th",
            "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        ]

    def test_deduplicated_and_order_stable(self):
        assert align_models.align_repos_for(["ja", "th", "ja"]) == [
            "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
            "airesearch/wav2vec2-large-xlsr-53-th",
        ]

    def test_empty_input(self):
        assert align_models.align_repos_for([]) == []


class TestAlignModelCached:
    """align_model_cached probes the HF cache to decide if a load will download."""

    def _install_fake_hub(self, monkeypatch, return_value):
        """Inject a fake huggingface_hub whose try_to_load_from_cache is stubbed."""
        fake = types.ModuleType("huggingface_hub")
        calls: list[tuple] = []

        def _probe(repo_id, filename):
            calls.append((repo_id, filename))
            if isinstance(return_value, Exception):
                raise return_value
            return return_value

        fake.try_to_load_from_cache = _probe
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
        return calls

    def test_cached_path_string_is_present(self, monkeypatch):
        calls = self._install_fake_hub(monkeypatch, "/cache/models--foo/config.json")
        assert align_models.align_model_cached("foo/bar") is True
        assert calls == [("foo/bar", "config.json")]

    def test_not_cached_none_is_absent(self, monkeypatch):
        self._install_fake_hub(monkeypatch, None)
        assert align_models.align_model_cached("foo/bar") is False

    def test_known_missing_sentinel_is_absent(self, monkeypatch):
        # huggingface_hub returns a _CACHED_NO_EXIST sentinel (not a str) for a
        # file known to be absent from the repo.
        sentinel = object()
        self._install_fake_hub(monkeypatch, sentinel)
        assert align_models.align_model_cached("foo/bar") is False

    def test_probe_error_degrades_to_present(self, monkeypatch):
        self._install_fake_hub(monkeypatch, RuntimeError("cache scan failed"))
        assert align_models.align_model_cached("foo/bar") is True

    def test_missing_huggingface_hub_degrades_to_present(self, monkeypatch):
        # Absent in the lightweight CI env: the lazy import raises, and the probe
        # must degrade to "assume present" rather than blow up.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        assert align_models.align_model_cached("foo/bar") is True


class TestHfAlignReposDriftGuard:
    """Guard that the copied HF map stays in sync with the installed WhisperX.

    Dev/ML-env only: WhisperX is absent in CI, so this is skipped there — it is a
    developer guard, not a CI gate. Drift is bounded regardless because provisioning
    and the transcriber read the *same* map (worst case: a stale-but-valid repo,
    still watchdogged at the align stage).
    """

    def test_matches_whisperx_defaults_on_shared_keys(self):
        whisperx_alignment = pytest.importorskip("whisperx.alignment")
        defaults = whisperx_alignment.DEFAULT_ALIGN_MODELS_HF
        for code, repo in defaults.items():
            assert align_models.HF_ALIGN_REPOS.get(code) == repo, f"drift for {code}"

    def test_thai_is_an_app_only_override(self):
        whisperx_alignment = pytest.importorskip("whisperx.alignment")
        assert "th" not in whisperx_alignment.DEFAULT_ALIGN_MODELS_HF
        assert align_models.HF_ALIGN_REPOS["th"] == "airesearch/wav2vec2-large-xlsr-53-th"

    def test_torch_native_languages_match_whisperx(self):
        """Torch-native languages must equal WhisperX's torchaudio set, so a
        language is never classified torch-native here while WhisperX resolves it
        to an (un-provisioned) HF repo."""
        whisperx_alignment = pytest.importorskip("whisperx.alignment")
        assert align_models.TORCH_ALIGN_LANGUAGES == set(whisperx_alignment.DEFAULT_ALIGN_MODELS_TORCH)
