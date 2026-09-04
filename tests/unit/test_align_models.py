from __future__ import annotations

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
