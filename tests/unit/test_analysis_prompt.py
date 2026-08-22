from __future__ import annotations

from backend.schemas import Transcript, TranscriptSegment
from backend.services import analysis_prompt


class TestFormatTimestamp:
    def test_zero(self):
        assert analysis_prompt.format_timestamp(0) == "00:00:00"

    def test_minutes_seconds(self):
        assert analysis_prompt.format_timestamp(75.9) == "00:01:15"

    def test_hours(self):
        assert analysis_prompt.format_timestamp(3661) == "01:01:01"


class TestBuildTranscriptText:
    def _transcript(self):
        return Transcript(
            segments=[
                TranscriptSegment(id="s1", start=0.0, end=1.5, speaker="SPEAKER_00", text=" Hello there."),
                TranscriptSegment(id="s2", start=1.5, end=3.0, speaker="SPEAKER_01", text=" How are you?"),
            ],
            language="en",
        )

    def test_uses_speaker_display_names(self):
        text = analysis_prompt.build_transcript_text(self._transcript(), {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
        assert text == "[00:00:00] Alice:  Hello there.\n[00:00:01] Bob:  How are you?"

    def test_falls_back_to_raw_speaker_id(self):
        text = analysis_prompt.build_transcript_text(self._transcript(), {})
        assert text.startswith("[00:00:00] SPEAKER_00:  Hello there.")


class TestAssemblePrompt:
    TEMPLATE = "Intro\n[AUDIO ANALYSIS CONTEXT]\nMid\n[MEETING CONTEXT]\n\nEnd\n[PASTE TRANSCRIPT HERE]\n"

    def test_all_sections_present(self):
        out = analysis_prompt.assemble_prompt(self.TEMPLATE, "AUDIO", "notes", "TX")
        assert "AUDIO" in out
        assert "## Meeting Context\n\nnotes" in out
        assert "TX" in out
        assert "[PASTE TRANSCRIPT HERE]" not in out
        assert "[AUDIO ANALYSIS CONTEXT]" not in out
        assert "[MEETING CONTEXT]" not in out

    def test_empty_audio_strips_placeholder_line(self):
        out = analysis_prompt.assemble_prompt(self.TEMPLATE, "", "notes", "TX")
        assert "[AUDIO ANALYSIS CONTEXT]" not in out
        assert "Intro\nMid" in out

    def test_empty_meeting_context_strips_placeholder(self):
        out = analysis_prompt.assemble_prompt(self.TEMPLATE, "AUDIO", "", "TX")
        assert "[MEETING CONTEXT]" not in out
        assert "## Meeting Context" not in out

    def test_only_first_occurrence_replaced(self):
        template = "[PASTE TRANSCRIPT HERE] and [PASTE TRANSCRIPT HERE]"
        out = analysis_prompt.assemble_prompt(template, "", "", "TX")
        assert out == "TX and [PASTE TRANSCRIPT HERE]"
