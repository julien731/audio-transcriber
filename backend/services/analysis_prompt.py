"""Server-side analysis-prompt assembly (BR-16, BR-17).

Centralizes the placeholder substitution that previously lived in
``analysis-viewer.js`` so any client can request a ready-to-use prompt. Output is
kept byte-identical to the historical client behavior (BR-18): each placeholder
is replaced once (JavaScript ``String.replace(str, str)`` replaces only the first
occurrence), and the same strip rules apply when a section is empty.
"""

from __future__ import annotations

from backend.schemas import Transcript

AUDIO_PLACEHOLDER = "[AUDIO ANALYSIS CONTEXT]"
MEETING_PLACEHOLDER = "[MEETING CONTEXT]"
TRANSCRIPT_PLACEHOLDER = "[PASTE TRANSCRIPT HERE]"


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS, matching the frontend ``formatTimestamp`` (Math.floor)."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_transcript_text(transcript: Transcript, speakers: dict[str, str]) -> str:
    """Render ``[HH:MM:SS] Speaker: text`` lines using current speaker names.

    Mirrors ``buildPlainTextTranscript`` — segment text is emitted verbatim
    (including WhisperX's leading space) so output matches the web client.
    """
    lines = []
    for segment in transcript.segments:
        name = speakers.get(segment.speaker, segment.speaker)
        lines.append(f"[{format_timestamp(segment.start)}] {name}: {segment.text}")
    return "\n".join(lines)


def assemble_prompt(template: str, audio_context: str, meeting_context: str, transcript_text: str) -> str:
    prompt = template
    if audio_context:
        prompt = prompt.replace(AUDIO_PLACEHOLDER, audio_context, 1)
    else:
        # Strip the placeholder line entirely so output is byte-identical to the
        # pre-audio-analysis prompt (BR-4.4 of the audio-analysis spec).
        prompt = prompt.replace(AUDIO_PLACEHOLDER + "\n", "", 1)

    if meeting_context:
        prompt = prompt.replace(MEETING_PLACEHOLDER, "## Meeting Context\n\n" + meeting_context, 1)
    else:
        prompt = prompt.replace(MEETING_PLACEHOLDER + "\n\n", "", 1)

    prompt = prompt.replace(TRANSCRIPT_PLACEHOLDER, transcript_text, 1)
    return prompt
