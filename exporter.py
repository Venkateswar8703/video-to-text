import os
from pathlib import Path
from typing import Union, List, Optional
from stt_client import TranscriptionResult, TranscriptionWord

def format_plain_text(result: TranscriptionResult, include_speakers: bool = True) -> str:
    """
    Formats the transcription result as clean plain text.
    If include_speakers is True and speaker diarization data is present, groups sequential words by speaker.
    Otherwise, returns the clean continuous transcript.
    """
    if not result.words or not include_speakers:
        return result.raw_text.strip()

    # Check if speaker diarization data actually exists in words
    has_speakers = any(w.speaker_id is not None for w in result.words)
    if not has_speakers:
        return result.raw_text.strip()

    formatted_lines: List[str] = []
    current_speaker: Optional[str] = None
    current_line_words: List[str] = []

    for w in result.words:
        # Determine effective speaker (keep previous if a spacing token lacks speaker_id)
        spk = w.speaker_id or current_speaker or "Speaker ?"
        if spk != current_speaker:
            # Commit the previous line if it exists
            if current_line_words:
                line_text = "".join(current_line_words).strip()
                if line_text:
                    formatted_lines.append(f"[{current_speaker}]: {line_text}")
                current_line_words = []
            current_speaker = spk
        
        # Format word/spacing or audio events
        if w.type == "audio_event":
            # Tag event cleanly in text
            current_line_words.append(f" ({w.text}) ")
        else:
            current_line_words.append(w.text)

    # Commit any remaining words
    if current_line_words:
        line_text = "".join(current_line_words).strip()
        if line_text:
            formatted_lines.append(f"[{current_speaker}]: {line_text}")

    return "\n\n".join(formatted_lines)


def save_transcript_to_file(
    result: TranscriptionResult,
    output_path: Union[str, Path],
    include_speakers: bool = True
) -> Path:
    """
    Saves the plain text transcription to the specified file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = format_plain_text(result, include_speakers=include_speakers)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
            
    return output_path
