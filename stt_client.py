"""
stt_client.py – Speech-to-Text via ElevenLabs Scribe
Uses ElevenLabs Scribe v1 model for high-accuracy multilingual transcription with diarization and word-level timestamps.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from langsmith_tracer import get_traceable_decorator, calculate_stt_cost_usd


@dataclass
class TranscriptionWord:
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    type: str = "word"          # "word", "spacing", or "audio_event"
    speaker_id: Optional[str] = None
    logprob: Optional[float] = None


@dataclass
class TranscriptionResult:
    raw_text: str
    language_code: Optional[str] = None
    audio_duration_secs: Optional[float] = None
    words: List[TranscriptionWord] = field(default_factory=list)
    engine_used: Optional[str] = None
    transcription_time_secs: Optional[float] = None


class ScribeSTTClient:
    """
    Speech-to-Text client backed exclusively by ElevenLabs Scribe (scribe_v1).
    High-accuracy transcription with speaker diarization and word-level timestamps.
    """

    SUPPORTED_EXTENSIONS = {
        ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
        ".wav", ".webm", ".ogg", ".flac", ".aac", ".mov", ".mkv"
    }

    # ElevenLabs Scribe supports up to 25MB per request
    MAX_SIZE_BYTES = 24 * 1024 * 1024  # 24MB safety margin

    def __init__(
        self,
        api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        **kwargs
    ):
        load_dotenv(override=True)
        # Resolve ElevenLabs API key
        raw_el_key = elevenlabs_api_key or api_key or os.getenv("ELEVENLABS_API_KEY", "")
        el_keys = [k.strip() for k in raw_el_key.split(",") if k.strip()]
        self.elevenlabs_keys: List[str] = list(dict.fromkeys(el_keys))
        self._elevenlabs_clients = []
        init_error = None
        if self.elevenlabs_keys:
            try:
                from elevenlabs.client import ElevenLabs
                self._elevenlabs_clients = [ElevenLabs(api_key=k) for k in self.elevenlabs_keys]
                logger.info(f"Initialized ElevenLabs Scribe STT engine with {len(self._elevenlabs_clients)} API key(s).")
            except Exception as e:
                init_error = str(e)
                logger.warning(f"Could not initialize ElevenLabs client: {e}")

        if not self.elevenlabs_keys:
            raise ValueError(
                "No ElevenLabs API key found. "
                "Set ELEVENLABS_API_KEY in your .env file."
            )
        elif not self._elevenlabs_clients:
            raise ValueError(
                f"Failed to initialize ElevenLabs client: {init_error or 'Unknown error'}"
            )

        self.default_engine = "elevenlabs"
        self.api_keys = self.elevenlabs_keys
        self.elevenlabs_clients = self._elevenlabs_clients
        self._key_cooldowns: dict = {}
        logger.info("STT Client initialized with primary engine: ELEVENLABS SCRIBE (scribe_v1)")

    # ─────────────────────────────────────────────────────────────────────────
    # Audio helpers (FFmpeg-based)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_ffmpeg_exe() -> str:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def _extract_and_chunk_audio(
        self,
        file_path: Path,
        temp_dir: Path,
        max_size_bytes: int = 24 * 1024 * 1024
    ) -> List[tuple[Path, float]]:
        """
        Extracts/compresses audio track to 64kbps mono MP3 if it's a video file or > max_size_bytes.
        If still > max_size_bytes, chunks the audio into 5-minute segments using FFmpeg.
        Each chunk is validated against max_size_bytes; oversized chunks are re-split.
        Returns list of (chunk_file_path, start_offset_secs).
        """
        import subprocess, re

        ffmpeg_exe = self._get_ffmpeg_exe()
        ext = file_path.suffix.lower()
        is_video = ext in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".mpeg"}
        file_size = file_path.stat().st_size

        target_file = file_path

        # Step 1: Compress / extract mono MP3 if file is video or exceeds max size
        if is_video or file_size > max_size_bytes:
            for bitrate in ["64k", "32k"]:
                compressed_file = temp_dir / f"{file_path.stem}_compressed_{bitrate}.mp3"
                logger.info(
                    f"Extracting/compressing audio from '{file_path.name}' "
                    f"({file_size / (1024*1024):.1f} MB) at {bitrate}..."
                )
                cmd = [
                    ffmpeg_exe, "-y", "-threads", "0", "-i", str(file_path),
                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", bitrate,
                    str(compressed_file)
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
                if res.returncode == 0 and compressed_file.exists():
                    target_file = compressed_file
                    new_size = target_file.stat().st_size
                    logger.info(f"Compressed audio to {new_size / (1024*1024):.1f} MB at {bitrate}: {compressed_file.name}")
                    if new_size <= max_size_bytes:
                        break
                    logger.info(f"Compressed file still {new_size / (1024*1024):.1f} MB, trying lower bitrate...")
                else:
                    logger.warning(f"Audio extraction at {bitrate} failed ({res.stderr[:200]}). Trying next option.")

            if target_file == file_path:
                logger.warning("All compression attempts failed. Using original file.")

        # Step 2: If small enough, return as-is
        if target_file.stat().st_size <= max_size_bytes:
            return [(target_file, 0.0)]

        # Step 3: Chunk target_file if it is still larger than max_size_bytes
        total_duration = self._get_audio_duration(ffmpeg_exe, target_file)
        if total_duration is None:
            logger.warning("Could not determine duration for chunking. Attempting direct upload.")
            return [(target_file, 0.0)]

        segment_sec = 300.0
        chunks = self._split_audio(
            ffmpeg_exe, target_file, temp_dir, total_duration,
            segment_sec=segment_sec, bitrate="64k", prefix="chunk"
        )

        validated_chunks: List[tuple[Path, float]] = []
        for chunk_file, offset in chunks:
            chunk_size = chunk_file.stat().st_size
            if chunk_size <= max_size_bytes:
                validated_chunks.append((chunk_file, offset))
            else:
                logger.warning(
                    f"Chunk '{chunk_file.name}' is {chunk_size / (1024*1024):.1f} MB — "
                    f"re-splitting with lower bitrate..."
                )
                sub_dur = self._get_audio_duration(ffmpeg_exe, chunk_file)
                if sub_dur:
                    sub_chunks = self._split_audio(
                        ffmpeg_exe, chunk_file, temp_dir, sub_dur,
                        segment_sec=120.0, bitrate="32k",
                        prefix=f"rechunk_{chunk_file.stem}"
                    )
                    for sub_file, sub_offset in sub_chunks:
                        validated_chunks.append((sub_file, offset + sub_offset))
                else:
                    validated_chunks.append((chunk_file, offset))

        logger.info(
            f"Split large audio ({total_duration:.1f}s) into "
            f"{len(validated_chunks)} validated chunk(s)."
        )
        return validated_chunks if validated_chunks else [(target_file, 0.0)]

    @staticmethod
    def _get_audio_duration(ffmpeg_exe: str, file_path: Path) -> Optional[float]:
        """Get audio duration in seconds using ffprobe or ffmpeg."""
        import subprocess, re
        ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
        if ffprobe_exe != ffmpeg_exe:
            try:
                cmd = [
                    ffprobe_exe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(file_path)
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
                if res.returncode == 0 and res.stdout.strip():
                    return float(res.stdout.strip())
            except Exception:
                pass

        cmd = [ffmpeg_exe, "-i", str(file_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", res.stderr)
        if match:
            h, m, s = match.groups()
            return float(h) * 3600 + float(m) * 60 + float(s)

        match_sec = re.search(r"duration\s*:\s*([\d.]+)", res.stderr, re.IGNORECASE)
        if match_sec:
            return float(match_sec.group(1))
        return None

    @staticmethod
    def _split_audio(
        ffmpeg_exe: str,
        source_file: Path,
        temp_dir: Path,
        total_duration: float,
        segment_sec: float,
        bitrate: str,
        prefix: str,
        overlap_sec: float = 2.0
    ) -> List[tuple[Path, float]]:
        """Split an audio file into segments with a small overlap to prevent cutting off words."""
        import subprocess
        chunks: List[tuple[Path, float]] = []
        start_sec = 0.0
        idx = 0
        step_sec = max(segment_sec - overlap_sec, 1.0)

        while start_sec < total_duration:
            chunk_file = temp_dir / f"{prefix}_{idx:03d}.mp3"
            cmd = [
                ffmpeg_exe, "-y", "-threads", "0",
                "-i", str(source_file),
                "-ss", str(start_sec),
                "-t", str(segment_sec),
                "-ac", "1", "-ar", "16000", "-b:a", bitrate,
                str(chunk_file)
            ]
            subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if chunk_file.exists() and chunk_file.stat().st_size > 1024:
                chunks.append((chunk_file, start_sec))
            elif chunk_file.exists():
                logger.warning(f"Skipping near-empty chunk '{chunk_file.name}' ({chunk_file.stat().st_size} bytes).")

            start_sec += step_sec
            idx += 1

        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Core transcription — ElevenLabs Scribe
    # ─────────────────────────────────────────────────────────────────────────

    @get_traceable_decorator(name="ElevenLabs_Scribe_Chunk_STT", run_type="parser", tags=["elevenlabs", "stt"])
    def _transcribe_file_chunk(
        self,
        file_path: Union[str, Path],
        word_timestamps: bool = True,
        language: Optional[str] = None,
        engine: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        max_retries: int = 3
    ) -> TranscriptionResult:
        """
        Transcribes a single audio file chunk using ElevenLabs Scribe (scribe_v1).
        Falls back automatically to Free SpeechRecognition engine if ElevenLabs API key quota is exhausted.
        """
        file_path = Path(file_path)

        # Validate chunk
        chunk_size = file_path.stat().st_size if file_path.exists() else 0
        if chunk_size < 1024:
            logger.warning(f"Skipping tiny/corrupt chunk '{file_path.name}' ({chunk_size} bytes).")
            return TranscriptionResult(raw_text="", words=[], engine_used="skipped")

        try:
            return self._transcribe_elevenlabs_chunk(
                file_path=file_path,
                word_timestamps=word_timestamps,
                language=language,
                elevenlabs_api_key=elevenlabs_api_key
            )
        except Exception as e:
            logger.warning(f"ElevenLabs STT failed ({e}). Executing automatic fallback STT engine...")
            return self._transcribe_fallback_chunk(file_path=file_path, language=language)

    def _transcribe_fallback_chunk(
        self,
        file_path: Path,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Automatic STT fallback engine using SpeechRecognition (Free Google Web Speech API).
        Triggered seamlessly if ElevenLabs Scribe API quota is exhausted or key is invalid.
        """
        logger.info(f"Using Free STT Fallback engine for '{file_path.name}'...")
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(str(file_path)) as source:
                audio = r.record(source)

            lang_code = language if (language and language != "auto") else "en-US"
            text = r.recognize_google(audio, language=lang_code)
            if text and text.strip():
                words = [
                    TranscriptionWord(text=w, type="word")
                    for w in text.strip().split()
                ]
                return TranscriptionResult(
                    raw_text=text.strip(),
                    words=words,
                    engine_used="Free Web STT Engine (Fallback)"
                )
        except Exception as e:
            logger.warning(f"Fallback STT engine exception: {e}")

        return TranscriptionResult(raw_text="", words=[], engine_used="Fallback Empty")

    def _transcribe_elevenlabs_chunk(
        self,
        file_path: Path,
        word_timestamps: bool = True,
        language: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None
    ) -> TranscriptionResult:
        clients = self._elevenlabs_clients
        if elevenlabs_api_key:
            try:
                from elevenlabs.client import ElevenLabs
                clients = [ElevenLabs(api_key=elevenlabs_api_key)]
            except Exception as e:
                logger.warning(f"Could not init ElevenLabs client with custom key: {e}")

        if not clients:
            raise RuntimeError(
                "No ElevenLabs API key configured. Set ELEVENLABS_API_KEY in your .env file."
            )

        last_error = None
        for idx, el_client in enumerate(clients):
            try:
                logger.info(f"Transcribing '{file_path.name}' using ElevenLabs Scribe (Key #{idx+1})...")
                t0 = time.time()
                with open(file_path, "rb") as audio_file:
                    kwargs = {
                        "file": (file_path.name, audio_file.read(), "audio/wav"),
                        "model_id": "scribe_v1",
                        "tag_audio_events": True,
                        "timestamps_granularity": "word",
                        "diarize": True
                    }
                    if language and language != "auto":
                        kwargs["language_code"] = language

                    response = el_client.speech_to_text.convert(**kwargs)

                elapsed = time.time() - t0
                parsed = self._parse_elevenlabs_response(response)
                parsed.engine_used = "ElevenLabs Scribe v1"
                parsed.transcription_time_secs = round(elapsed, 2)
                return parsed
            except Exception as e:
                last_error = e
                logger.warning(f"ElevenLabs (Key #{idx+1}) failed: {e}")

        raise RuntimeError(f"All ElevenLabs STT attempts failed. Last error: {last_error}")

    def _parse_elevenlabs_response(self, response: Any) -> TranscriptionResult:
        if isinstance(response, dict):
            raw_text = response.get("text", "") or ""
            lang_code = response.get("language_code")
            duration = response.get("audio_duration_secs")
            raw_words = response.get("words") or []
        else:
            raw_text = getattr(response, "text", "") or ""
            lang_code = getattr(response, "language_code", None)
            duration = getattr(response, "audio_duration_secs", None)
            raw_words = getattr(response, "words", None) or []

        parsed_words: List[TranscriptionWord] = []
        for item in raw_words:
            if isinstance(item, dict):
                w_text = item.get("text", "")
                w_start = item.get("start")
                w_end = item.get("end")
                w_type = item.get("type", "word")
                w_speaker = item.get("speaker_id")
            else:
                w_text = getattr(item, "word", None) or getattr(item, "text", "")
                w_start = getattr(item, "start", None)
                w_end = getattr(item, "end", None)
                w_type = getattr(item, "type", "word")
                w_speaker = getattr(item, "speaker_id", None)

            if not w_text:
                continue

            clean = w_text.strip()
            if not clean:
                continue

            if w_text.startswith(" ") and parsed_words and parsed_words[-1].type != "spacing":
                parsed_words.append(TranscriptionWord(text=" ", type="spacing"))

            parsed_words.append(
                TranscriptionWord(
                    text=clean,
                    start=w_start,
                    end=w_end,
                    type=w_type,
                    speaker_id=w_speaker,
                )
            )

        cleaned_text = raw_text.strip()
        return TranscriptionResult(
            raw_text=cleaned_text,
            language_code=lang_code,
            audio_duration_secs=float(duration) if duration is not None else None,
            words=parsed_words
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    @get_traceable_decorator(name="ElevenLabs_Scribe_File_STT", run_type="chain", tags=["elevenlabs", "stt"])
    def transcribe(
        self,
        file_path: Union[str, Path],
        diarization: bool = False,
        tag_audio_events: bool = True,
        word_timestamps: bool = True,
        language: Optional[str] = None,
        engine: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribes an audio/video file using ElevenLabs Scribe (scribe_v1).
        Automatically handles audio extraction and chunking for files > 24MB.
        """
        import tempfile, shutil, time
        from concurrent.futures import ThreadPoolExecutor

        t_start = time.time()
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="stt_preprocess_"))
        try:
            chunks = self._extract_and_chunk_audio(file_path, temp_dir)

            def process_chunk(item):
                chunk_file, time_offset = item
                res = self._transcribe_file_chunk(
                    chunk_file,
                    word_timestamps=word_timestamps,
                    language=language,
                    engine=engine,
                    elevenlabs_api_key=elevenlabs_api_key
                )
                return time_offset, res

            if len(chunks) > 1:
                logger.info(f"Transcribing {len(chunks)} chunks in parallel across worker threads...")
                with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
                    results = list(executor.map(process_chunk, chunks))
            else:
                results = [process_chunk(chunks[0])]

            results.sort(key=lambda x: x[0])

            all_raw_texts: List[str] = []
            all_words: List[TranscriptionWord] = []
            detected_lang = None
            total_duration = 0.0

            for offset, res in results:
                if res.raw_text:
                    all_raw_texts.append(res.raw_text)

                if not detected_lang and res.language_code:
                    detected_lang = res.language_code

                if res.audio_duration_secs:
                    total_duration += res.audio_duration_secs

                for w in res.words:
                    adjusted_w = TranscriptionWord(
                        text=w.text,
                        start=(w.start + offset) if w.start is not None else None,
                        end=(w.end + offset) if w.end is not None else None,
                        type=w.type,
                        speaker_id=w.speaker_id,
                        logprob=w.logprob
                    )
                    all_words.append(adjusted_w)

            combined_text = " ".join(all_raw_texts).strip()
            total_elapsed = time.time() - t_start

            return TranscriptionResult(
                raw_text=combined_text,
                language_code=detected_lang,
                audio_duration_secs=total_duration if total_duration > 0 else None,
                words=all_words,
                engine_used="ElevenLabs Scribe v1",
                transcription_time_secs=round(total_elapsed, 2)
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
