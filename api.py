import os
import sys
import re
import io
import uuid
import wave
import json
import shutil
import asyncio
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add Deno JS runtime to PATH for yt-dlp signature solving if installed
deno_bin = str(Path.home() / ".deno" / "bin")
if os.path.exists(deno_bin) and deno_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = deno_bin + os.pathsep + os.environ.get("PATH", "")

from stt_client import ScribeSTTClient
from exporter import format_plain_text
from summary_generator import SummaryGenerator, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODELS, get_language_name
from langsmith_tracer import calculate_stt_cost_usd, calculate_token_cost_usd

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="ScribeAI ElevenLabs Summarizer API",
    description="AI Speech-to-Text powered by ElevenLabs Scribe v1 and Multilingual Gemini Summarization",
    version="4.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AUDIO EXTENSIONS vs TEXT DOCUMENT EXTENSIONS
AUDIO_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mkv", ".webm", ".mpeg", ".mov", ".wma"}
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".json", ".csv", ".srt", ".vtt", ".xml", ".html", ".py", ".js", ".ts", ".c", ".cpp", ".java"}

# Active live stream sessions: session_id -> subprocess.Popen
active_stream_sessions: dict = {}

# Direct stream extensions/protocols that bypass yt-dlp
_DIRECT_STREAM_PATTERNS = re.compile(
    r"(\.m3u8|\.mpd|\.mp3|\.mp4|\.aac|\.flac|\.wav|\.ogg|\.m4a)($|\?)"
    r"|^(rtmp|rtsp|udp|rtp)://",
    re.IGNORECASE
)

# Patterns that indicate a URL needs yt-dlp to resolve a direct stream URL
_YTDLP_URL_PATTERNS = re.compile(
    r"(youtube\.com|youtu\.be|twitch\.tv|instagram\.com|twitter\.com|x\.com"
    r"|facebook\.com|dailymotion\.com|vimeo\.com|tiktok\.com)",
    re.IGNORECASE,
)


def _clean_channel_slug(url: str) -> str:
    """Extracts a clean brand/channel search query from domain host and URL path for live search fallbacks."""
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        domain_brand = re.sub(r'^(?:www\.|m\.|mobile\.)', '', parsed.netloc.lower())
        domain_brand = re.sub(r'\.(com|in|org|net|co\.uk|tv|news|cc|io|gov|edu).*$', '', domain_brand)
        path = parsed.path
        slug = re.sub(r'\.(html?|php|asp)$', '', path.strip('/'), flags=re.IGNORECASE)
        cleaned_path = re.sub(r'[/\-_]+', ' ', slug)
        generic_words = {'livetv', 'live-tv', 'live', 'watch', 'stream', 'online', 'news', 'tv', 'video', 'index', 'home'}
        path_words = [w for w in cleaned_path.split() if w.lower() not in generic_words]
        p_str = ' '.join(path_words)
        query = (domain_brand + ' ' + p_str).strip() if p_str else domain_brand
        return f"{query} live stream"
    except Exception:
        return "news live stream"


def _resolve_stream_url(url: str, ffmpeg_exe: str) -> tuple:
    """
    Universally resolves ANY live stream URL, news website URL, YouTube live link,
    or direct HLS / RTMP stream into a playable stream source for FFmpeg / yt-dlp.

    Returns (resolved_url: str, is_yt_live: bool).
    """
    url = url.strip()

    # Step 1: Direct HLS / RTMP / audio file links bypass yt-dlp resolution
    if _DIRECT_STREAM_PATTERNS.search(url):
        logger.info(f"Direct stream URL detected: {url}")
        return (url, False)

    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is required to stream from video websites. "
            "Install it with: pip install yt-dlp"
        )

    logger.info(f"Resolving stream URL via universal resolver: {url}")

    base_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "extractor_args": {"youtube": {"player_client": ["mweb", "ios", "android", "web"]}},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "http_headers": {
            "ngrok-skip-browser-warning": "69420",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    def _try_extract(opts: dict, target_url: str) -> Optional[dict]:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(target_url, download=False)
        except Exception as e:
            logger.debug(f"yt-dlp attempt failed for {target_url}: {e}")
            return None

    info = None

    # Step 2: Direct yt-dlp extraction (with cookies.txt if present)
    cookies_path = Path("cookies.txt")
    if cookies_path.exists():
        opts = dict(base_opts)
        opts["cookiefile"] = str(cookies_path.resolve())
        info = _try_extract(opts, url)
        if info:
            logger.info("Successfully extracted stream using cookies.txt")

    if not info:
        info = _try_extract(base_opts, url)
        if info:
            logger.info("Successfully extracted stream using base_opts")

    # Step 3: Try alternative player_client configurations for YouTube / social platforms
    if not info and _YTDLP_URL_PATTERNS.search(url):
        alt_clients = [
            ["mweb", "android"],
            ["android"],
            ["ios"],
            ["tv_embedded"],
            ["web"],
        ]
        for client_list in alt_clients:
            opts = dict(base_opts)
            opts["extractor_args"] = {"youtube": {"player_client": client_list}}
            info = _try_extract(opts, url)
            if info:
                logger.info(f"Successfully extracted stream using player_client={client_list}")
                break

    # Step 4: Webpage HTML embed & metadata scraping (urllib + curl fallback)
    if not info and not ("youtube.com" in url.lower() or "youtu.be" in url.lower()):
        page_html = ""
        try:
            import urllib.request
            req = urllib.request.Request(url, headers=base_opts["http_headers"])
            page_html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
        except Exception as urllib_err:
            logger.debug(f"urllib HTML fetch failed for {url}: {urllib_err}")
            try:
                curl_cmd = [
                    "curl", "-s", "-L",
                    "-H", f"User-Agent: {base_opts['user_agent']}",
                    "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "-H", "Accept-Language: en-US,en;q=0.9",
                    url
                ]
                res = subprocess.run(curl_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=10)
                if len(res.stdout) > 500:
                    page_html = res.stdout
            except Exception as curl_err:
                logger.debug(f"curl fallback fetch failed for {url}: {curl_err}")

        if page_html:
            candidate_urls = []

            # 4a. Direct .m3u8 / .mpd links in HTML / JS scripts
            for m3u8_link in re.findall(r'https?://[^\s"\'<>]+\.(?:m3u8|mpd)[^\s"\'<>]*', page_html, re.IGNORECASE):
                candidate_urls.append(m3u8_link)

            # 4b. Embedded YouTube video / live stream links
            for yt_id in re.findall(r'https?://(?:www\.)?(?:youtube\.com/(?:embed/|watch\?v=|live/)|youtu\.be/)([a-zA-Z0-9_-]{11})', page_html):
                candidate_urls.append(f"https://www.youtube.com/watch?v={yt_id}")

            # 4c. JSON-LD VideoObject script tags
            for script_content in re.findall(r'<script[^>]*>(.*?)</script>', page_html, re.DOTALL):
                if "VideoObject" in script_content:
                    try:
                        vdata = json.loads(script_content)
                        if isinstance(vdata, dict):
                            e_url = vdata.get("embedUrl") or vdata.get("contentUrl")
                            if e_url:
                                candidate_urls.append(e_url)
                    except Exception:
                        pass

            # 4d. Embedded video iframe URLs (Brightcove, JWPlayer, Video.js, Vimeo)
            for iframe_src in re.findall(r'<iframe[^>]+src=["\']([^"\']+)["\']', page_html, re.IGNORECASE):
                if any(k in iframe_src.lower() for k in ["brightcove", "player", "embed", "youtube", "vimeo", "jwplayer"]):
                    candidate_urls.append(iframe_src)

            # 4e. OpenGraph video meta tags
            for og_vid in re.findall(r'<meta[^>]+property=["\']og:video:?(?:url|secure_url)?["\'][^>]+content=["\']([^"\']+)["\']', page_html, re.IGNORECASE):
                candidate_urls.append(og_vid)

            for embed_target in candidate_urls:
                if _DIRECT_STREAM_PATTERNS.search(embed_target):
                    logger.info(f"Extracted direct stream URL from webpage: {embed_target}")
                    return (embed_target, False)
                extracted = _try_extract(base_opts, embed_target)
                if extracted and (extracted.get("url") or extracted.get("formats") or extracted.get("is_live")):
                    logger.info(f"Successfully extracted stream from embedded player URL: {embed_target}")
                    return (embed_target, True)

    # Step 5: Universal Channel Live Search Fallback (for CDN/Akamai 403-blocked news sites or JS-only players)
    if not info and not ("youtube.com" in url.lower() or "youtu.be" in url.lower()):
        search_query = _clean_channel_slug(url)
        logger.info(f"Webpage stream non-extractable directly. Executing channel live search fallback for '{search_query}'")
        search_opts = dict(base_opts)
        search_opts["default_search"] = "ytsearch5"
        search_info = _try_extract(search_opts, f"ytsearch5:{search_query}")
        if search_info and search_info.get("entries"):
            # Select the first entry that is currently live
            live_entry = None
            for entry in search_info["entries"]:
                if entry and entry.get("is_live"):
                    live_entry = entry
                    break
            if not live_entry:
                live_entry = search_info["entries"][0]

            if live_entry:
                info = live_entry
                logger.info(f"Resolved channel live stream for '{search_query}': '{info.get('title')}' ({info.get('webpage_url')})")

    # Step 6: Browser cookies auto-detection fallback (only for installed browsers)
    if not info:
        home = Path.home()
        appdata = Path(os.environ.get("APPDATA", ""))
        localappdata = Path(os.environ.get("LOCALAPPDATA", ""))
        browser_paths = {
            "chrome": [home / "Library/Application Support/Google/Chrome", home / ".config/google-chrome", localappdata / "Google/Chrome"],
            "firefox": [home / "Library/Application Support/Firefox", home / ".mozilla/firefox", appdata / "Mozilla/Firefox"],
            "safari": [home / "Library/Safari"],
            "edge": [home / "Library/Application Support/Microsoft Edge", localappdata / "Microsoft/Edge"],
            "brave": [home / "Library/Application Support/BraveSoftware/Brave-Browser", localappdata / "BraveSoftware/Brave-Browser"],
        }
        browsers_to_try = [b for b, paths in browser_paths.items() if any(p.exists() for p in paths if str(p))]
        for b in browsers_to_try:
            try:
                opts = dict(base_opts)
                opts["cookiesfrombrowser"] = (b,)
                info = _try_extract(opts, url)
                if info:
                    logger.info(f"Successfully extracted stream using browser cookies from '{b}'")
                    break
            except Exception as b_err:
                logger.debug(f"Browser cookies retry with '{b}' failed: {b_err}")

    if not info:
        if "youtube" in url.lower() or "youtu.be" in url.lower():
            logger.warning(f"yt-dlp info extraction restricted for YouTube URL: {url}. Falling back to webpage stream mode.")
            return (url, True)
        else:
            raise RuntimeError(
                f"Could not extract a live audio stream from webpage '{url}'. "
                "Please verify the URL or provide a direct .m3u8 HLS stream link or YouTube live stream link."
            )

    # Prefer direct playable audio/HLS manifest URL extracted by yt-dlp
    direct_url = info.get("url") or info.get("manifest_url")
    if not direct_url:
        formats = info.get("formats") or []
        for fmt in reversed(formats):
            if fmt.get("url"):
                direct_url = fmt["url"]
                break

    resolved_webpage = info.get("webpage_url") or url
    use_ytdlp_pipe = (
        bool(info.get("is_live"))
        or any(domain in resolved_webpage.lower() for domain in ["youtube.com", "youtu.be", "brightcove", "jwplayer", "vimeo", "dailymotion", "twitch.tv"])
    )
    if use_ytdlp_pipe:
        logger.info(f"Using yt-dlp live pipe mode for: {resolved_webpage}")
        return (resolved_webpage, True)

    if direct_url and not re.search(r"\.(html?|php|asp)($|\?)", direct_url, re.IGNORECASE):
        logger.info(f"Successfully resolved direct playable stream URL: {direct_url[:90]}...")
        return (direct_url, False)

    logger.info(f"Falling back to yt-dlp live pipe mode for: {resolved_webpage}")
    return (resolved_webpage, True)

    raise RuntimeError(
        f"Could not extract a live audio stream from webpage '{url}'. "
        "Please use a direct .m3u8 HLS stream URL or YouTube live stream link."
    )

    logger.info(f"Resolved to: {direct_url[:100]}...")
    return (direct_url, False)


class StreamRequest(BaseModel):
    """Request body for the live stream transcription endpoint."""
    stream_url: str
    chunk_seconds: int = 6
    language: Optional[str] = None
    realtime: bool = True
    english_only: bool = False



@app.get("/api/config")
async def get_config():
    """
    Returns current model configuration and available endpoints.
    """
    import summary_generator
    return {
        "primary_model": summary_generator.DEFAULT_PRIMARY_MODEL,
        "fallback_models": summary_generator.DEFAULT_FALLBACK_MODELS,
        "supported_audio_formats": list(AUDIO_EXTENSIONS),
        "supported_text_formats": list(TEXT_EXTENSIONS),
        "stt_engine": "ElevenLabs Scribe v1"
    }


@app.post("/api/summarize")
async def summarize_document(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    tag_events: bool = Form(True),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None)
):
    """
    Accepts an audio file or written text document, extracts/transcribes text, and generates a structured summary.
    Pass engine='elevenlabs' for deep diarized STT via ElevenLabs Scribe.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file filename provided.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in AUDIO_EXTENSIONS and file_ext not in TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: {', '.join(sorted(list(AUDIO_EXTENSIONS | TEXT_EXTENSIONS)))}"
        )

    transcript_text = ""
    word_timestamps = []
    file_type = "audio" if file_ext in AUDIO_EXTENSIONS else "document"
    audio_duration_secs = None
    detected_lang_code = None  # populated from STT result for audio files
    stt_engine_used = None
    transcription_time_secs = None

    # 1. Extraction / Transcription Step
    if file_type == "audio":
        logger.info(f"Processing audio file: {file.filename} (engine={engine}, diarize={diarize}, events={tag_events}, lang={language})")

        # Initialize STT client
        try:
            stt_client = ScribeSTTClient(elevenlabs_api_key=elevenlabs_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize STT Client: {e}")
            raise HTTPException(status_code=500, detail=f"Speech-to-Text configuration error: {str(e)}")

        # Save uploaded audio to temp disk location
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / file.filename
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Perform speech-to-text (offloaded to thread to prevent blocking event loop)
            result = await asyncio.to_thread(
                stt_client.transcribe,
                file_path=temp_path,
                diarization=diarize,
                tag_audio_events=tag_events,
                word_timestamps=True,
                language=language,
                engine=engine,
                elevenlabs_api_key=elevenlabs_api_key
            )
            transcript_text = format_plain_text(result, include_speakers=diarize)
            audio_duration_secs = result.audio_duration_secs
            detected_lang_code = result.language_code  # capture detected language
            stt_engine_used = getattr(result, "engine_used", None)
            transcription_time_secs = getattr(result, "transcription_time_secs", None)
            word_timestamps = [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "speaker_id": w.speaker_id,
                    "type": w.type
                }
                for w in result.words
            ]
        except Exception as e:
            logger.error(f"Transcription failed for {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # It is a text document, read directly
        logger.info(f"Processing text document: {file.filename}")
        try:
            content_bytes = await file.read()
            transcript_text = content_bytes.decode("utf-8", errors="replace").strip()
        except Exception as e:
            logger.error(f"Failed to read text document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read document text: {str(e)}")

        if not transcript_text:
            raise HTTPException(status_code=400, detail="Uploaded document is empty.")

    # 2. Summarization Step with Gemini LLM
    logger.info("Starting summary generation...")
    try:
        summary_generator = SummaryGenerator()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generator init error: {str(e)}")

    fallback_events = []

    def on_fallback_event(from_model, to_model, err):
        logger.warning(f"Fallback triggered from {from_model} to {to_model}: {err}")
        fallback_events.append({"from_model": from_model, "to_model": to_model, "error": str(err)})

    try:
        summary_text = await asyncio.to_thread(
            summary_generator.generate_summary,
            transcript_text=transcript_text,
            on_fallback_callback=on_fallback_event
        )
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

    # 3. Post-Transcription Translation to English via LLM
    logger.info("Translating transcript to English via LLM...")
    try:
        english_transcript = await asyncio.to_thread(
            summary_generator.translate_to_english,
            transcript_text=transcript_text,
            detected_language=detected_lang_code,
            on_fallback_callback=on_fallback_event
        )
    except Exception as e:
        logger.warning(f"Translation to English failed: {e}")
        english_transcript = transcript_text

    actual_model_used = fallback_events[-1]["to_model"] if fallback_events else DEFAULT_PRIMARY_MODEL

    detected_lang_name = get_language_name(detected_lang_code) if detected_lang_code else None

    return {
        "status": "success",
        "filename": file.filename,
        "file_type": file_type,
        "audio_duration_secs": audio_duration_secs,
        "transcript": transcript_text,
        "english_transcript": english_transcript,
        "summary": summary_text,
        "words": word_timestamps,
        "model_used": actual_model_used,
        "primary_model_configured": DEFAULT_PRIMARY_MODEL,
        "fallbacks": fallback_events,
        "token_usage": summary_generator.total_token_usage.to_dict(),
        "detected_language": detected_lang_code,
        "detected_language_name": detected_lang_name,
        "stt_engine_used": stt_engine_used,
        "transcription_time_secs": transcription_time_secs,
    }


class URLSummarizeRequest(BaseModel):
    url: str
    diarize: bool = True
    tag_events: bool = True
    language: Optional[str] = None
    engine: Optional[str] = None


def _download_youtube_audio(url: str, output_path: Path, ffmpeg_exe: str) -> Path:
    import urllib.request
    logger.info(f"Extracting audio from YouTube / URL: {url}")
    out_stem = str(output_path.parent / output_path.stem)
    is_social = bool(_YTDLP_URL_PATTERNS.search(url))

    # 1. Try yt-dlp first
    try:
        import yt_dlp
        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": out_stem + ".%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "ffmpeg_location": ffmpeg_exe,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "socket_timeout": 30,
            "retries": 5,
            "extractor_retries": 3,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "http_headers": {
                "ngrok-skip-browser-warning": "69420",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        }

        if Path("cookies.txt").exists():
            ydl_opts["cookiefile"] = str(Path("cookies.txt").resolve())

        download_success = False
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            download_success = True
        except Exception as first_err:
            err_str = str(first_err)
            if any(k in err_str for k in ["not a bot", "Sign in", "cookies"]):
                for b in ["chrome", "edge", "firefox", "brave"]:
                    try:
                        logger.info(f"Retrying download with browser cookies from '{b}'...")
                        retry_opts = dict(ydl_opts)
                        retry_opts["cookiesfrombrowser"] = (b,)
                        with yt_dlp.YoutubeDL(retry_opts) as ydl:
                            ydl.download([url])
                        download_success = True
                        break
                    except Exception as b_err:
                        logger.warning(f"Browser cookie '{b}' failed: {b_err}")
            if not download_success:
                raise first_err

        mp3_file = Path(out_stem + ".mp3")
        if mp3_file.exists() and mp3_file.stat().st_size > 500:
            return mp3_file

        matching = [f for f in output_path.parent.glob(f"{output_path.stem}.*") if f.suffix.lower() in AUDIO_EXTENSIONS and f.stat().st_size > 500]
        if matching:
            return matching[0]
    except Exception as yt_err:
        logger.warning(f"yt-dlp download failed ({yt_err}).")
        if is_social:
            raise RuntimeError(f"Could not extract playable audio from YouTube/social URL: {yt_err}")

    # 2. Direct HTTP download fallback for direct audio stream URLs (.mp3, .wav, .mp4, etc.)
    try:
        ext = ".mp4" if ".mp4" in url.lower() else (".wav" if ".wav" in url.lower() else ".mp3")
        direct_out = Path(out_stem + ext)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "ngrok-skip-browser-warning": "69420"
            }
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                raise RuntimeError(f"URL returned web page HTML ({content_type}) instead of audio file.")
            with open(direct_out, "wb") as f:
                f.write(resp.read())

        if direct_out.exists() and direct_out.stat().st_size > 1024:
            logger.info(f"Successfully downloaded direct stream file ({direct_out.stat().st_size:,} bytes).")
            return direct_out
    except Exception as http_err:
        logger.error(f"Direct HTTP download failed: {http_err}")

    raise RuntimeError("Could not download valid audio file from the specified URL.")



@app.post("/api/summarize-url")
async def summarize_url(body: URLSummarizeRequest):
    """
    Accepts a YouTube video or audio URL, extracts/transcribes the audio, and generates a structured summary.
    """
    if not body.url or not body.url.strip():
        raise HTTPException(status_code=400, detail="No stream/video URL provided.")

    url = body.url.strip()

    try:
        stt_client = ScribeSTTClient()
    except Exception as e:
        logger.error(f"Failed to initialize STT Client: {e}")
        raise HTTPException(status_code=500, detail=f"Speech-to-Text configuration error: {str(e)}")

    ffmpeg_exe = stt_client._get_ffmpeg_exe()
    temp_dir = Path(tempfile.mkdtemp())
    temp_base = temp_dir / "yt_audio"

    try:
        audio_file = await asyncio.get_event_loop().run_in_executor(
            None, _download_youtube_audio, url, temp_base, ffmpeg_exe
        )

        result = await asyncio.to_thread(
            stt_client.transcribe,
            file_path=audio_file,
            diarization=body.diarize,
            tag_audio_events=body.tag_events,
            word_timestamps=True,
            language=body.language,
            engine=stt_client.default_engine
        )

        transcript_text = format_plain_text(result, include_speakers=body.diarize)
        word_timestamps = [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "speaker_id": w.speaker_id,
                "type": w.type
            }
            for w in result.words
        ]

        summary_generator = SummaryGenerator()
        fallback_events = []

        def on_fallback_event(from_model, to_model, err):
            logger.warning(f"Fallback triggered from {from_model} to {to_model}: {err}")
            fallback_events.append({"from_model": from_model, "to_model": to_model, "error": str(err)})

        summary_text = await asyncio.to_thread(
            summary_generator.generate_summary,
            transcript_text=transcript_text,
            on_fallback_callback=on_fallback_event
        )

        actual_model_used = fallback_events[-1]["to_model"] if fallback_events else DEFAULT_PRIMARY_MODEL

        # Surface detected language from STT result
        detected_lang_code = getattr(result, "language_code", None)
        detected_lang_name = get_language_name(detected_lang_code) if detected_lang_code else None

        # Post-Transcription Translation to English via LLM
        logger.info("Translating URL transcript to English via LLM...")
        try:
            english_transcript = await asyncio.to_thread(
                summary_generator.translate_to_english,
                transcript_text=transcript_text,
                detected_language=detected_lang_code,
                on_fallback_callback=on_fallback_event
            )
        except Exception as e:
            logger.warning(f"Translation to English failed: {e}")
            english_transcript = transcript_text

        return {
            "status": "success",
            "url": url,
            "filename": url,
            "file_type": "youtube_url",
            "audio_duration_secs": result.audio_duration_secs,
            "transcript": transcript_text,
            "english_transcript": english_transcript,
            "summary": summary_text,
            "words": word_timestamps,
            "model_used": actual_model_used,
            "primary_model_configured": DEFAULT_PRIMARY_MODEL,
            "fallbacks": fallback_events,
            "token_usage": summary_generator.total_token_usage.to_dict(),
            "detected_language": detected_lang_code,
            "detected_language_name": detected_lang_name,
            "stt_engine_used": getattr(result, "engine_used", None),
            "transcription_time_secs": getattr(result, "transcription_time_secs", None),
        }

    except Exception as e:
        err_msg = str(e)
        if "not a bot" in err_msg or "Sign in" in err_msg:
            err_msg = (
                "YouTube requires bot verification for this video. "
                "To fix this, export your browser cookies to a 'cookies.txt' file in the project folder, "
                "or try a different video / live stream link."
            )
        logger.error(f"URL summarization failed for {url}: {err_msg}")
        raise HTTPException(status_code=400 if "bot verification" in err_msg else 500, detail=err_msg)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TranslateRequest(BaseModel):
    text: str
    target_language: str = "English"  # Full language name, e.g. "English", "Hindi", "Spanish"


@app.post("/api/translate")
async def translate_text(body: TranslateRequest):
    """
    Translates raw transcript text from any detected source language into the specified
    target_language (default: English) using Gemini LLM model with fallback support.
    Pass target_language as a full language name, e.g. "English", "Hindi", "Spanish", "French".
    """
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="No transcript text provided.")

    target_lang = body.target_language.strip() or "English"

    try:
        summary_generator = SummaryGenerator()
        fallback_events = []

        def on_fallback_event(from_model, to_model, err):
            fallback_events.append({"from_model": from_model, "to_model": to_model, "error": str(err)})

        translated_text = await asyncio.to_thread(
            summary_generator.translate_text,
            transcript_text=body.text.strip(),
            target_language=target_lang,
            on_fallback_callback=on_fallback_event
        )

        actual_model_used = fallback_events[-1]["to_model"] if fallback_events else DEFAULT_PRIMARY_MODEL

        return {
            "status": "success",
            "model_used": actual_model_used,
            "target_language": target_lang,
            "translated_text": translated_text,
            "fallbacks": fallback_events,
            "token_usage": summary_generator.total_token_usage.to_dict()
        }
    except Exception as e:
        logger.error(f"Translation to '{target_lang}' failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/api/stream-transcribe")
async def stream_live(body: StreamRequest):
    """
    Streams live transcription from a URL (HLS / RTMP / HTTP audio) via Server-Sent Events.

    ffmpeg taps into the live feed, extracts raw 16-bit mono PCM audio in real time,
    and the backend slices it into N-second chunks. Each chunk is written to a temp
    WAV file and sent to ElevenLabs Scribe STT. The resulting text is immediately
    pushed to the browser as an SSE event so the transcript appears word-by-word live.

    SSE Event types emitted:
      event: session     → {"session_id": "<uuid>"}
      event: transcript  → {"chunk_index": N, "timestamp_secs": T, "text": "...", "words": [...]}
      event: chunkError  → {"chunk_index": N, "detail": "..."}
      event: error       → {"detail": "..."}   (fatal — stream will stop)
      event: done        → {"session_id": "<uuid>"}
    """
    try:
        stt_client = ScribeSTTClient()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scribe STT init failed: {str(e)}")

    ffmpeg_exe = stt_client._get_ffmpeg_exe()
    session_id = str(uuid.uuid4())

    # Resolve YouTube / social-media URLs to a direct stream URL
    # (done once here so the SSE generator can yield an error event if it fails)
    try:
        resolved_url, is_yt_live = await asyncio.get_event_loop().run_in_executor(
            None, _resolve_stream_url, body.stream_url, ffmpeg_exe
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not resolve stream URL: {e}"
        )

    SAMPLE_RATE      = 16000
    BYTES_PER_SAMPLE = 2          # s16le = 16-bit signed LE → 2 bytes per sample
    CHANNELS         = 1          # mono
    chunk_bytes      = body.chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS

    async def event_stream():
        # Immediately hand the client a session ID and resolved stream URL so UI can play video
        yield f"event: session\ndata: {json.dumps({'session_id': session_id, 'resolved_url': resolved_url, 'is_yt_live': is_yt_live})}\n\n"

        proc = None
        ytdlp_proc = None
        stderr_lines = []

        is_live_stream = (
            is_yt_live
            or ".m3u8" in resolved_url.lower()
            or "rtmp://" in resolved_url.lower()
            or "rtsp://" in resolved_url.lower()
            or "live" in body.stream_url.lower()
            or "youtube.com" in body.stream_url.lower()
            or "youtu.be" in body.stream_url.lower()
            or "twitch.tv" in body.stream_url.lower()
        )

        try:
            if is_yt_live:
                ffmpeg_dir = os.path.dirname(ffmpeg_exe)
                for alias in ["ffmpeg", "ffmpeg.exe"]:
                    target = os.path.join(ffmpeg_dir, alias)
                    if not os.path.exists(target):
                        try:
                            shutil.copy2(ffmpeg_exe, target)
                            os.chmod(target, 0o755)
                        except Exception:
                            pass

                cookies_path = Path("cookies.txt")
                ytdlp_cmd = [
                    sys.executable, "-m", "yt_dlp",
                    "--ffmpeg-location", ffmpeg_dir,
                    "--extractor-args", "youtube:player_client=mweb,ios,android,tv_embedded,web",
                    "-f", "bestaudio/best",
                    "-o", "-",
                    "--quiet",
                ]
                if cookies_path.exists():
                    ytdlp_cmd.extend(["--cookies", str(cookies_path.resolve())])

                ytdlp_cmd.append(resolved_url)

                logger.info(f"[{session_id[:8]}] Spawning yt-dlp pipe for YouTube LIVE stream: {resolved_url}")
                ytdlp_proc = subprocess.Popen(
                    ytdlp_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                cmd = [
                    ffmpeg_exe,
                    "-i", "pipe:0",
                    "-vn",
                    "-ac", str(CHANNELS),
                    "-ar", str(SAMPLE_RATE),
                    "-f", "s16le",
                    "pipe:1",
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdin=ytdlp_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            else:
                # Expand protocol whitelist to support all common stream types
                cmd = [
                    ffmpeg_exe,
                    "-protocol_whitelist", "file,http,https,tcp,tls,crypto,hls,rtmp,rtsp,udp,rtp,pipe",
                    "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "-headers", "ngrok-skip-browser-warning: 69420\r\nAccept: */*\r\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\r\n",
                    "-seekable", "0",
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-reconnect_on_network_error", "1",
                    "-reconnect_on_http_error", "4xx,5xx",
                    "-rw_timeout", "15000000",
                    "-fflags", "+genpts+discardcorrupt",
                    "-flags", "low_delay",
                    "-i", resolved_url,
                    "-vn",
                    "-ac", str(CHANNELS),
                    "-ar", str(SAMPLE_RATE),
                    "-f", "s16le",
                    "pipe:1",
                ]
                # For VOD/file links, pace with -re BEFORE -i. For all live streams, skip -re.
                if body.realtime and not is_live_stream:
                    cmd.insert(cmd.index("-i"), "-re")

                logger.info(f"[{session_id[:8]}] Starting ffmpeg: {' '.join(cmd[:8])}...")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,   # capture stderr for diagnostics
                    bufsize=0,
                )

            active_stream_sessions[session_id] = proc

            # --- Drain stderr in background threads so processes don't block ---
            import threading
            def _drain_stream(stream, tag=""):
                for line in iter(stream.readline, b""):
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        prefix = f"[{tag}] " if tag else ""
                        stderr_lines.append(f"{prefix}{decoded}")
                        if len(stderr_lines) > 100:
                            stderr_lines.pop(0)

            if proc and proc.stderr:
                threading.Thread(target=_drain_stream, args=(proc.stderr, "ffmpeg"), daemon=True).start()
            if ytdlp_proc and ytdlp_proc.stderr:
                threading.Thread(target=_drain_stream, args=(ytdlp_proc.stderr, "yt-dlp"), daemon=True).start()

            # --- Wait for ffmpeg to produce initial PCM byte ---
            loop = asyncio.get_event_loop()
            first_byte = b""
            try:
                first_byte = await asyncio.wait_for(
                    loop.run_in_executor(None, proc.stdout.read, 1),
                    timeout=35.0
                )
            except (asyncio.TimeoutError, Exception):
                first_byte = b""

            if not first_byte:
                exit_code = proc.poll()
                ffmpeg_err = "\n".join(stderr_lines[-15:]) if stderr_lines else "No stderr output captured."
                if exit_code is not None and exit_code != 0:
                    if any(k in ffmpeg_err for k in ["503", "5XX", "Service Unavailable", "Server Error", "3200", "offline"]):
                        error_msg = (
                            f"Remote stream server returned 503 Service Unavailable / Offline for '{resolved_url}'. "
                            "The ngrok tunnel or origin media server is offline, unreachable, or crashed."
                        )
                    else:
                        error_msg = f"ffmpeg exited with code {exit_code}: {ffmpeg_err}"
                else:
                    error_msg = f"Stream timed out — ffmpeg did not produce audio within 35s. Check the URL. {ffmpeg_err}"
                logger.error(f"[{session_id[:8]}] {error_msg}")
                yield f"event: error\ndata: {json.dumps({'detail': error_msg})}\n\n"
                return

            logger.info(f"[{session_id[:8]}] ffmpeg producing audio — transcription starting.")

            chunk_index = 0

            def _read_pcm_chunk(seed_byte: bytes = b"") -> bytes:
                """
                Blocking helper: read exactly chunk_bytes of raw PCM from ffmpeg stdout.
                Called in a thread-pool executor to avoid blocking the event loop.
                seed_byte: any byte(s) already read before the loop started.
                """
                buf = seed_byte
                while len(buf) < chunk_bytes:
                    try:
                        data = proc.stdout.read(chunk_bytes - len(buf))
                    except Exception:
                        break
                    if not data:
                        break
                    buf += data
                return buf

            def _transcribe_chunk(pcm_data: bytes, idx: int) -> dict:
                """
                Blocking helper: wrap PCM in WAV container, write to a temp file,
                call ElevenLabs Scribe STT, clean up, and return a serialisable result dict.
                """
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(BYTES_PER_SAMPLE)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm_data)
                wav_bytes = wav_buf.getvalue()

                tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
                try:
                    os.write(tmp_fd, wav_bytes)
                    os.close(tmp_fd)
                    result = stt_client._transcribe_file_chunk(
                        file_path=Path(tmp_name),
                        word_timestamps=True,
                        language=body.language if (body.language and body.language != "auto") else None,
                        engine=stt_client.default_engine
                    )
                    detected_lang_code = getattr(result, "language_code", None)
                    detected_lang_name = get_language_name(detected_lang_code) if detected_lang_code else None

                    english_chunk_text = result.raw_text
                    if result.raw_text and result.raw_text.strip():
                        try:
                            summary_gen = SummaryGenerator()
                            english_chunk_text = summary_gen.translate_to_english(
                                transcript_text=result.raw_text,
                                detected_language=detected_lang_code
                            )
                        except Exception as e:
                            logger.warning(f"Live chunk translation error: {e}")

                    raw_txt = result.raw_text or ""
                    display_text = english_chunk_text if (body.english_only and english_chunk_text) else raw_txt

                    stt_cost = calculate_stt_cost_usd(body.chunk_seconds)
                    llm_cost = summary_gen.last_token_usage.cost_usd if 'summary_gen' in locals() else 0.0
                    tot_cost = round(stt_cost + llm_cost, 6)

                    return {
                        "chunk_index": idx,
                        "timestamp_secs": idx * body.chunk_seconds,
                        "text": display_text,
                        "original_text": raw_txt,
                        "english_text": english_chunk_text,
                        "english_only": body.english_only,
                        "words": [
                            {"text": w.text, "start": w.start, "end": w.end}
                            for w in result.words if w.type == "word"
                        ],
                        "detected_language": detected_lang_code,
                        "detected_language_name": detected_lang_name,
                        "token_usage": summary_gen.last_token_usage.to_dict() if 'summary_gen' in locals() else None,
                        "stt_cost_usd": stt_cost,
                        "llm_cost_usd": llm_cost,
                        "total_cost_usd": tot_cost,
                    }
                finally:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass

            # First chunk uses the seed byte we already read
            seed = first_byte
            while True:
                chunk_start_time = asyncio.get_event_loop().time()
                pcm = await loop.run_in_executor(None, _read_pcm_chunk, seed)
                seed = b""  # only use seed on first iteration

                if not pcm:
                    exit_code = proc.poll()
                    if exit_code is not None and exit_code != 0:
                        ffmpeg_err = "\n".join(stderr_lines[-8:]) if stderr_lines else "No error info captured."
                        logger.error(f"[{session_id[:8]}] ffmpeg exited with code {exit_code}: {ffmpeg_err}")
                        yield f"event: error\ndata: {json.dumps({'detail': f'ffmpeg exited ({exit_code}): {ffmpeg_err}' })}\n\n"
                    else:
                        logger.info(f"[{session_id[:8]}] Stream ended normally after {chunk_index} chunk(s).")
                    break

                if len(pcm) < BYTES_PER_SAMPLE * SAMPLE_RATE * CHANNELS:
                    # Got very few bytes — stream may be ending, still try to transcribe
                    logger.info(f"[{session_id[:8]}] Short final chunk ({len(pcm)} bytes), transcribing anyway.")

                try:
                    result = await loop.run_in_executor(None, _transcribe_chunk, pcm, chunk_index)
                    if result["text"]:
                        logger.info(
                            f"[{session_id[:8]}] Chunk {chunk_index} @ "
                            f"{result['timestamp_secs']}s: \"{result['text'][:80]}\""
                        )
                        yield f"event: transcript\ndata: {json.dumps(result)}\n\n"
                    else:
                        # Empty transcript — send a heartbeat to keep connection alive
                        yield ": silence\n\n"
                except Exception as exc:
                    logger.error(f"[{session_id[:8]}] Chunk {chunk_index} transcription error: {exc}")
                    yield (
                        f"event: chunkError\ndata: "
                        f"{json.dumps({'chunk_index': chunk_index, 'detail': str(exc)})}\n\n"
                    )

                chunk_index += 1

                # For live streams, skip artificial pacing — don't sleep, read continuously
                if body.realtime and not is_live_stream:
                    elapsed = asyncio.get_event_loop().time() - chunk_start_time
                    remaining_pace = body.chunk_seconds - elapsed
                    if remaining_pace > 0:
                        await asyncio.sleep(min(remaining_pace, body.chunk_seconds))

        except asyncio.CancelledError:
            logger.info(f"[{session_id[:8]}] Stream cancelled by client disconnect.")
        except asyncio.TimeoutError:
            logger.warning(f"[{session_id[:8]}] Timeout reading from ffmpeg.")
            yield f"event: error\ndata: {json.dumps({'detail': 'Timed out waiting for stream data from ffmpeg.'})}\n\n"
        except Exception as exc:
            logger.error(f"[{session_id[:8]}] Fatal stream error: {exc}")
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
        finally:
            if ytdlp_proc and ytdlp_proc.poll() is None:
                ytdlp_proc.terminate()
                try:
                    ytdlp_proc.wait(timeout=3)
                except Exception:
                    ytdlp_proc.kill()
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            active_stream_sessions.pop(session_id, None)
            logger.info(f"[{session_id[:8]}] Session cleaned up.")
            yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.delete("/api/stream-transcribe/{session_id}")
async def stop_stream(session_id: str):
    """
    Stop an active live transcription session.
    The running ffmpeg process is terminated and the session is removed.
    """
    proc = active_stream_sessions.pop(session_id, None)
    if not proc:
        raise HTTPException(
            status_code=404,
            detail="Session not found or already stopped."
        )
    if proc.poll() is None:
        proc.terminate()
    return {"status": "stopped", "session_id": session_id}


@app.post("/api/transcribe-chunk")
async def transcribe_chunk(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    elevenlabs_api_key: Optional[str] = Form(None),
    english_only: bool = Form(False)
):
    """
    Accepts a short audio blob (e.g. 2-5 sec WebM/WAV from browser MediaRecorder),
    standardizes it to 16kHz mono WAV via FFmpeg, transcribes it using ElevenLabs Scribe,
    and returns transcript text and word timestamps.
    """
    if not file.filename:
        file.filename = "mic_chunk.webm"

    try:
        stt_client = ScribeSTTClient(elevenlabs_api_key=elevenlabs_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT init error: {str(e)}")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    clean_wav_path = Path(temp_dir) / "clean_chunk.wav"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Standardize audio format via FFmpeg to 16kHz mono WAV + volume boost for high STT accuracy
        ffmpeg_exe = stt_client._get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", str(temp_path),
            "-af", "volume=2.0,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac", "1", "-ar", "16000", str(clean_wav_path)
        ]
        res = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, errors="replace")
        if res.returncode != 0:
            # Fallback to simple volume boost if loudnorm is unavailable
            cmd_simple = [
                ffmpeg_exe, "-y", "-i", str(temp_path),
                "-af", "volume=2.0",
                "-ac", "1", "-ar", "16000", str(clean_wav_path)
            ]
            res = await asyncio.to_thread(subprocess.run, cmd_simple, capture_output=True, text=True, errors="replace")
        target_path = clean_wav_path if (clean_wav_path.exists() and clean_wav_path.stat().st_size >= 44) else temp_path

        result = await asyncio.to_thread(
            stt_client._transcribe_file_chunk,
            file_path=target_path,
            word_timestamps=True,
            language=language if (language and language != "auto") else None,
            engine=engine or "elevenlabs",
            elevenlabs_api_key=elevenlabs_api_key
        )

        detected_lang_code = getattr(result, "language_code", None)
        detected_lang_name = get_language_name(detected_lang_code) if detected_lang_code else None

        english_chunk_text = result.raw_text
        if result.raw_text and result.raw_text.strip():
            try:
                summary_gen = SummaryGenerator()
                english_chunk_text = summary_gen.translate_to_english(
                    transcript_text=result.raw_text,
                    detected_language=detected_lang_code
                )
            except Exception as e:
                logger.warning(f"Mic chunk translation error: {e}")

        raw_txt = result.raw_text or ""
        display_text = english_chunk_text if (english_only and english_chunk_text) else raw_txt

        stt_cost = calculate_stt_cost_usd(4.0)
        llm_cost = summary_gen.last_token_usage.cost_usd if 'summary_gen' in locals() else 0.0
        tot_cost = round(stt_cost + llm_cost, 6)

        return {
            "status": "success",
            "text": display_text,
            "original_text": raw_txt,
            "english_text": english_chunk_text,
            "english_only": english_only,
            "words": [
                {"text": w.text, "start": w.start, "end": w.end}
                for w in result.words
                if w.type == "word"
            ],
            "engine_used": result.engine_used,
            "detected_language": detected_lang_code,
            "detected_language_name": detected_lang_name,
            "token_usage": summary_gen.last_token_usage.to_dict() if 'summary_gen' in locals() else None,
            "stt_cost_usd": stt_cost,
            "llm_cost_usd": llm_cost,
            "total_cost_usd": tot_cost,
        }
    except Exception as e:
        logger.warning(f"Chunk transcription skipped: {e}")
        return {
            "status": "success",
            "text": "",
            "words": [],
            "engine_used": "skipped",
            "detected_language": None,
            "detected_language_name": None,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



# Ensure static folder exists before mounting
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "API running! Please create index.html inside the static folder."})
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ScribeAI Summarizer Server at http://0.0.0.0:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, reload_includes=["*.py"])
