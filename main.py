import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

from stt_client import ScribeSTTClient, TranscriptionResult
from exporter import format_plain_text, save_transcript_to_file
from summary_generator import SummaryGenerator

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="ElevenLabs Scribe Speech-to-Text & Gemini Summary CLI"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=str,
        default="audio.mp4",
        help="Path to the input audio file (default: audio.mp4)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the generated plain text transcript (default: <audio_file_name>_transcript.txt)"
    )
    parser.add_argument(
        "-d", "--diarize",
        dest="diarize",
        action="store_true",
        default=True,
        help="Enable speaker diarization (groups text by Speaker ID, enabled by default)"
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Disable speaker diarization"
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable tagging of non-speech audio events (e.g. laughter, applause)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Target language code (e.g., 'en', 'hi', 'te', 'es') to optimize transcription accuracy"
    )
    parser.add_argument(
        "-e", "--engine",
        type=str,
        choices=["elevenlabs"],
        default="elevenlabs",
        help="STT engine to use: 'elevenlabs' (ElevenLabs Scribe v1)"
    )
    parser.add_argument(
        "-s", "--summarize",
        dest="summarize",
        action="store_true",
        default=True,
        help="Generate an executive summary using Gemini (enabled by default)"
    )
    parser.add_argument(
        "--no-summarize",
        dest="summarize",
        action="store_false",
        help="Disable executive summary generation"
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Path to save the generated summary (default: <audio_file_name>_summary.txt)"
    )
    parser.add_argument(
        "-t", "--translate",
        dest="translate",
        action="store_true",
        default=True,
        help="Translate transcript to English using LLM after generation (enabled by default)"
    )
    parser.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        help="Disable LLM translation to English"
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        default=False,
        help="Pipeline output mode: Output and save translated English transcript directly as primary transcript"
    )
    return parser.parse_args()


def handle_fallback(from_model: str, to_model: str, error_msg: str):
    console.print(f"[bold yellow]⚠️ Primary model '{from_model}' failed ({error_msg}).[/bold yellow]")
    console.print(f"[bold cyan]🔄 Falling back to alternative model: [white]{to_model}[/white]...[/bold cyan]")


def main():
    load_dotenv()
    args = parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        console.print(f"[bold red]❌ Error: Audio file not found at '{audio_path}'[/bold red]")
        sys.exit(1)

    # Determine output file paths
    default_base = audio_path.stem
    transcript_out_path = Path(args.output) if args.output else Path(f"{default_base}_transcript.txt")
    summary_out_path = Path(args.summary_output) if args.summary_output else Path(f"{default_base}_summary.txt")

    console.print(Panel.fit(
        "[bold magenta]🎙️  ElevenLabs Scribe STT & Gemini Summary Tool[/bold magenta]",
        border_style="magenta"
    ))

    # 1. Transcription step
    try:
        stt_client = ScribeSTTClient()
    except Exception as e:
        console.print(f"[bold red]❌ STT Client Initialization Failed: {e}[/bold red]")
        sys.exit(1)

    active_engine = args.engine or stt_client.default_engine
    console.print(f"\n[cyan]File:[/cyan] [bold white]{audio_path.resolve()}[/bold white]")
    console.print(f"[cyan]Options:[/cyan] Engine=[bold green]{active_engine}[/bold green], Diarize=[bold green]{args.diarize}[/bold green], Events=[bold green]{not args.no_events}[/bold green], Language=[bold green]{args.language or 'Auto-Detect'}[/bold green]\n")

    with console.status(f"[bold cyan]Transcribing audio using {active_engine}...[/bold cyan]", spinner="dots"):
        try:
            result = stt_client.transcribe(
                file_path=audio_path,
                diarization=args.diarize,
                tag_audio_events=not args.no_events,
                word_timestamps=True,
                language=args.language,
                engine=args.engine
            )
        except Exception as e:
            console.print(f"[bold red]❌ Transcription failed: {e}[/bold red]")
            sys.exit(1)

    # Format plain text transcript
    transcript_text = format_plain_text(result, include_speakers=args.diarize)
    
    # Display transcript in terminal
    console.print(Panel(
        Text(transcript_text, style="white"),
        title="[bold green]📝 Transcription Result (Plain Text)[/bold green]",
        border_style="green"
    ))

    # Save to file
    saved_path = save_transcript_to_file(result, transcript_out_path, include_speakers=args.diarize)
    console.print(f"[bold green]✔ Saved plain text transcript to:[/bold green] [underline white]{saved_path.resolve()}[/underline white]\n")

    summary_generator = SummaryGenerator()

    # 2. LLM Post-Transcription Translation step
    if args.translate:
        try:
            detected_lang = getattr(result, "language_code", None)
            with console.status("[bold cyan]Translating transcript to English with LLM...[/bold cyan]", spinner="earth"):
                english_transcript = summary_generator.translate_to_english(
                    transcript_text=transcript_text,
                    detected_language=detected_lang,
                    on_fallback_callback=handle_fallback
                )

            if english_transcript != transcript_text:
                console.print(Panel(
                    Text(english_transcript, style="white"),
                    title="[bold blue]🌐 English Translation (LLM Transformed)[/bold blue]",
                    border_style="blue"
                ))
                eng_out_path = Path(f"{default_base}_english_transcript.txt")
                with open(eng_out_path, "w", encoding="utf-8") as ef:
                    ef.write(english_transcript)
                console.print(f"[bold green]✔ Saved English transcript to:[/bold green] [underline white]{eng_out_path.resolve()}[/underline white]\n")
        except Exception as e:
            console.print(f"[bold red]❌ English translation failed: {e}[/bold red]\n")

    # 3. Optional Summarization step
    if args.summarize:
        try:
            with console.status("[bold cyan]Generating executive summary with LLM (with fallback support)...[/bold cyan]", spinner="earth"):
                summary = summary_generator.generate_summary(
                    transcript_text=transcript_text,
                    on_fallback_callback=handle_fallback
                )
            
            console.print(Panel(
                Text(summary, style="white"),
                title="[bold yellow]💡 Executive Summary & Action Items[/bold yellow]",
                border_style="yellow"
            ))

            # Save summary to disk
            with open(summary_out_path, "w", encoding="utf-8") as sf:
                sf.write(summary)
                if not summary.endswith("\n"):
                    sf.write("\n")
            console.print(f"[bold green]✔ Saved executive summary to:[/bold green] [underline white]{summary_out_path.resolve()}[/underline white]\n")

            if summary_generator.total_token_usage.total_tokens > 0:
                usage = summary_generator.total_token_usage
                console.print(Panel.fit(
                    f"[bold cyan]📊 Token Usage (LangSmith Monitored):[/bold cyan]\n"
                    f"• Model: [bold white]{usage.model_name or 'LLM'}[/bold white]\n"
                    f"• Prompt Tokens: [bold green]{usage.prompt_tokens:,}[/bold green]\n"
                    f"• Completion Tokens: [bold green]{usage.completion_tokens:,}[/bold green]\n"
                    f"• Total Tokens: [bold yellow]{usage.total_tokens:,}[/bold yellow]",
                    title="[bold dim]LangSmith Telemetry[/bold dim]",
                    border_style="cyan"
                ))


        except Exception as e:
            console.print(f"[bold red]❌ Summarization failed: {e}[/bold red]\n")


if __name__ == "__main__":
    main()