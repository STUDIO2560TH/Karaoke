"""
Auto Sync Lyrics Tool
=====================
Downloads audio from a YouTube URL, uses Whisper to detect vocal timing,
and matches user-provided lyrics to produce a Lua Subtitles table for
the Roblox karaoke module.

Usage:
    python sync_lyrics.py lyrics/SongName.txt [--model small] [--output output/]
"""

import argparse
import os
import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def parse_lyrics_file(filepath: str) -> tuple[str, list[str]]:
    """
    Parse a lyrics .txt file.
    First line starting with 'url:' is the YouTube URL.
    Remaining non-empty lines are the lyrics.
    """
    url = None
    lyrics = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("url:"):
                url = line[4:].strip()
            elif line:
                lyrics.append(line)

    if not url:
        print(f"ERROR: No 'url:' line found in {filepath}")
        sys.exit(1)

    if not lyrics:
        print(f"ERROR: No lyrics found in {filepath}")
        sys.exit(1)

    return url, lyrics


def download_audio(url: str, output_path: str) -> str:
    """Download audio from YouTube URL using yt-dlp, returns path to audio file."""
    audio_file = os.path.join(output_path, "audio.wav")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", os.path.join(output_path, "audio.%(ext)s"),
        "--no-playlist",
        "--extractor-args", "youtube:player-client=android,web",
        "--no-check-certificate",
        "--prefer-free-formats",
        "--add-header", "Accept-Language:en-US,en;q=0.5",
        url,
    ]

    print(f"Downloading audio from: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR downloading audio:\n{result.stderr}")
        sys.exit(1)

    # yt-dlp might create the file with a different name during conversion
    # Find the .wav file in the output directory
    for f in os.listdir(output_path):
        if f.endswith(".wav"):
            audio_file = os.path.join(output_path, f)
            break

    if not os.path.exists(audio_file):
        print("ERROR: Audio file not found after download")
        sys.exit(1)

    print(f"Audio saved to: {audio_file}")
    return audio_file


def transcribe_audio(audio_path: str, model_size: str = "small") -> list[dict]:
    """
    Transcribe audio using faster-whisper.
    Returns list of segments: [{"start": float, "end": float, "text": str}, ...]
    """
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}' (this may take a moment)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("Transcribing audio (Thai language)...")
    segments_gen, info = model.transcribe(
        audio_path,
        language="th",
        beam_size=5,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
    )

    segments = []
    for seg in segments_gen:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        print(f"  [{format_time(seg.start)}] {seg.text.strip()}")

    print(f"Found {len(segments)} segments")
    return segments


def normalize_text(text: str) -> str:
    """Normalize text for comparison — remove spaces, punctuation, lowercase."""
    # Remove common non-lyric characters
    text = re.sub(r'[^\u0E00-\u0E7F\w]', '', text)  # Keep Thai chars and word chars
    text = text.lower().strip()
    return text


def similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def is_instrumental(line: str) -> bool:
    """Check if a lyrics line is an instrumental/music marker."""
    stripped = line.strip()
    return stripped in ("🎶", "🎵", "♪", "♫", "...")


def match_lyrics_to_segments(
    user_lyrics: list[str],
    segments: list[dict],
) -> list[tuple[float, str]]:
    """
    Match user-provided lyrics lines to Whisper segments.

    Uses greedy forward matching with text similarity,
    enforcing chronological order.

    Returns: [(timestamp_seconds, lyric_line), ...]
    """
    results = []
    seg_idx = 0
    last_end_time = 0.0

    for lyric_line in user_lyrics:
        # Handle instrumental markers
        if is_instrumental(lyric_line):
            # Place at the end of the previous segment (or 0 for start)
            results.append((last_end_time, lyric_line))
            continue

        if seg_idx >= len(segments):
            # No more Whisper segments — estimate timestamp
            results.append((last_end_time + 1.0, lyric_line))
            last_end_time += 3.0
            continue

        # Try matching this lyric line against upcoming segments
        best_score = 0.0
        best_start_time = segments[seg_idx]["start"]
        best_end_idx = seg_idx
        look_ahead = min(seg_idx + 8, len(segments))

        for start in range(seg_idx, look_ahead):
            combined_text = ""
            for end in range(start, min(start + 4, len(segments))):
                combined_text += segments[end]["text"]
                score = similarity(lyric_line, combined_text)

                if score > best_score:
                    best_score = score
                    best_start_time = segments[start]["start"]
                    best_end_idx = end

        # Accept match if score is reasonable
        if best_score >= 0.25:
            results.append((best_start_time, lyric_line))
            last_end_time = segments[best_end_idx]["end"]
            seg_idx = best_end_idx + 1
        else:
            # Low confidence — use the next segment's timestamp anyway
            # (Whisper might have transcribed it very differently)
            results.append((segments[seg_idx]["start"], lyric_line))
            last_end_time = segments[seg_idx]["end"]
            seg_idx += 1

    return results


def format_time(seconds: float) -> str:
    """Convert seconds to M:SS format (e.g., 1:35)."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def generate_lua_subtitles(matched: list[tuple[float, str]]) -> str:
    """Generate the Lua Subtitles table from matched lyrics."""
    lines = []
    lines.append('["Subtitles"] =')
    lines.append('{')

    for timestamp, lyric in matched:
        time_str = format_time(timestamp)
        # Escape special Lua characters in lyrics
        escaped = lyric.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'\t["{time_str}"] = "{escaped}",')

    lines.append('},')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Auto Sync Lyrics Tool")
    parser.add_argument("input", help="Path to lyrics .txt file")
    parser.add_argument(
        "--model", default="small",
        help="Whisper model size: tiny, base, small, medium, large (default: small)"
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory for .lua files (default: output/)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Parse lyrics file
    url, lyrics = parse_lyrics_file(str(input_path))
    print(f"Song: {input_path.stem}")
    print(f"URL: {url}")
    print(f"Lyrics lines: {len(lyrics)}")
    print()

    # Create temp directory for audio
    tmp_dir = Path("tmp_audio")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Download audio
        audio_file = download_audio(url, str(tmp_dir))
        print()

        # Step 2: Transcribe with Whisper
        segments = transcribe_audio(audio_file, args.model)
        print()

        # Step 3: Match lyrics to segments
        print("Matching lyrics to timestamps...")
        matched = match_lyrics_to_segments(lyrics, segments)
        print()

        # Step 4: Generate Lua output
        lua_output = generate_lua_subtitles(matched)

        # Print to console
        print("=" * 60)
        print("GENERATED SUBTITLES:")
        print("=" * 60)
        print(lua_output)
        print("=" * 60)

        # Save to file
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{input_path.stem}.lua"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(lua_output + "\n")

        print(f"\nSaved to: {output_file}")

    finally:
        # Cleanup temp audio files
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print("Cleaned up temporary files.")


if __name__ == "__main__":
    main()
