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


def parse_lyrics_file(filepath: str) -> tuple[str, list[str], float, str]:
    """
    Parse a lyrics .txt file.
    Tags:
    - url: YouTube URL
    - offset: Decimal offset in seconds (e.g., -0.2)
    - model: Whisper model size (tiny, base, small, medium, large)
    """
    url = None
    lyrics = []
    offset = 0.0
    model_size = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            if line_lower.startswith("url:"):
                url = line[4:].strip()
            elif line_lower.startswith("offset:"):
                try:
                    offset = float(line[7:].strip())
                except ValueError:
                    print(f"WARNING: Invalid offset value in {filepath}")
            elif line_lower.startswith("model:"):
                model_size = line[6:].strip()
            else:
                lyrics.append(line)

    if not url:
        print(f"NOTE: No 'url:' line found in {filepath}. Script will strictly require a local audio file.")
    
    if not lyrics:
        print(f"ERROR: No lyrics found in {filepath}")
        sys.exit(1)

    return url, lyrics, offset, model_size


def download_audio(url: str, output_path: str, song_name: str, lyrics_dir: str = "lyrics") -> str:
    """
    Check for local audio file first, otherwise download from YouTube.
    Returns path to a usable audio file (translated to WAV if necessary).
    """
    # Step 0: Check for local audio file first
    audio_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac"]
    local_audio = None
    
    # Try to find a file with the same name as the lyrics file in the lyrics directory
    if os.path.exists(lyrics_dir):
        for ext in audio_extensions:
            potential_file = os.path.join(lyrics_dir, f"{song_name}{ext}")
            if os.path.exists(potential_file):
                local_audio = potential_file
                break
    
    if local_audio:
        print(f"Using local audio file found: {local_audio}")
        # If it's already a wav, just copy it to tmp (or use it directly)
        target_wav = os.path.join(output_path, "audio.wav")
        if local_audio.lower().endswith(".wav"):
            import shutil
            shutil.copy(local_audio, target_wav)
        else:
            # Convert to wav using ffmpeg
            print(f"Converting {local_audio} to wav...")
            cmd = ["ffmpeg", "-i", local_audio, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", target_wav, "-y"]
            subprocess.run(cmd, capture_output=True)
            
        if os.path.exists(target_wav):
            return target_wav
        else:
            print("ERROR: Failed to prepare local audio file.")
            sys.exit(1)

    # Step 1: Download from YouTube if no local file (only if URL is provided)
    if not url:
        print("ERROR: No YouTube URL provided and no matching local audio file found.")
        print(f"Place a song file (e.g. {song_name}.mp3) in the same folder as your lyrics.")
        sys.exit(1)

    audio_file = os.path.join(output_path, "audio.wav")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", os.path.join(output_path, "audio.%(ext)s"),
        "--no-playlist",
        "--extractor-args", "youtube:player-client=ios,web",
        "--no-check-certificate",
        "--prefer-free-formats",
        "--add-header", "Accept-Language:en-US,en;q=0.5",
        url,
    ]

    print(f"Downloading audio from: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR downloading audio:\n{result.stderr}")
        print("\nTIP: If YouTube is blocking the download, try uploading the song file (.mp3 or .wav)")
        print(f"directly to the 'lyrics/' folder with the same name as your text file.")
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
        word_timestamps=True,  # Precision improvement
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400,  # Adjusted for singing
        ),
    )

    segments = []
    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "text": w.word.strip()
                })
        
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words
        })
        print(f"  [{format_time(seg.start)}] {seg.text.strip()}")

    print(f"Found {len(segments)} segments")
    return segments


def normalize_text(text: str) -> str:
    """Normalize text for comparison — remove spaces, punctuation, tone marks, etc."""
    # 1. Lowercase and strip
    text = text.lower().strip()
    # 2. Remove Thai tone marks (may help with fuzzy matching if transcription is slightly off)
    # Thai tone marks: \u0E48 (mai ek), \u0E49 (mai tho), \u0E4A (mai tri), \u0E4B (mai chattawa)
    # Also \u0E4C (thanthakhat), \u0E4D (nikhahit), \u0E47 (mai taikhu)
    text = re.sub(r'[\u0E47-\u0E4E]', '', text)
    # 3. Remove common non-lyric characters, keep Thai chars and alphanumeric
    text = re.sub(r'[^\u0E00-\u0E7F\w]', '', text)
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
    base_offset: float = -0.20
) -> list[tuple[float, str]]:
    """
    Match user-provided lyrics lines to Whisper segments using word-level precision.
    """
    results = []
    seg_idx = 0
    last_end_time = 0.0
    
    # Use base_offset (default or from .txt file)
    OFFSET = base_offset 

    for lyric_line in user_lyrics:
        # 1. Handle instrumental markers
        if is_instrumental(lyric_line):
            results.append((max(0, last_end_time + OFFSET), lyric_line))
            continue

        # 2. Check if we've run out of audio segments
        if seg_idx >= len(segments):
            results.append((last_end_time + 1.5, lyric_line))
            last_end_time += 3.0
            continue

        # 3. Try matching this lyric line against upcoming segments (best of N)
        best_score = 0.0
        best_start_time = segments[seg_idx]["start"]
        best_end_idx = seg_idx
        
        # Increase lookahead for singing (words can be stretched/transcribed late)
        look_ahead = min(seg_idx + 15, len(segments))

        for start in range(seg_idx, look_ahead):
            combined_text = ""
            for end in range(start, min(start + 5, len(segments))):
                combined_text += segments[end]["text"]
                score = similarity(lyric_line, combined_text)

                if score > best_score:
                    best_score = score
                    # Use first word start if available for higher precision
                    if segments[start].get("words") and len(segments[start]["words"]) > 0:
                        best_start_time = segments[start]["words"][0]["start"]
                    else:
                        best_start_time = segments[start]["start"]
                    best_end_idx = end
            
            # If we found an extremely high confidence match, stop looking further
            if best_score > 0.9:
                break

        # 4. Resolve match
        if best_score >= 0.20: # Lowered threshold slightly for singing, but added resilience
            final_time = best_start_time + OFFSET
            results.append((max(0, final_time), lyric_line))
            last_end_time = segments[best_end_idx]["end"]
            seg_idx = best_end_idx + 1
        else:
            # Low confidence - skip current vocal segment to catch up? 
            # Or just use current segment if it's "mostly" quiet
            final_time = segments[seg_idx]["start"] + OFFSET
            results.append((max(0, final_time), lyric_line))
            last_end_time = segments[seg_idx]["end"]
            seg_idx += 1

    return results

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
        "--model", default="medium",
        help="Whisper model size: tiny, base, small, medium, large (default: medium)"
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

    # Skip audio files if they are passed as the main input (the workflow loop might hit them)
    if input_path.suffix.lower() in [".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac"]:
        print(f"Skipping audio file: {input_path} (it will be used as a source for the corresponding lyrics file)")
        return

    # Parse lyrics file
    url, lyrics, file_offset, file_model = parse_lyrics_file(str(input_path))
    
    # Priority: model from file > model from command line argument
    model_to_use = file_model if file_model else args.model
    
    print(f"Song: {input_path.stem}")
    if url:
        print(f"URL: {url}")
    print(f"Lyrics lines: {len(lyrics)}")
    print(f"Sync Offset: {file_offset}s")
    print(f"Whisper Model: {model_to_use}")
    print()

    # Create temp directory for audio
    tmp_dir = Path("tmp_audio")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Download audio (checks for local file first)
        audio_file = download_audio(url, str(tmp_dir), input_path.stem, str(input_path.parent))
        print()

        # Step 2: Transcribe with Whisper
        segments = transcribe_audio(audio_file, model_to_use)
        print()

        # Step 3: Match lyrics to segments
        print("Matching lyrics to timestamps...")
        matched = match_lyrics_to_segments(lyrics, segments, base_offset=file_offset)
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
