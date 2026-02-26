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
        # First: exact name match
        for ext in audio_extensions:
            potential_file = os.path.join(lyrics_dir, f"{song_name}{ext}")
            if os.path.exists(potential_file):
                local_audio = potential_file
                break
        
        # Fallback: fuzzy match normalizing dashes (en-dash – / em-dash — / hyphen -)
        if not local_audio:
            norm_name = re.sub(r'[\u2013\u2014\-]', '-', song_name).strip()
            for f in os.listdir(lyrics_dir):
                f_base, f_ext = os.path.splitext(f)
                if f_ext.lower() not in [e for e in audio_extensions]:
                    continue
                norm_f = re.sub(r'[\u2013\u2014\-]', '-', f_base).strip()
                if norm_f == norm_name:
                    local_audio = os.path.join(lyrics_dir, f)
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


def transcribe_audio(audio_path: str, model_size: str = "medium", language: str = None) -> list[dict]:
    """
    Transcribe audio using faster-whisper.
    Returns list of segments: [{"start": float, "end": float, "text": str}, ...]
    """
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}' (this may take a moment)...")
    # Use float16 if possible, else int8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing audio (Language: {language if language else 'Auto-detect'})...")
    
    # Disable VAD filter by default for singing - it's too aggressive and often finds 0 segments
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=False, # DISABLED: VAD eats singing
    )
    
    # Log detected language if auto-detect was used
    if not language:
        print(f"  Detected language: '{info.language}' (probability: {info.language_probability:.2f})")
        # If auto-detect finds English with low probability, it's a common hallucination for Thai/Music
        if info.language == "en" and info.language_probability < 0.6:
            print("  [Note] Auto-detect might be wrong. If sync fails, add 'lang: th' to your file.")

    segments = []
    full_text_debug = ""
    for seg in segments_gen:
        full_text_debug += seg.text + " "
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
        # print(f"  [{format_time(seg.start)}] {seg.text.strip()}")

    print(f"Found {len(segments)} segments. Raw transcription preview:")
    print(f"  {full_text_debug[:300]}...")
    return segments


def normalize_text(text: str, aggressive: bool = False) -> str:
    """Normalize text for comparison — remove spaces, punctuation, tone marks, etc."""
    text = text.lower().strip()
    
    # Aggressive: Remove all Thai vowels and tone marks to find the "skeleton" of the words
    # This helps significantly with singing where vowels are stretched or tone marks are ignored.
    if aggressive:
        # Thai vowels and tone marks range
        text = re.sub(r'[\u0E30-\u0E4E]', '', text)
        
    # Remove common non-lyric characters, keep Thai chars and alphanumeric
    text = re.sub(r'[^\u0E00-\u0E7F\w]', '', text)
    return text


def similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings using aggressive Thai normalization."""
    # Try normal normalization first
    na = normalize_text(a, aggressive=False)
    nb = normalize_text(b, aggressive=False)
    if not na or not nb: return 0.0
    score = SequenceMatcher(None, na, nb).ratio()
    
    # If score is low, try aggressive (consonant-only) matching
    if score < 0.4:
        na_aggr = normalize_text(a, aggressive=True)
        nb_aggr = normalize_text(b, aggressive=True)
        if na_aggr and nb_aggr:
            aggr_score = SequenceMatcher(None, na_aggr, nb_aggr).ratio()
            score = max(score, aggr_score * 0.75) 
            
    return score


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
    Match user-provided lyrics lines to AI-transcribed words using a global pool.
    Uses interpolation for unmatched lines to prevent "jumping" or "skipping".
    """
    # 1. Flatten all words into a pool
    all_words = []
    for s in segments:
        if s.get("words"):
            all_words.extend(s["words"])
        else:
            all_words.append({"start": s["start"], "end": s["end"], "text": s["text"]})
    
    print(f"DEBUG: Processing {len(all_words)} transcribed words against {len(user_lyrics)} lyrics lines.")
    
    results = []
    word_idx = 0
    last_word_time = 0.0
    OFFSET = base_offset 

    # We'll first pass through and find high-confidence matches
    # Then we'll interpolate the "missing" ones if any.
    
    matches = [] # List of (lyric_idx, timestamp_start, timestamp_end, confidence)
    
    thresholds = [0.4, 0.25, 0.12] # Multi-pass thresholds: Strict, Fuzzy, Skeleton
    
    for l_idx, lyric_line in enumerate(user_lyrics):
        if is_instrumental(lyric_line):
            matches.append((l_idx, -1, -1, 1.0))
            continue

        best_score = 0.0
        best_start = all_words[word_idx]["start"] if word_idx < len(all_words) else 0.0
        best_end = all_words[word_idx]["end"] if word_idx < len(all_words) else 0.0
        best_w_idx = word_idx
        is_reuse = False
        
        # Primary forward scan (if words remain)
        if word_idx < len(all_words):
            look_ahead = min(word_idx + 80, len(all_words))
            for start in range(word_idx, look_ahead):
                combined = ""
                for end in range(start, min(start + 15, len(all_words))):
                    combined += all_words[end]["text"]
                    score = similarity(lyric_line, combined)
                    if score > best_score:
                        best_score = score
                        best_start = all_words[start]["start"]
                        best_end = all_words[end]["end"]
                        best_w_idx = end
                if best_score > 0.9: break

        # Segment-level fallback: match against full segment texts (better for Thai)
        if best_score < 0.4:
            for seg in segments:
                seg_score = similarity(lyric_line, seg["text"])
                if seg_score > best_score:
                    best_score = seg_score
                    best_start = seg["start"]
                    best_end = seg["end"]
                    is_reuse = True
                if best_score > 0.9: break

        # Re-scan entire word pool for chorus repetitions if forward scan failed
        if best_score < 0.4 and word_idx >= len(all_words) * 0.5:
            rescan_best_score = 0.0
            rescan_best_start = 0.0
            rescan_best_end = 0.0
            for start in range(0, len(all_words)):
                combined = ""
                for end in range(start, min(start + 15, len(all_words))):
                    combined += all_words[end]["text"]
                    score = similarity(lyric_line, combined)
                    if score > rescan_best_score:
                        rescan_best_score = score
                        rescan_best_start = all_words[start]["start"]
                        rescan_best_end = all_words[end]["end"]
                if rescan_best_score > 0.9: break
            # Accept reuse with lower threshold for repeated sections
            if rescan_best_score >= 0.25 and rescan_best_score > best_score:
                best_score = rescan_best_score
                best_start = rescan_best_start
                best_end = rescan_best_end
                is_reuse = True

        if best_score >= 0.18:
            if is_reuse:
                conf_str = "Reuse"
            else:
                conf_str = "Strict" if best_score > 0.6 else "Fuzzy" if best_score > 0.3 else "Skeleton"
            print(f"  [{conf_str}] Line {l_idx+1}: '{lyric_line[:15]}...' -> {format_time(best_start)} (conf: {best_score:.2f})")
            if is_reuse:
                # Reuse: lyrics confirmed but timestamp is from a different section.
                # Store None to let interpolation assign the correct chronological position.
                matches.append((l_idx, None, None, best_score))
            else:
                matches.append((l_idx, best_start, best_end, best_score))
                # Only advance word_idx for forward matches
                if word_idx < len(all_words):
                    word_idx = best_w_idx + 1
        else:
            print(f"  [Miss]   Line {l_idx+1}: '{lyric_line[:15]}...' (best conf: {best_score:.2f})")
            matches.append((l_idx, None, None, 0.0))

    # 2. Second pass: Interpolation for missing timestamps
    audio_end = segments[-1]["end"] if segments else all_words[-1]["end"] if all_words else 180.0
    
    # Estimate average line duration from successful matches for better interpolation
    matched_durations = []
    for k in range(len(matches) - 1):
        if matches[k][1] is not None and matches[k][1] != -1 and matches[k+1][1] is not None and matches[k+1][1] != -1:
            dur = matches[k+1][1] - matches[k][1]
            if 1.0 < dur < 15.0:  # Reasonable line duration
                matched_durations.append(dur)
    avg_line_duration = sum(matched_durations) / len(matched_durations) if matched_durations else 3.5
    
    final_results = []
    # Base offset (e.g. -0.2s or -0.3s) helps visuals leading slightly
    # Start from first detected speech, not 0 — avoids placing lyrics in the instrumental intro
    first_speech_time = segments[0]["start"] if segments else 0.0
    last_word_time = first_speech_time
    
    i = 0
    while i < len(matches):
        l_idx, start, end, score = matches[i]
        lyric_text = user_lyrics[l_idx]
        
        # 1. High-confidence match or manual music break "🎶"
        if (start is not None and start != -1) or is_instrumental(lyric_text):
            # MONOTONE: Never move backwards. Minimum 0.8s gap between valid lines.
            final_start = max(last_word_time + 0.8, start if start != -1 else last_word_time + 1.0)
            
            # Auto-insert music bridge for huge gaps (e.g. 12s of silence)
            if start != -1 and start - last_word_time > 12.0:
                final_results.append((last_word_time + 1.5 + OFFSET, "🎶"))
            
            final_results.append((final_start + OFFSET, lyric_text))
            last_word_time = max(final_start, end if end != -1 else final_start + 1.0)
            i += 1
        else:
            # 2. Sequential block of missing matches
            j = i
            while j < len(matches) and matches[j][1] is None and not is_instrumental(user_lyrics[matches[j][0]]):
                j += 1
            
            num_missing = j - i
            next_anchor = audio_end
            if j < len(matches) and matches[j][1] is not None and matches[j][1] != -1:
                next_anchor = matches[j][1]
            
            total_gap = next_anchor - last_word_time
            
            # Smart spacing: use average line duration when gap is large enough,
            # otherwise distribute evenly across the available gap.
            if total_gap >= num_missing * avg_line_duration:
                spacing = avg_line_duration
            else:
                # Distribute evenly — never less than 2s apart for readability
                spacing = max(2.0, total_gap / (num_missing + 1))
            
            for k in range(num_missing):
                est_time = last_word_time + spacing
                # Safety clamp: don't go past audio end, but leave room for remaining lines
                remaining = num_missing - k
                est_time = min(est_time, audio_end - remaining * 1.5)
                # Don't crowd the next anchor
                if est_time >= next_anchor - 0.8 and next_anchor < audio_end:
                    est_time = next_anchor - 0.8
                # Ensure monotonic progress even when clamped
                est_time = max(est_time, last_word_time + 1.0)
                
                final_results.append((est_time + OFFSET, user_lyrics[matches[i+k][0]]))
                last_word_time = est_time
            
            i = j

    # Step 3: Global Sort to be 100% sure we are monotone before Lua generation
    final_results.sort(key=lambda x: x[0])
    return final_results


def parse_lyrics_file(filepath: str) -> tuple[str, list[str], float, str, str]:
    """Parse the lyrics file for URL, lyrics lines, offset, model, and language."""
    url = None
    lyrics = []
    offset = 0.0
    model_size = None
    language = None
    
    # Use utf-8-sig to automatically handle BOM if present
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped: continue
            
            # More flexible tag parsing using regex or better string checks
            lower_line = stripped.lower()
            
            if lower_line.startswith("url"):
                # Handle "url:", "url :", etc.
                match = re.match(r"^url\s*:\s*(.*)$", stripped, re.IGNORECASE)
                if match:
                    url = match.group(1).strip()
                    continue
            elif lower_line.startswith("offset"):
                match = re.match(r"^offset\s*:\s*(.*)$", stripped, re.IGNORECASE)
                if match:
                    try:
                        offset = float(match.group(1).strip())
                    except ValueError:
                        print(f"WARNING: Invalid offset value in file: {stripped}")
                    continue
            elif lower_line.startswith("model"):
                match = re.match(r"^model\s*:\s*(.*)$", stripped, re.IGNORECASE)
                if match:
                    model_size = match.group(1).strip()
                    continue
            elif lower_line.startswith("lang"):
                match = re.match(r"^lang\s*:\s*(.*)$", stripped, re.IGNORECASE)
                if match:
                    language = match.group(1).strip()
                    # Common Thai aliases
                    if language.lower() in ("thai", "th-th"): language = "th"
                    continue
            
            # If it's not a tag, it's a lyrics line
            lyrics.append(stripped)
    
    return url, lyrics, offset, model_size, language


def format_time(seconds: float) -> str:
    """Convert seconds to M:SS format (e.g., 1:35)."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def generate_lua_subtitles(matched: list[tuple[float, str]]) -> str:
    """Generate the Lua Subtitles table. Spreads colliding timestamps by 1s instead of merging."""
    # Step 1: Assign unique timestamps — spread collisions by 1 second
    entries = []  # (timestamp_seconds, lyric_text)
    used_times = set()
    
    for timestamp, lyric in matched:
        t_str = format_time(timestamp)
        
        # If this exact time string is taken, bump by 1 second until free
        original_ts = timestamp
        while t_str in used_times:
            timestamp += 1.0
            t_str = format_time(timestamp)
        
        used_times.add(t_str)
        entries.append((t_str, lyric))

    # Step 2: Build the Lua table
    lines = ['["Subtitles"] =', '{']
    
    for t_str, lyric_text in entries:
        escaped = lyric_text.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'\t["{t_str}"] = "{escaped}",')

    lines.append('},')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Auto Sync Lyrics Tool")
    parser.add_argument("input", help="Path to lyrics .txt file")
    parser.add_argument(
        "--model", default="medium",
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: medium)"
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
    url, lyrics, file_offset, file_model, file_lang = parse_lyrics_file(str(input_path))
    
    # Priority: model from file > model from command line argument
    model_to_use = file_model if file_model else args.model
    
    # NEW: Detect if lyrics are Thai to force language mode (Fixes AI English misdetection)
    if not file_lang:
        thai_char_count = len(re.findall(r'[\u0E00-\u0E7F]', "".join(lyrics)))
        if thai_char_count > 50:
            file_lang = "th"
            print("  [Auto] Thai text detected. Forcing Whisper to 'th' mode for accuracy.")
    
    print(f"Song: {input_path.stem}")
    if url:
        print(f"URL: {url}")
    print(f"Lyrics lines: {len(lyrics)}")
    print(f"Sync Offset: {file_offset}s")
    print(f"Whisper Model: {model_to_use}")
    print(f"Language: {file_lang if file_lang else 'Auto-detect'}")
    print()

    # Create temp directory for audio
    tmp_dir = Path("tmp_audio")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Download audio (checks for local file first)
        audio_file = download_audio(url, str(tmp_dir), input_path.stem, str(input_path.parent))
        print()

        # Step 2: Transcribe with Whisper
        segments = transcribe_audio(audio_file, model_to_use, file_lang)
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
