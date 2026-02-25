# Karaoke

Auto sync Thai song lyrics with timestamps for the Roblox karaoke module.

## How to Use

### 1. Upload lyrics to GitHub

The easiest way is to use the GitHub website directly:

1. Open your repository on **GitHub.com**.
2. Go into the **`lyrics`** folder.
3. Click **Add file** -> **Upload files**.
4. Drag and drop your `.txt` file into the box.
5. Click **Commit changes**.

**Format for the `.txt` file:**
```
url: https://www.youtube.com/watch?v=VIDEO_ID

🎶
First lyrics line
Second lyrics line
🎶
Third lyrics line
...
```
- First line: `url:` followed by the YouTube URL.
- `🎶` marks instrumental/music sections.

### 2. Get results

The GitHub Action will:
1. Download the audio from YouTube
2. Use AI (Whisper) to detect when each lyric is sung
3. Generate a `.lua` file in `output/` with synced timestamps
4. Auto-commit the result and delete the source `.txt` file

### Output format

```lua
["Subtitles"] =
{
    ["0:00"] = "🎶",
    ["0:12"] = "แค่ผ่านมาเพียงสบตา (โฮ้)",
    ["0:18"] = "เหมือนว่าใจลอยหลุดไป (โฮ้)",
    ...
},
```

## Local Usage

```bash
pip install -r requirements.txt
python sync_lyrics.py lyrics/YourSong.txt --model small --output output
```

Available models: `tiny`, `base`, `small` (default), `medium`, `large`
