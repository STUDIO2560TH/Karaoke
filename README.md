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
- Line 1 (Optional): `url:` followed by the YouTube URL.
- Line 2 (Optional): `offset:` followed by seconds (e.g., `-0.5` to make lyrics appear earlier).
- Line 3 (Optional): `model:` followed by Whisper model (e.g., `large-v3` for higher accuracy).
- `🎶` marks instrumental/music sections.

### 2. Fine-tuning Accuracy
If the timing is slightly off, you can add an `offset:` line to the top of your text file:
- Use a **negative number** (like `offset: -0.5`) if the lyrics appear **too late**.
- Use a **positive number** (like `offset: 0.2`) if the lyrics appear **too early**.

### 2. Bypass YouTube blocks (Optional)

If you see an error about **"Sign in to confirm you're not a bot"** in GitHub Actions, you can upload your own song file:

1. Follow the steps above but **upload BOTH** the lyrics file (e.g., `SongName.txt`) and the song file (e.g., `SongName.mp3`) **at the same time** (in one upload/commit).
2. The files MUST have the **exact same name** (e.g., `โต๊ะริม.txt` and `โต๊ะริม.mp3`).
3. The script will find your file and use it instead of trying to download from YouTube.

### 3. Get results

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
