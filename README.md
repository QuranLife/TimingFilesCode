# Quran Word Timing Files & Generation Scripts

**بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ**

This repository contains word-by-word timing data for Quran recitations and the Python scripts used to generate them. These timing files enable precise word highlighting synchronized with audio playback.

## Timing Files

Located in `/timing_files/`:

| File | Reciter | Audio Quality |
|------|---------|---------------|
| `Mishary_Alafasy_64kbps.json` | Mishary Rashid Alafasy | 64 kbps |
| `Yasser_Ad-Dussary_128kbps.json` | Yasser Ad-Dossari | 128 kbps |
| `MaherAlMuaiqly128kbps.json` | Maher Al-Muaiqly | 128 kbps |
| `Abdullah_Basfar_192kbps.json` | Abdullah Basfar | 192 kbps |

### Timing File Format

Each JSON file contains an array of verses with word-level timing segments:

```json
[
  {
    "surah": 1,
    "ayah": 1,
    "segments": [
      [0, 1, 0, 500],
      [1, 2, 500, 1200],
      ...
    ]
  }
]
```

Segment format: `[word_index, word_number, start_ms, end_ms]`

---

## Setup Instructions

### Step 1: Install Python

Ensure you have Python 3.8 or higher installed:
```bash
python --version
```

### Step 2: Install FFmpeg

FFmpeg is required for audio processing.

**Windows:**
```bash
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

### Step 3: Install Python Dependencies

```bash
pip install openai-whisper
```

For dual-engine timing (optional):
```bash
pip install vosk
```

For hybrid timing with Google (optional):
```bash
pip install google-cloud-speech
```

### Step 4: Download Quran Text Data

Download the Uthmani Quran text JSON and place it in the `scripts/` folder:
- Source: [Quran.com API](https://api.quran.com/api/v4/quran/verses/uthmani)
- Save as: `scripts/quran_uthmani.json`

Or set an environment variable pointing to your existing file:
```bash
export QURAN_JSON_PATH="/path/to/your/quran_uthmani.json"
```

---

## Usage Guide

### Option A: Generate Timing for a Single Verse

Use `generate_timing.py` for quick single-file processing:

```bash
cd scripts

# Download an audio file first (e.g., from everyayah.com)
# Example: Surah 1, Ayah 1 from Alafasy
curl -o 001001.mp3 "https://everyayah.com/data/Alafasy_64kbps/001001.mp3"

# Generate timing
python generate_timing.py 001001.mp3
```

Output: `001001_timing.json`

### Option B: Batch Process Multiple Surahs (Recommended)

Use `batch_generate_timing.py` for processing entire surahs or the full Quran:

```bash
cd scripts

# Process a single surah
python batch_generate_timing.py --reciter alafasy --surah 1

# Process multiple surahs
python batch_generate_timing.py --reciter alafasy --surah 1-10

# Process specific surahs
python batch_generate_timing.py --reciter dossary --surah 36,67,78

# Process entire Quran (takes several hours)
python batch_generate_timing.py --reciter alafasy --all
```

**Available reciters:**
- `alafasy` - Mishary Rashid Alafasy (128kbps)
- `alafasy64` - Mishary Rashid Alafasy (64kbps)
- `dossary` - Yasser Ad-Dossari (128kbps)
- `muaiqly` - Maher Al-Muaiqly (128kbps)
- `basfar` - Abdullah Basfar (64kbps)
- `sudais` - Abdurrahman As-Sudais (192kbps)
- `husary` - Mahmoud Khalil Al-Husary (128kbps)

Output is saved to `scripts/output/` folder.

### Option C: Dual Engine for Better Accuracy

Use `dual_engine_timing.py` for improved accuracy using both Whisper and Vosk:

```bash
cd scripts

# Requires vosk model (auto-downloads on first run, ~180MB)
pip install vosk

python dual_engine_timing.py
```

This cross-validates timing between two speech recognition engines.

### Option D: Forced Alignment with Known Text

Use `forced_align_timing.py` when you have the exact verse text:

```bash
cd scripts
python forced_align_timing.py
```

This aligns Whisper output against the known Quran text for better word matching.

---

## Converting Output Format

If you need to convert raw Whisper output to app format:

```bash
python convert_to_app_format.py
```

Edit the script to specify your input/output paths.

---

## Audio Sources

Audio files are downloaded from [EveryAyah.com](https://everyayah.com). The batch script handles downloads automatically.

URL pattern:
```
https://everyayah.com/data/{reciter_folder}/{surah:03d}{ayah:03d}.mp3
```

Example:
```
https://everyayah.com/data/Alafasy_64kbps/002282.mp3
```

---

## Tips for Best Results

1. **Use the `small` or `medium` Whisper model** for better Arabic recognition
2. **Long verses (like 2:282)** may need manual adjustment
3. **Check for short segments** (<100ms) which may indicate alignment issues
4. **Audio quality matters** - use consistent bitrate files
5. **Repetitions by reciters** can confuse the alignment - forced alignment helps

---

## Project Status

This project is a work in progress. The timing files and scripts require refinement and additional testing. In shaa Allah, once the [Quran Life App](https://play.google.com/apps/testing/com.quranlife.app) reaches a stable state, I plan to return and improve this work further.

If you find these resources beneficial for your Quran application or Islamic project, please remember us in your duas.

---

## License

These timing files and scripts are provided free for use in Quran applications and Islamic educational projects.

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Please credit: **QuranLife App (github.com/QuranLife)**

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition
- [EveryAyah.com](https://everyayah.com) - Quran audio source
- [Quran.com API](https://quran.com) - Uthmani Quran text data
- [quran-align (cpfair)](https://github.com/cpfair/quran-align) - Inspiration and methodology

## Contact

For questions, suggestions, or contributions:
- Email: quranlifeapp@gmail.com

---

رَبَّنَا تَقَبَّلْ مِنَّا إِنَّكَ أَنتَ السَّمِيعُ الْعَلِيمُ

*"Our Lord, accept from us. Indeed, You are the All-Hearing, the All-Knowing."*
