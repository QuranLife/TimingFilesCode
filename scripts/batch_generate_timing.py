"""
Batch Timing Generator for Quran Recitations
Generates word-by-word timing files for any reciter using Whisper + Forced Alignment

Usage:
  python batch_generate_timing.py --reciter alafasy --surah 67
  python batch_generate_timing.py --reciter dossary --surah 1-114
  python batch_generate_timing.py --reciter alafasy --surah 67,68,69
  python batch_generate_timing.py --reciter dossary --all

Output: Creates/updates timing JSON file in output folder
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# Fix Windows console encoding
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# FFmpeg must be installed and in system PATH
# Install via: winget install ffmpeg (Windows) or apt install ffmpeg (Linux)

import whisper

# Reciter configurations (everyayah.com subfolders)
RECITERS = {
    'alafasy': 'Alafasy_128kbps',
    'alafasy64': 'Alafasy_64kbps',
    'dossary': 'Yasser_Ad-Dussary_128kbps',
    'abdulbasit': 'Abdul_Basit_Mujawwad_128kbps',
    'sudais': 'Abdurrahmaan_As-Sudais_192kbps',
    'shuraim': 'Saood_ash-Shuraym_128kbps',
    'husary': 'Husary_128kbps',
    'minshawi': 'Minshawy_Mujawwad_192kbps',
    'hani_rifai': 'Hani_Rifai_192kbps',
    'shaatree': 'Abu_Bakr_Ash-Shaatree_128kbps',
    'muaiqly': 'MaherAlMuaiqly128kbps',
    # User's priority list additions
    'juhaynee': 'Abdullaah_3awwaad_Al-Juhaynee_128kbps',
    'basfar': 'Abdullah_Basfar_64kbps',
    'hudhaify': 'Hudhaify_128kbps',
    'jibreel': 'Muhammad_Jibreel_128kbps',
    'sahl_yassin': 'Sahl_Yassin_128kbps',
}

# Surah verse counts
SURAH_VERSES = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

# Script directory
SCRIPT_DIR = Path(__file__).parent
AUDIO_CACHE_DIR = SCRIPT_DIR / "audio_cache"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Quran text JSON - download from: https://api.quran.com/api/v4/quran/verses/uthmani
# Or use environment variable QURAN_JSON_PATH
QURAN_JSON_PATH = Path(os.environ.get("QURAN_JSON_PATH", SCRIPT_DIR / "quran_uthmani.json"))

# Global Whisper model (loaded once)
_whisper_model = None


def get_whisper_model():
    """Load Whisper model once and reuse"""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (small)... This may take a moment.")
        _whisper_model = whisper.load_model("small")
        print("Model loaded.")
    return _whisper_model


def normalize_arabic(text):
    """Normalize Arabic text for comparison"""
    if not text:
        return ""
    # Remove diacritics
    diacritics = [
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652',
        '\u0653', '\u0654', '\u0655', '\u0656', '\u0657', '\u0658', '\u0659', '\u065A',
        '\u065B', '\u065C', '\u065D', '\u065E', '\u065F', '\u0670'
    ]
    for d in diacritics:
        text = text.replace(d, '')
    text = re.sub('[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('\u0640', '')
    return text.strip()


def similarity_score(word1, word2):
    """Calculate similarity between two Arabic words"""
    w1 = normalize_arabic(word1)
    w2 = normalize_arabic(word2)
    if not w1 or not w2:
        return 0.0
    if w1 == w2:
        return 1.0
    if w1 in w2 or w2 in w1:
        return 0.8
    set1 = set(w1)
    set2 = set(w2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def find_best_match(whisper_word, verse_words, current_pos, lookahead=4):
    """Find best matching verse word for a whisper word"""
    best_match = None
    best_score = 0.0
    threshold = 0.5
    for i in range(current_pos, min(current_pos + lookahead, len(verse_words))):
        score = similarity_score(whisper_word, verse_words[i])
        if score > best_score and score >= threshold:
            best_score = score
            best_match = i
    return best_match, best_score


def get_audio_duration_ms(audio_path):
    """Get audio duration in milliseconds using ffprobe"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10
        )
        duration_sec = float(result.stdout.strip())
        return int(duration_sec * 1000)
    except Exception:
        return None


def calculate_dynamic_lookahead(verse_word_count):
    """Calculate lookahead based on verse length"""
    if verse_word_count <= 10:
        return 4
    elif verse_word_count <= 30:
        return 6
    elif verse_word_count <= 60:
        return 10
    else:
        return 15  # For very long verses like 2:282


def validate_and_fix_timing(segments, audio_duration_ms, verse_word_count):
    """
    Validate timing against audio duration and fix if needed.
    Uses hybrid approach: keep good Whisper timing, fix only broken parts.
    """
    if not segments or not audio_duration_ms:
        return segments

    # STEP 1: Find consecutive runs of short segments (<100ms)
    short_runs = []
    run_start = None
    for i, seg in enumerate(segments):
        duration = seg[3] - seg[2]
        if duration < 100:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                short_runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        short_runs.append((run_start, len(segments) - 1))

    # STEP 2: Fix each run of short segments by redistributing time
    for run_start, run_end in short_runs:
        # Find time boundaries: previous segment's end and next segment's start
        time_start = segments[run_start - 1][3] if run_start > 0 else 0
        time_end = segments[run_end + 1][2] if run_end + 1 < len(segments) else audio_duration_ms

        # If time_end <= time_start, we need to look further ahead
        if time_end <= time_start:
            # Find the next segment with a valid start time
            for j in range(run_end + 1, len(segments)):
                if segments[j][2] > time_start:
                    time_end = segments[j][2]
                    break
            else:
                time_end = audio_duration_ms

        # Redistribute time evenly among the short segments
        run_length = run_end - run_start + 1
        available_time = time_end - time_start
        if available_time > 0 and run_length > 0:
            time_per_word = available_time // run_length
            for k, idx in enumerate(range(run_start, run_end + 1)):
                segments[idx][2] = time_start + k * time_per_word
                segments[idx][3] = time_start + (k + 1) * time_per_word

    # STEP 3: Scale if total exceeds audio duration
    last_end_ms = segments[-1][3] if segments else 0
    if last_end_ms > audio_duration_ms + 500:
        scale = audio_duration_ms / last_end_ms
        for seg in segments:
            seg[2] = int(seg[2] * scale)
            seg[3] = int(seg[3] * scale)

    # STEP 4: Extend if significantly shorter than audio
    elif last_end_ms < audio_duration_ms * 0.9:
        remaining_ms = audio_duration_ms - last_end_ms
        words_to_extend = min(5, len(segments))
        if words_to_extend > 0:
            extra_per_word = remaining_ms // words_to_extend
            for i in range(len(segments) - words_to_extend, len(segments)):
                if i >= 0:
                    offset = (i - (len(segments) - words_to_extend) + 1) * extra_per_word
                    segments[i][3] += offset
                    if i + 1 < len(segments):
                        segments[i + 1][2] = segments[i][3]

    # STEP 5: Final cleanup - fix any remaining issues
    for i, seg in enumerate(segments):
        # Fix zero/negative duration
        if seg[3] <= seg[2]:
            seg[3] = seg[2] + 150
        # Ensure doesn't exceed audio
        seg[3] = min(seg[3], audio_duration_ms)
        # Fix continuity - each segment should start where previous ended
        if i > 0 and seg[2] < segments[i-1][3]:
            seg[2] = segments[i-1][3]
            if seg[3] <= seg[2]:
                seg[3] = seg[2] + 150

    return segments


def load_quran_data():
    """Load Quran text data"""
    if not QURAN_JSON_PATH.exists():
        print(f"ERROR: Quran JSON not found at {QURAN_JSON_PATH}")
        return None
    with open(QURAN_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_verse_text(quran_data, surah, ayah):
    """Get verse text from loaded Quran data"""
    if not quran_data:
        return ""
    if "data" in quran_data and "surahs" in quran_data["data"]:
        for surah_data in quran_data["data"]["surahs"]:
            if surah_data.get("number") == surah:
                for ayah_data in surah_data.get("ayahs", []):
                    if ayah_data.get("numberInSurah") == ayah:
                        return ayah_data.get("text", "")
    return ""


def download_audio(reciter_id, surah, ayah):
    """Download audio file from everyayah.com"""
    if reciter_id not in RECITERS:
        raise ValueError(f"Unknown reciter: {reciter_id}")

    subfolder = RECITERS[reciter_id]
    surah_str = str(surah).zfill(3)
    ayah_str = str(ayah).zfill(3)
    filename = f"{surah_str}{ayah_str}.mp3"

    # Create cache directory for this reciter
    cache_dir = AUDIO_CACHE_DIR / reciter_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / filename

    if local_path.exists():
        return str(local_path)

    url = f"https://everyayah.com/data/{subfolder}/{filename}"

    try:
        urllib.request.urlretrieve(url, local_path)
        return str(local_path)
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return None


def generate_timing_for_verse(audio_path, verse_text, surah, ayah):
    """Generate timing data for a single verse with improved accuracy"""
    model = get_whisper_model()

    # Get audio duration for validation
    audio_duration_ms = get_audio_duration_ms(audio_path)

    result = model.transcribe(
        audio_path,
        language="ar",
        word_timestamps=True,
        verbose=False
    )

    # Extract whisper words
    whisper_words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            whisper_words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    # Parse verse text
    verse_words = [w.strip() for w in verse_text.split() if w.strip()]

    if not whisper_words or not verse_words:
        return None

    # Calculate dynamic lookahead based on verse length
    lookahead = calculate_dynamic_lookahead(len(verse_words))

    # Forced alignment
    segments = []
    verse_pos = 0
    i = 0

    while i < len(whisper_words):
        w = whisper_words[i]
        match_idx, score = find_best_match(w["word"], verse_words, verse_pos, lookahead=lookahead)

        # Check for repetition
        if match_idx is None and verse_pos > 0:
            earlier_match, earlier_score = find_best_match(w["word"], verse_words, 0, lookahead=verse_pos)
            if earlier_match is not None and earlier_score >= 0.6:
                i += 1
                continue

        if match_idx is not None and match_idx >= verse_pos:
            # Fill skipped words with proportional timing
            if verse_pos < match_idx:
                gap_start = segments[-1][3] if segments else 0
                gap_end = int(w["start"] * 1000)
                gap_duration = gap_end - gap_start
                words_to_fill = match_idx - verse_pos

                # Distribute gap time evenly among skipped words
                if words_to_fill > 0 and gap_duration > 0:
                    time_per_word = gap_duration // words_to_fill
                    for k in range(words_to_fill):
                        word_start = gap_start + k * time_per_word
                        word_end = gap_start + (k + 1) * time_per_word
                        segments.append([verse_pos + k, verse_pos + k + 1, word_start, word_end])
                    verse_pos = match_idx
                else:
                    # Fallback: minimal duration for skipped words
                    while verse_pos < match_idx:
                        prev_end = segments[-1][3] if segments else 0
                        segments.append([verse_pos, verse_pos + 1, prev_end, prev_end + 50])
                        verse_pos += 1

            start_ms = int(w["start"] * 1000)
            end_ms = int(w["end"] * 1000)

            # Look ahead for extended duration (with dynamic lookahead)
            j = i + 1
            while j < len(whisper_words):
                next_w = whisper_words[j]
                if verse_pos + 1 < len(verse_words):
                    next_match, _ = find_best_match(next_w["word"], verse_words, verse_pos + 1, lookahead=min(3, lookahead))
                    if next_match is not None:
                        break
                rep_match, rep_score = find_best_match(next_w["word"], verse_words, 0, lookahead=verse_pos + 1)
                if rep_match is not None and rep_match <= verse_pos:
                    end_ms = int(next_w["end"] * 1000)
                    j += 1
                else:
                    break

            segments.append([verse_pos, verse_pos + 1, start_ms, end_ms])
            verse_pos += 1
            i = j
        else:
            if segments:
                segments[-1][3] = int(w["end"] * 1000)
            i += 1

    # Handle remaining words with proportional timing
    if verse_pos < len(verse_words):
        remaining_words = len(verse_words) - verse_pos
        last_end = segments[-1][3] if segments else 0

        # Use audio duration if available, otherwise estimate
        if audio_duration_ms and audio_duration_ms > last_end:
            remaining_time = audio_duration_ms - last_end
            time_per_word = remaining_time // remaining_words
        else:
            time_per_word = 400  # Default ~400ms per word

        for k in range(remaining_words):
            word_start = last_end + k * time_per_word
            word_end = last_end + (k + 1) * time_per_word
            segments.append([verse_pos + k, verse_pos + k + 1, word_start, word_end])

    # Sort and renumber
    segments.sort(key=lambda x: x[0])
    for idx, seg in enumerate(segments):
        seg[0] = idx
        seg[1] = idx + 1

    # Validate and fix timing against audio duration
    if audio_duration_ms:
        segments = validate_and_fix_timing(segments, audio_duration_ms, len(verse_words))

    return {
        "surah": surah,
        "ayah": ayah,
        "segments": segments
    }


def process_surah(reciter_id, surah_num, quran_data, existing_data=None):
    """Process all verses in a surah"""
    if surah_num not in SURAH_VERSES:
        print(f"Invalid surah number: {surah_num}")
        return []

    verse_count = SURAH_VERSES[surah_num]
    results = existing_data or []

    # Remove existing entries for this surah
    results = [v for v in results if v.get("surah") != surah_num]

    print(f"\nProcessing Surah {surah_num} ({verse_count} verses)...")

    for ayah in range(1, verse_count + 1):
        verse_text = get_verse_text(quran_data, surah_num, ayah)
        if not verse_text:
            print(f"  {surah_num}:{ayah} - No verse text found, skipping")
            continue

        # Download audio
        audio_path = download_audio(reciter_id, surah_num, ayah)
        if not audio_path:
            print(f"  {surah_num}:{ayah} - Audio download failed, skipping")
            continue

        # Generate timing
        try:
            timing = generate_timing_for_verse(audio_path, verse_text, surah_num, ayah)
            if timing:
                results.append(timing)
                word_count = len(timing["segments"])
                print(f"  {surah_num}:{ayah} - {word_count} words aligned")
            else:
                print(f"  {surah_num}:{ayah} - Failed to generate timing")
        except Exception as e:
            print(f"  {surah_num}:{ayah} - Error: {e}")

    return results


def parse_surah_arg(surah_arg):
    """Parse surah argument (e.g., '67', '1-10', '67,68,69')"""
    surahs = []
    for part in surah_arg.split(','):
        if '-' in part:
            start, end = part.split('-')
            surahs.extend(range(int(start), int(end) + 1))
        else:
            surahs.append(int(part))
    return sorted(set(surahs))


def main():
    parser = argparse.ArgumentParser(description='Generate Quran word timing files')
    parser.add_argument('--reciter', required=True, help='Reciter ID (e.g., alafasy, dossary)')
    parser.add_argument('--surah', help='Surah number(s): 67 or 1-10 or 67,68,69')
    parser.add_argument('--all', action='store_true', help='Process all 114 surahs')
    parser.add_argument('--output', help='Output JSON filename (default: {reciter}_timing.json)')

    args = parser.parse_args()

    if args.reciter not in RECITERS:
        print(f"Unknown reciter: {args.reciter}")
        print(f"Available: {', '.join(RECITERS.keys())}")
        return

    # Determine which surahs to process
    if args.all:
        surahs = list(range(1, 115))
    elif args.surah:
        surahs = parse_surah_arg(args.surah)
    else:
        print("Please specify --surah or --all")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_file = args.output or f"{args.reciter}_timing.json"
    output_path = OUTPUT_DIR / output_file

    # Load existing data if any
    existing_data = []
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"Loaded {len(existing_data)} existing verse timings from {output_file}")

    # Load Quran text
    print("Loading Quran text data...")
    quran_data = load_quran_data()
    if not quran_data:
        return

    # Process surahs
    all_results = existing_data
    start_time = time.time()

    for surah_num in surahs:
        all_results = process_surah(args.reciter, surah_num, quran_data, all_results)

        # Save after each surah (in case of interruption)
        all_results.sort(key=lambda x: (x["surah"], x["ayah"]))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"\nCompleted! {len(all_results)} verses processed in {elapsed:.1f}s")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
