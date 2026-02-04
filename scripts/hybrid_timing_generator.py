"""
Hybrid Timing Generator - Uses both Whisper and Google Speech-to-Text
for improved Arabic word timing accuracy.

Google provides better Arabic recognition, Whisper provides backup.
Cross-validates and merges for best results.
"""

import json
import os
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# FFmpeg must be installed and in system PATH
# Install via: winget install ffmpeg (Windows) or apt install ffmpeg (Linux)


def get_google_word_timestamps(audio_path):
    """
    Get word-level timestamps using Google Cloud Speech-to-Text API.
    Returns list of {"word": str, "start": float, "end": float}
    """
    try:
        from google.cloud import speech_v1p1beta1 as speech
    except ImportError:
        print("  Google Cloud Speech library not installed. Run: pip install google-cloud-speech")
        return None

    client = speech.SpeechClient()

    # Read audio file
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    # Configure recognition
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,  # Common for MP3
        language_code="ar-SA",  # Arabic (Saudi Arabia)
        enable_word_time_offsets=True,  # Get word timestamps
        enable_automatic_punctuation=False,
        model="default",
    )

    try:
        response = client.recognize(config=config, audio=audio)
    except Exception as e:
        print(f"  Google API error: {e}")
        return None

    # Extract word timestamps
    words = []
    for result in response.results:
        alternative = result.alternatives[0]
        for word_info in alternative.words:
            words.append({
                "word": word_info.word,
                "start": word_info.start_time.total_seconds(),
                "end": word_info.end_time.total_seconds()
            })

    return words


def get_whisper_word_timestamps(audio_path):
    """
    Get word-level timestamps using Whisper.
    Returns list of {"word": str, "start": float, "end": float}
    """
    import whisper

    # Load model (cached globally)
    global _whisper_model
    if '_whisper_model' not in globals() or _whisper_model is None:
        print("  Loading Whisper model...")
        _whisper_model = whisper.load_model("small")

    result = _whisper_model.transcribe(
        audio_path,
        language="ar",
        word_timestamps=True,
        verbose=False
    )

    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    return words


def merge_timestamps(whisper_words, google_words, verse_word_count, audio_duration_ms):
    """
    Merge Whisper and Google timestamps intelligently.

    Strategy:
    1. If Google has good coverage (>70% of words), prefer Google
    2. Use Whisper to fill gaps where Google missed words
    3. Cross-validate: if both have a word, use the one closer to expected position
    """
    if not google_words and not whisper_words:
        return None

    # If only one source available, use it
    if not google_words:
        print("  Using Whisper only (Google unavailable)")
        return whisper_words
    if not whisper_words:
        print("  Using Google only (Whisper unavailable)")
        return google_words

    print(f"  Whisper detected {len(whisper_words)} words")
    print(f"  Google detected {len(google_words)} words")
    print(f"  Expected: {verse_word_count} words")

    # Calculate coverage
    google_coverage = len(google_words) / verse_word_count if verse_word_count > 0 else 0
    whisper_coverage = len(whisper_words) / verse_word_count if verse_word_count > 0 else 0

    print(f"  Google coverage: {google_coverage:.1%}, Whisper coverage: {whisper_coverage:.1%}")

    # If Google has good coverage (>60%), prefer it as primary
    if google_coverage >= 0.6:
        primary = google_words
        secondary = whisper_words
        print("  Primary: Google, Secondary: Whisper")
    else:
        primary = whisper_words
        secondary = google_words
        print("  Primary: Whisper, Secondary: Google")

    # Build merged timeline
    # For now, return primary but validate segment durations
    merged = []
    for w in primary:
        duration = w["end"] - w["start"]
        # If segment is too short (<50ms), try to find corresponding word in secondary
        if duration < 0.05 and secondary:
            # Find word in secondary with similar start time (within 500ms)
            for sw in secondary:
                if abs(sw["start"] - w["start"]) < 0.5:
                    if sw["end"] - sw["start"] > duration:
                        w = sw  # Use secondary's timing
                        break
        merged.append(w)

    return merged


def generate_hybrid_timing(audio_path, verse_text, surah, ayah, use_google=True):
    """
    Generate timing using hybrid Whisper + Google approach.
    """
    import subprocess

    # Get audio duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10
        )
        audio_duration_ms = int(float(result.stdout.strip()) * 1000)
    except:
        audio_duration_ms = None

    verse_words = [w.strip() for w in verse_text.split() if w.strip()]
    verse_word_count = len(verse_words)

    print(f"\n  Audio duration: {audio_duration_ms}ms")
    print(f"  Verse words: {verse_word_count}")

    # Get timestamps from both sources
    whisper_words = get_whisper_word_timestamps(audio_path)

    google_words = None
    if use_google:
        google_words = get_google_word_timestamps(audio_path)

    # Merge timestamps
    merged_words = merge_timestamps(whisper_words, google_words, verse_word_count, audio_duration_ms)

    if not merged_words:
        return None

    # Convert to segments format and align to verse words
    segments = align_to_verse(merged_words, verse_words, audio_duration_ms)

    return {
        "surah": surah,
        "ayah": ayah,
        "segments": segments
    }


def normalize_arabic(text):
    """Normalize Arabic text for comparison"""
    import re
    if not text:
        return ""
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


def align_to_verse(detected_words, verse_words, audio_duration_ms):
    """
    Align detected words to verse words using forced alignment.
    Returns segments in format [[idx, idx+1, start_ms, end_ms], ...]
    """
    segments = []
    verse_pos = 0
    detected_pos = 0

    lookahead = 15 if len(verse_words) > 60 else 10 if len(verse_words) > 30 else 6

    while detected_pos < len(detected_words) and verse_pos < len(verse_words):
        dw = detected_words[detected_pos]

        # Find best match in verse words
        best_match = None
        best_score = 0.0
        for i in range(verse_pos, min(verse_pos + lookahead, len(verse_words))):
            score = similarity_score(dw["word"], verse_words[i])
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = i

        if best_match is not None:
            # Fill skipped words with proportional timing
            if verse_pos < best_match:
                gap_start = segments[-1][3] if segments else 0
                gap_end = int(dw["start"] * 1000)
                words_to_fill = best_match - verse_pos
                if gap_end > gap_start and words_to_fill > 0:
                    time_per_word = (gap_end - gap_start) // words_to_fill
                    for k in range(words_to_fill):
                        start = gap_start + k * time_per_word
                        end = gap_start + (k + 1) * time_per_word
                        segments.append([verse_pos + k, verse_pos + k + 1, start, end])
                verse_pos = best_match

            # Add matched word
            start_ms = int(dw["start"] * 1000)
            end_ms = int(dw["end"] * 1000)
            segments.append([verse_pos, verse_pos + 1, start_ms, end_ms])
            verse_pos += 1

        detected_pos += 1

    # Fill remaining verse words
    if verse_pos < len(verse_words):
        last_end = segments[-1][3] if segments else 0
        remaining = len(verse_words) - verse_pos
        if audio_duration_ms and audio_duration_ms > last_end:
            time_per_word = (audio_duration_ms - last_end) // remaining
        else:
            time_per_word = 400
        for k in range(remaining):
            start = last_end + k * time_per_word
            end = last_end + (k + 1) * time_per_word
            segments.append([verse_pos + k, verse_pos + k + 1, start, end])

    # Renumber segments
    for i, seg in enumerate(segments):
        seg[0] = i
        seg[1] = i + 1

    return segments


# Test function
def test_hybrid():
    """Test hybrid timing on verse 2:282"""
    sys.path.insert(0, str(Path(__file__).parent))
    from batch_generate_timing import load_quran_data, get_verse_text, download_audio

    print("=" * 60)
    print("HYBRID TIMING TEST - Whisper + Google")
    print("=" * 60)

    quran_data = load_quran_data()

    # Test on 2:282
    surah, ayah = 2, 282
    verse_text = get_verse_text(quran_data, surah, ayah)
    audio_path = download_audio('alafasy', surah, ayah)

    print(f"\nTesting verse {surah}:{ayah}")

    # Try with Google
    timing = generate_hybrid_timing(audio_path, verse_text, surah, ayah, use_google=True)

    if timing:
        segs = timing['segments']
        print(f"\nGenerated {len(segs)} segments")

        # Compare with existing timing file (if available)
        v1_path = Path(__file__).parent.parent / "timing_files" / "Mishary_Alafasy_64kbps.json"
        if v1_path.exists():
            with open(v1_path, 'r') as f:
                v1_data = json.load(f)

            v1_segs = None
            for v in v1_data:
                if v['surah'] == surah and v['ayah'] == ayah:
                    v1_segs = v['segments']
                    break

            if v1_segs:
                print(f"\nComparison with existing ({len(v1_segs)} segments):")
                print(f"{'Point':<8} | {'New':>10} | {'V1':>10} | {'Diff':>8}")
                print("-" * 45)
                for pct in [0, 0.25, 0.5, 0.75, 1.0]:
                    new_idx = min(int(len(segs) * pct), len(segs) - 1)
                    v1_idx = min(int(len(v1_segs) * pct), len(v1_segs) - 1)
                    new_start = segs[new_idx][2]
                    v1_start = v1_segs[v1_idx][2]
                    diff = new_start - v1_start
                    print(f"{int(pct*100):>3}%     | {new_start:>10} | {v1_start:>10} | {diff:>+8}")


if __name__ == "__main__":
    test_hybrid()
