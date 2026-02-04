"""
Forced Alignment Timing Generator for Quran Recitations
Uses Whisper for word detection + matching against known Quran text
Handles reciter repetitions by extending word durations
"""

import json
import os
import re
import sys
import unicodedata

# Fix Windows console encoding for Arabic
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Add FFmpeg to PATH
ffmpeg_path = r"C:\Users\Lapto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

import whisper


def normalize_arabic(text):
    """Normalize Arabic text for comparison - remove diacritics, normalize forms"""
    if not text:
        return ""

    # Remove common diacritics (tashkeel)
    diacritics = [
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652',
        '\u0653', '\u0654', '\u0655', '\u0656', '\u0657', '\u0658', '\u0659', '\u065A',
        '\u065B', '\u065C', '\u065D', '\u065E', '\u065F', '\u0670'
    ]
    for d in diacritics:
        text = text.replace(d, '')

    # Normalize alef variations
    text = re.sub('[إأآا]', 'ا', text)

    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')

    # Normalize yeh variations
    text = text.replace('ى', 'ي')

    # Remove tatweel (kashida)
    text = text.replace('\u0640', '')

    # Strip and lowercase (for any latin mixed in)
    text = text.strip().lower()

    return text


def similarity_score(word1, word2):
    """Calculate similarity between two Arabic words (0-1)"""
    w1 = normalize_arabic(word1)
    w2 = normalize_arabic(word2)

    if not w1 or not w2:
        return 0.0

    if w1 == w2:
        return 1.0

    # Check if one contains the other (partial match)
    if w1 in w2 or w2 in w1:
        return 0.8

    # Simple character overlap ratio
    set1 = set(w1)
    set2 = set(w2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def find_best_match(whisper_word, verse_words, current_pos, lookahead=3):
    """
    Find if whisper_word matches any of the next few expected verse words.
    Returns (matched_index, score) or (None, 0) if no match.
    """
    best_match = None
    best_score = 0.0
    threshold = 0.5  # Minimum similarity to consider a match

    for i in range(current_pos, min(current_pos + lookahead, len(verse_words))):
        score = similarity_score(whisper_word, verse_words[i])
        if score > best_score and score >= threshold:
            best_score = score
            best_match = i

    return best_match, best_score


def generate_aligned_timing(audio_path, verse_text, surah, ayah, output_path=None):
    """
    Generate timing data using Whisper + forced alignment against verse text.

    Args:
        audio_path: Path to MP3/audio file
        verse_text: The known Arabic text of the verse (space-separated words)
        surah: Surah number
        ayah: Ayah number
        output_path: Optional output JSON path

    Returns:
        App-format timing data
    """
    print(f"Loading Whisper model (small for better accuracy)...")
    model = whisper.load_model("small")  # Use small model for better Arabic recognition

    print(f"Transcribing: {audio_path}")
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

    # Get audio duration for coverage analysis
    audio_duration = result.get("duration", 0)
    if whisper_words:
        last_word_end = whisper_words[-1]["end"]
    else:
        last_word_end = 0

    print(f"Whisper detected {len(whisper_words)} words")
    print(f"Audio duration: {audio_duration:.1f}s, Words detected up to: {last_word_end:.1f}s")

    if audio_duration > 0 and last_word_end < audio_duration * 0.8:
        print(f"  WARNING: Whisper only covered {last_word_end/audio_duration*100:.0f}% of audio!")

    # Parse verse text into words
    verse_words = [w.strip() for w in verse_text.split() if w.strip()]
    print(f"Verse has {len(verse_words)} words")

    # IMPROVED FORCED ALIGNMENT ALGORITHM
    # Strategy: Track verse position, look for matches in whisper output
    # When we find matches to earlier words (repetition), extend current word's duration
    # When we find match to next expected word, move forward

    segments = []
    verse_pos = 0  # Current position in verse we're looking to match
    last_matched_verse_pos = -1  # Track highest verse position we've matched

    i = 0
    while i < len(whisper_words):
        w = whisper_words[i]

        # Try to match against expected verse word (and next 2-3)
        match_idx, score = find_best_match(w["word"], verse_words, verse_pos, lookahead=4)

        # Also check if this matches an EARLIER word (repetition detection)
        if match_idx is None and verse_pos > 0:
            earlier_match, earlier_score = find_best_match(w["word"], verse_words, 0, lookahead=verse_pos)
            if earlier_match is not None and earlier_score >= 0.6:
                # This is a repetition! Extend the current word's duration
                if segments:
                    # Don't create new segment, just note we're in repetition
                    print(f"  Repetition detected: '{w['word']}' matches earlier word {earlier_match + 1}")
                i += 1
                continue

        if match_idx is not None and match_idx >= verse_pos:
            # Found a match forward!
            # Fill in any skipped words first
            while verse_pos < match_idx:
                # Estimate timing for skipped word
                if segments:
                    prev_end = segments[-1][3]
                else:
                    prev_end = 0
                # Give skipped words a small duration
                segments.append([
                    verse_pos,
                    verse_pos + 1,
                    prev_end,
                    int(w["start"] * 1000)  # End at start of matched word
                ])
                verse_pos += 1

            # Now add the matched word
            start_ms = int(w["start"] * 1000)
            end_ms = int(w["end"] * 1000)

            # Look ahead to see if next whisper words are still this verse word (extended)
            j = i + 1
            while j < len(whisper_words):
                next_w = whisper_words[j]
                # Check if next whisper word matches NEXT verse word
                if verse_pos + 1 < len(verse_words):
                    next_match, next_score = find_best_match(
                        next_w["word"], verse_words, verse_pos + 1, lookahead=2
                    )
                    if next_match is not None:
                        # Next whisper word matches next verse word, stop extending
                        break
                # Check if it's a repetition of earlier word
                rep_match, rep_score = find_best_match(next_w["word"], verse_words, 0, lookahead=verse_pos + 1)
                if rep_match is not None and rep_match <= verse_pos:
                    # Still in repetition zone, extend current word
                    end_ms = int(next_w["end"] * 1000)
                    j += 1
                else:
                    break

            segments.append([
                verse_pos,
                verse_pos + 1,
                start_ms,
                end_ms
            ])

            last_matched_verse_pos = verse_pos
            verse_pos += 1
            i = j  # Skip past any extended words
        else:
            # No match - could be noise or unrecognized repetition
            # If we have segments, extend the last one
            if segments:
                segments[-1][3] = int(w["end"] * 1000)
            i += 1

    # Handle remaining verse words if any
    while verse_pos < len(verse_words):
        if segments:
            prev_end = segments[-1][3]
        else:
            prev_end = 0
        # Estimate ~500ms per remaining word
        segments.append([
            verse_pos,
            verse_pos + 1,
            prev_end,
            prev_end + 500
        ])
        verse_pos += 1

    # Sort and renumber
    segments.sort(key=lambda x: x[0])
    for i, seg in enumerate(segments):
        seg[0] = i
        seg[1] = i + 1

    print(f"Generated {len(segments)} aligned segments")

    # Create app format
    app_format = [{
        "surah": surah,
        "ayah": ayah,
        "segments": segments
    }]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(app_format, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {output_path}")

    return app_format, whisper_words, verse_words


def get_verse_text(surah, ayah, quran_json_path):
    """Load verse text from quran JSON file"""
    with open(quran_json_path, 'r', encoding='utf-8') as f:
        quran_data = json.load(f)

    # Handle different JSON structures
    if isinstance(quran_data, dict):
        # Check for nested structure: {"data": {"surahs": [...]}}
        if "data" in quran_data and "surahs" in quran_data["data"]:
            surahs = quran_data["data"]["surahs"]
            for surah_data in surahs:
                if surah_data.get("number") == surah:
                    for ayah_data in surah_data.get("ayahs", []):
                        if ayah_data.get("numberInSurah") == ayah:
                            return ayah_data.get("text", "")
            return ""

        # Format: {"1": {"1": "text", "2": "text"}, ...}
        return quran_data.get(str(surah), {}).get(str(ayah), "")
    elif isinstance(quran_data, list):
        # Format: [{"surah": 1, "ayah": 1, "text": "..."}, ...]
        for verse in quran_data:
            if verse.get("surah") == surah and verse.get("ayah") == ayah:
                return verse.get("text", "")

    return ""


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Test with Dossary 2:282
    audio_file = os.path.join(script_dir, "dossary_002282.mp3")
    output_file = os.path.join(script_dir, "Yasser_Ad-Dussary_128kbps_aligned.json")

    # Load verse text from quran data
    quran_path = r"C:\test\PlayGround\QA5\assets\quran_data\quran_uthmani.json"

    # For 2:282, we need the verse text
    verse_text = get_verse_text(2, 282, quran_path)

    if not verse_text:
        print("Could not load verse text. Using fallback...")
        # You can paste the verse text here as fallback
        verse_text = ""

    if verse_text:
        print(f"Verse text loaded: {len(verse_text.split())} words")
        result, whisper_words, verse_words = generate_aligned_timing(
            audio_file,
            verse_text,
            surah=2,
            ayah=282,
            output_path=output_file
        )

        print("\nFirst 10 segments:")
        for seg in result[0]['segments'][:10]:
            print(f"  Word {seg[1]}: {seg[2]}ms - {seg[3]}ms")
    else:
        print("No verse text available!")
