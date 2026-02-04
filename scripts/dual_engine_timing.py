"""
Dual Engine Timing Generator
Uses Whisper (medium) + Vosk for cross-validation
Both are FREE and run locally - no subscriptions needed.

Whisper medium has better Arabic than small.
Vosk provides a second opinion for validation.
"""

import json
import os
import sys
import wave
import subprocess
import urllib.request
import zipfile
from pathlib import Path

# Fix Windows console encoding
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
VOSK_MODEL_DIR = MODELS_DIR / "vosk-model-ar-0.22-linto-1.1.0"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-ar-0.22-linto-1.1.0.zip"

# FFmpeg path
ffmpeg_path = r"C:\Users\Lapto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

# Global models
_whisper_model = None
_vosk_model = None


def download_vosk_model():
    """Download Arabic Vosk model if not present"""
    if VOSK_MODEL_DIR.exists():
        return True

    print("Downloading Arabic Vosk model (180MB)... This is a one-time download.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = MODELS_DIR / "vosk-model-ar.zip"

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            print(f"\r  Downloading: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path, report_progress)
        print("\n  Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)

        zip_path.unlink()  # Delete zip file
        print("  Vosk model ready!")
        return True

    except Exception as e:
        print(f"\n  Error downloading Vosk model: {e}")
        return False


def get_whisper_model():
    """Load Whisper medium model (better Arabic than small)"""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("Loading Whisper medium model (first time takes a minute)...")
        _whisper_model = whisper.load_model("medium")
        print("Whisper model loaded.")
    return _whisper_model


def get_vosk_model():
    """Load Vosk Arabic model"""
    global _vosk_model
    if _vosk_model is None:
        if not download_vosk_model():
            return None
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)  # Suppress Vosk logs
        _vosk_model = Model(str(VOSK_MODEL_DIR))
        print("Vosk model loaded.")
    return _vosk_model


def convert_to_wav(mp3_path):
    """Convert MP3 to WAV for Vosk (16kHz mono)"""
    wav_path = mp3_path.replace('.mp3', '_temp.wav')
    ffmpeg_exe = os.path.join(ffmpeg_path, "ffmpeg.exe")

    subprocess.run([
        ffmpeg_exe, "-y", "-i", mp3_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
    ], capture_output=True)

    return wav_path


def get_audio_duration_ms(audio_path):
    """Get audio duration in milliseconds"""
    ffprobe_path = os.path.join(ffmpeg_path, "ffprobe.exe")
    try:
        result = subprocess.run(
            [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10
        )
        return int(float(result.stdout.strip()) * 1000)
    except:
        return None


def get_whisper_timestamps(audio_path):
    """Get word timestamps from Whisper medium model"""
    model = get_whisper_model()

    result = model.transcribe(
        audio_path,
        language="ar",
        word_timestamps=True,
        verbose=False,
        condition_on_previous_text=False,  # Helps with long audio
    )

    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"],
                "source": "whisper"
            })

    # Debug: show time range
    if words:
        first_time = words[0]["start"]
        last_time = words[-1]["end"]
        print(f"  Whisper time range: {first_time:.1f}s - {last_time:.1f}s")

    return words


def get_vosk_timestamps(audio_path):
    """Get word timestamps from Vosk"""
    model = get_vosk_model()
    if model is None:
        return []

    from vosk import KaldiRecognizer

    # Convert to WAV
    wav_path = convert_to_wav(audio_path)

    words = []
    try:
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        # Get final result
        final_result = json.loads(rec.FinalResult())

        if "result" in final_result:
            for word_info in final_result["result"]:
                words.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "source": "vosk"
                })

        # Debug: show time range
        if words:
            first_time = words[0]["start"]
            last_time = words[-1]["end"]
            print(f"  Vosk time range: {first_time:.1f}s - {last_time:.1f}s")

        wf.close()
    except Exception as e:
        print(f"  Vosk error: {e}")
    finally:
        # Clean up temp WAV
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    return words


def normalize_arabic(text):
    """Normalize Arabic for comparison"""
    import re
    if not text:
        return ""
    diacritics = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670'
    for d in diacritics:
        text = text.replace(d, '')
    text = re.sub('[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = text.replace('\u0640', '')
    return text.strip()


def similarity(w1, w2):
    """Word similarity score"""
    n1, n2 = normalize_arabic(w1), normalize_arabic(w2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    if n1 in n2 or n2 in n1:
        return 0.8
    s1, s2 = set(n1), set(n2)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def merge_dual_timestamps(whisper_words, vosk_words, verse_words, audio_duration_ms):
    """
    Merge Whisper and Vosk timestamps intelligently.

    Strategy:
    - Use Whisper as primary (usually better word detection)
    - When Whisper has short/suspicious segments, check Vosk
    - Cross-validate timing at anchor points
    """
    print(f"  Whisper: {len(whisper_words)} words, Vosk: {len(vosk_words)} words")

    if not whisper_words:
        return vosk_words if vosk_words else []

    # Build Vosk lookup by time ranges
    def find_vosk_at_time(time_sec, tolerance=0.5):
        for vw in vosk_words:
            if abs(vw["start"] - time_sec) < tolerance:
                return vw
        return None

    # Merge: use Whisper but validate/fix with Vosk
    merged = []
    for i, ww in enumerate(whisper_words):
        duration = ww["end"] - ww["start"]

        # If Whisper segment is suspiciously short (<100ms), check Vosk
        if duration < 0.1:
            vw = find_vosk_at_time(ww["start"])
            if vw and (vw["end"] - vw["start"]) > duration:
                # Vosk has better timing for this word
                merged.append({
                    "word": ww["word"],  # Keep Whisper's word
                    "start": vw["start"],
                    "end": vw["end"],
                    "source": "vosk_fix"
                })
                continue

        merged.append(ww)

    # Check for drift: compare anchor points
    if vosk_words and len(merged) > 10:
        # Check at 25%, 50%, 75%
        for pct in [0.25, 0.5, 0.75]:
            widx = int(len(merged) * pct)
            if widx < len(merged):
                wtime = merged[widx]["start"]
                # Find closest Vosk word
                closest_vosk = min(vosk_words, key=lambda v: abs(v["start"] - wtime))
                drift = wtime - closest_vosk["start"]

                # If drift > 2 seconds, Vosk might be more accurate
                if abs(drift) > 2.0:
                    print(f"  Drift detected at {int(pct*100)}%: {drift:.1f}s")

    return merged


def align_to_verse(detected_words, verse_words, audio_duration_ms):
    """
    Align detected words to verse text using anchor-based interpolation.

    Strategy for long verses:
    1. Find strong anchor matches (high similarity)
    2. Interpolate between anchors for unmatched words
    3. Ensure ALL verse words get segments spanning full audio
    """
    num_verse_words = len(verse_words)
    num_detected = len(detected_words)

    if not detected_words or not verse_words:
        # Fallback: distribute evenly
        return [[i, i+1, i * audio_duration_ms // num_verse_words,
                 (i+1) * audio_duration_ms // num_verse_words]
                for i in range(num_verse_words)]

    # Step 1: Find anchor matches using dynamic programming approach
    # anchors[verse_idx] = (detected_idx, start_ms, end_ms, score)
    anchors = {}

    # Use larger lookahead for long verses
    if num_verse_words > 100:
        lookahead = 25
        min_score = 0.35
    elif num_verse_words > 60:
        lookahead = 20
        min_score = 0.38
    else:
        lookahead = 15
        min_score = 0.4

    detected_idx = 0
    verse_idx = 0

    # Rate limiting: don't let verse_idx advance too fast relative to detected_idx
    # Expected ratio: verse_idx / num_verse_words ≈ detected_idx / num_detected
    max_verse_skip = max(3, num_verse_words // num_detected + 2) if num_detected > 0 else 5

    while detected_idx < num_detected and verse_idx < num_verse_words:
        dw = detected_words[detected_idx]

        # Calculate expected verse position based on progress through detected words
        expected_verse_pos = int(detected_idx * num_verse_words / num_detected)

        # If verse_idx is way ahead of expected, slow down matching
        max_allowed_verse = min(expected_verse_pos + lookahead, num_verse_words)

        # Search for best match in lookahead window, but don't jump too far ahead
        best_match_verse = None
        best_score = 0.0

        # Limit search to prevent runaway
        search_start = verse_idx
        search_end = min(verse_idx + max_verse_skip, max_allowed_verse, num_verse_words)

        for vi in range(search_start, search_end):
            score = similarity(dw["word"], verse_words[vi])
            if score > best_score and score >= min_score:
                best_score = score
                best_match_verse = vi

        if best_match_verse is not None:
            # Found anchor
            anchors[best_match_verse] = (
                detected_idx,
                int(dw["start"] * 1000),
                int(dw["end"] * 1000),
                best_score
            )
            verse_idx = best_match_verse + 1
        else:
            # No match with strict lookahead - use proportional position
            # This ensures we don't skip too much
            prop_verse_idx = expected_verse_pos
            if prop_verse_idx >= verse_idx and prop_verse_idx < num_verse_words:
                if prop_verse_idx not in anchors:
                    anchors[prop_verse_idx] = (
                        detected_idx,
                        int(dw["start"] * 1000),
                        int(dw["end"] * 1000),
                        0.2  # Low confidence marker
                    )
                # Advance verse_idx to stay synchronized
                verse_idx = max(verse_idx, prop_verse_idx + 1)

        detected_idx += 1

    # Step 2: Ensure first and last anchors are at correct times
    # First anchor: at time 0
    if 0 not in anchors and detected_words:
        anchors[0] = (0, int(detected_words[0]["start"] * 1000),
                      int(detected_words[0]["end"] * 1000), 0.5)

    # Last anchor: MUST be near the end of audio, regardless of what was matched
    last_verse_idx = num_verse_words - 1
    last_dw = detected_words[-1] if detected_words else None

    if last_dw:
        last_detected_time = int(last_dw["end"] * 1000)

        # If existing last anchor is too early (more than 20% before audio end),
        # force it to use the actual audio duration
        if last_verse_idx in anchors:
            existing_time = anchors[last_verse_idx][1]
            if existing_time < audio_duration_ms * 0.8:
                # Existing anchor is way too early - override it
                anchors[last_verse_idx] = (
                    num_detected - 1,
                    max(last_detected_time - 500, audio_duration_ms - 1500),
                    audio_duration_ms,
                    0.5
                )
        else:
            anchors[last_verse_idx] = (
                num_detected - 1,
                max(last_detected_time - 500, audio_duration_ms - 1500),
                audio_duration_ms,
                0.5
            )

    # Step 3: Build segments by interpolating between anchors
    segments = []
    sorted_anchors = sorted(anchors.keys())

    if not sorted_anchors:
        # No anchors at all - distribute evenly
        per_word = audio_duration_ms // num_verse_words
        return [[i, i+1, i * per_word, (i+1) * per_word] for i in range(num_verse_words)]

    # Handle words before first anchor
    first_anchor_idx = sorted_anchors[0]
    if first_anchor_idx > 0:
        first_anchor_start = anchors[first_anchor_idx][1]
        per_word = first_anchor_start // first_anchor_idx if first_anchor_idx > 0 else 300
        per_word = max(per_word, 150)  # Minimum 150ms per word
        for i in range(first_anchor_idx):
            segments.append([i, i+1, i * per_word, (i+1) * per_word])

    # Process anchors and gaps between them
    for ai, anchor_verse_idx in enumerate(sorted_anchors):
        anchor_data = anchors[anchor_verse_idx]
        anchor_start = anchor_data[1]
        anchor_end = anchor_data[2]

        # Add this anchor's segment
        segments.append([anchor_verse_idx, anchor_verse_idx + 1, anchor_start, anchor_end])

        # Interpolate to next anchor
        if ai + 1 < len(sorted_anchors):
            next_anchor_idx = sorted_anchors[ai + 1]
            next_anchor_start = anchors[next_anchor_idx][1]

            gap_words = next_anchor_idx - anchor_verse_idx - 1
            if gap_words > 0:
                gap_start = anchor_end
                gap_end = next_anchor_start
                if gap_end > gap_start:
                    per_word = (gap_end - gap_start) // gap_words
                    per_word = max(per_word, 100)  # Minimum 100ms
                    for k in range(gap_words):
                        word_idx = anchor_verse_idx + 1 + k
                        segments.append([
                            word_idx, word_idx + 1,
                            gap_start + k * per_word,
                            gap_start + (k + 1) * per_word
                        ])

    # Handle words after last anchor
    last_anchor_idx = sorted_anchors[-1]
    if last_anchor_idx < num_verse_words - 1:
        last_anchor_end = anchors[last_anchor_idx][2]
        remaining = num_verse_words - last_anchor_idx - 1
        remaining_time = audio_duration_ms - last_anchor_end
        per_word = remaining_time // remaining if remaining > 0 else 300
        per_word = max(per_word, 150)
        for k in range(remaining):
            word_idx = last_anchor_idx + 1 + k
            segments.append([
                word_idx, word_idx + 1,
                last_anchor_end + k * per_word,
                last_anchor_end + (k + 1) * per_word
            ])

    # Step 4: Sort and renumber segments
    segments.sort(key=lambda s: s[0])

    # Ensure we have exactly num_verse_words segments
    # Fill any gaps
    existing_indices = {s[0] for s in segments}
    for i in range(num_verse_words):
        if i not in existing_indices:
            # Interpolate from neighbors
            prev_end = segments[i-1][3] if i > 0 and len(segments) > i-1 else 0
            next_start = audio_duration_ms
            for s in segments:
                if s[0] > i:
                    next_start = s[2]
                    break
            segments.append([i, i+1, prev_end, min(prev_end + 400, next_start)])

    segments.sort(key=lambda s: s[0])

    # Renumber and fix overlaps
    for i, seg in enumerate(segments):
        seg[0], seg[1] = i, i + 1

        # Fix overlap with previous
        if i > 0 and seg[2] < segments[i-1][3]:
            seg[2] = segments[i-1][3]

        # Ensure minimum duration
        if seg[3] <= seg[2]:
            seg[3] = seg[2] + 150
        elif seg[3] - seg[2] < 80:
            # Borrow from previous if possible
            if i > 0 and segments[i-1][3] - segments[i-1][2] > 250:
                borrow = 100
                segments[i-1][3] -= borrow
                seg[2] = segments[i-1][3]

    # Final pass: ensure end time doesn't exceed audio duration
    for seg in segments:
        if seg[3] > audio_duration_ms:
            seg[3] = audio_duration_ms
        if seg[2] >= seg[3]:
            seg[2] = max(0, seg[3] - 100)

    # Debug: show anchor distribution
    if len(verse_words) > 50:
        anchor_times = [(idx, anchors[idx][1]) for idx in sorted(anchors.keys())]
        print(f"  Anchors found: {len(anchors)} at verse positions/times:")
        for idx, time_ms in anchor_times[:5]:
            print(f"    [{idx}] @ {time_ms}ms")
        if len(anchor_times) > 10:
            print("    ...")
        for idx, time_ms in anchor_times[-3:]:
            print(f"    [{idx}] @ {time_ms}ms")
    print(f"  Final: {len(anchors)} anchors, {len(segments)} segments")

    return segments[:num_verse_words]  # Ensure exactly right number


def generate_dual_timing(audio_path, verse_text, surah, ayah):
    """Generate timing using both Whisper medium and Vosk"""
    verse_words = verse_text.split()
    audio_duration_ms = get_audio_duration_ms(audio_path)

    print(f"\n  Processing {surah}:{ayah} ({len(verse_words)} words, {audio_duration_ms}ms)")

    # Get timestamps from both engines
    print("  Running Whisper medium...")
    whisper_words = get_whisper_timestamps(audio_path)

    print("  Running Vosk...")
    vosk_words = get_vosk_timestamps(audio_path)

    # Merge results
    merged = merge_dual_timestamps(whisper_words, vosk_words, verse_words, audio_duration_ms)

    # Align to verse
    segments = align_to_verse(merged, verse_words, audio_duration_ms)

    return {
        "surah": surah,
        "ayah": ayah,
        "segments": segments
    }


def test_dual_engine():
    """Test on verse 2:282"""
    sys.path.insert(0, str(SCRIPT_DIR))
    from batch_generate_timing import load_quran_data, get_verse_text, download_audio

    print("=" * 60)
    print("DUAL ENGINE TIMING TEST")
    print("Whisper Medium + Vosk (both FREE, local)")
    print("=" * 60)

    quran_data = load_quran_data()

    for surah, ayah in [(2, 282), (4, 154)]:
        verse_text = get_verse_text(quran_data, surah, ayah)
        audio_path = download_audio('alafasy', surah, ayah)

        timing = generate_dual_timing(audio_path, verse_text, surah, ayah)

        if timing:
            segs = timing['segments']
            print(f"\n  Generated {len(segs)} segments")

            # Short segment check
            short = sum(1 for s in segs if s[3] - s[2] < 100)
            print(f"  Short segments (<100ms): {short}")

            # Compare with V1
            with open(r"C:\test\PlayGround\QA5\assets\quran_data\Alafasy_128kbps.json", 'r') as f:
                v1_data = json.load(f)

            v1_segs = next((v['segments'] for v in v1_data
                          if v['surah'] == surah and v['ayah'] == ayah), None)

            if v1_segs:
                print(f"\n  Comparison with V1:")
                print(f"  {'Point':<6} | {'New':>8} | {'V1':>8} | {'Diff':>8}")
                print("  " + "-" * 40)
                for pct in [0, 0.25, 0.5, 0.75, 1.0]:
                    nidx = min(int(len(segs) * pct), len(segs) - 1)
                    vidx = min(int(len(v1_segs) * pct), len(v1_segs) - 1)
                    nstart = segs[nidx][2]
                    vstart = v1_segs[vidx][2]
                    diff = nstart - vstart
                    print(f"  {int(pct*100):>3}%   | {nstart:>8} | {vstart:>8} | {diff:>+8}")


if __name__ == "__main__":
    test_dual_engine()
