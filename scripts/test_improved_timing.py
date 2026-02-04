"""
Test improved timing generation on specific verses
"""
import json
import os
import sys

# Fix Windows console encoding
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Add FFmpeg to PATH
ffmpeg_path = r"C:\Users\Lapto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

from batch_generate_timing import (
    download_audio, generate_timing_for_verse, load_quran_data, get_verse_text
)

def test_verse(surah, ayah, quran_data):
    """Test timing generation for a single verse"""
    print(f"\n{'='*60}")
    print(f"Testing verse {surah}:{ayah}")
    print('='*60)

    verse_text = get_verse_text(quran_data, surah, ayah)
    if not verse_text:
        print(f"ERROR: No verse text found")
        return None

    word_count = len(verse_text.split())
    print(f"Text word count: {word_count}")

    # Download audio
    audio_path = download_audio('alafasy', surah, ayah)
    if not audio_path:
        print(f"ERROR: Audio download failed")
        return None

    print(f"Audio: {audio_path}")

    # Generate timing
    timing = generate_timing_for_verse(audio_path, verse_text, surah, ayah)
    if not timing:
        print(f"ERROR: Timing generation failed")
        return None

    segments = timing['segments']
    print(f"Generated segments: {len(segments)}")

    # Show first 10 and last 5 segments
    print(f"\nFirst 10 segments:")
    for i, seg in enumerate(segments[:10]):
        duration = seg[3] - seg[2]
        print(f"  [{i}] {seg[2]:6}ms - {seg[3]:6}ms (dur: {duration}ms)")

    if len(segments) > 15:
        print(f"\nLast 5 segments:")
        for i, seg in enumerate(segments[-5:], start=len(segments)-5):
            duration = seg[3] - seg[2]
            print(f"  [{i}] {seg[2]:6}ms - {seg[3]:6}ms (dur: {duration}ms)")

    # Check for issues
    zero_duration = sum(1 for s in segments if s[3] - s[2] <= 0)
    total_duration = segments[-1][3] if segments else 0

    print(f"\nTotal duration: {total_duration}ms ({total_duration/1000:.1f}s)")
    print(f"Zero-duration segments: {zero_duration}")

    return timing


def compare_with_existing(new_timing, existing_path, surah, ayah):
    """Compare new timing with existing file"""
    try:
        with open(existing_path, 'r') as f:
            existing_data = json.load(f)

        existing = None
        for v in existing_data:
            if v['surah'] == surah and v['ayah'] == ayah:
                existing = v
                break

        if not existing:
            print(f"\nNo existing timing found in {existing_path}")
            return

        new_segs = new_timing['segments']
        old_segs = existing['segments']

        print(f"\n{'='*60}")
        print(f"COMPARISON with {os.path.basename(existing_path)}")
        print('='*60)
        print(f"New segments: {len(new_segs)}")
        print(f"Old segments: {len(old_segs)}")

        # Compare at key points
        points = [0.0, 0.25, 0.5, 0.75, 1.0]
        print(f"\nTiming at key points:")
        print(f"{'Point':<8} | {'New Start':>10} | {'Old Start':>10} | {'Diff':>8}")
        print('-' * 45)

        for pct in points:
            new_idx = min(int(len(new_segs) * pct), len(new_segs) - 1)
            old_idx = min(int(len(old_segs) * pct), len(old_segs) - 1)

            new_start = new_segs[new_idx][2]
            old_start = old_segs[old_idx][2]
            diff = new_start - old_start

            print(f"{int(pct*100):>3}%     | {new_start:>10} | {old_start:>10} | {diff:>+8}")

    except Exception as e:
        print(f"Error comparing: {e}")


def main():
    print("Loading Quran data...")
    quran_data = load_quran_data()
    if not quran_data:
        return

    # Test verses
    test_verses = [(4, 154), (2, 282)]

    v1_path = r"C:\test\PlayGround\QA5\assets\quran_data\Alafasy_128kbps.json"
    v2_path = r"C:\test\PlayGround\QA5\assets\quran_data\Alafasy_64kbps.json"

    for surah, ayah in test_verses:
        timing = test_verse(surah, ayah, quran_data)
        if timing:
            print("\n--- Comparison with V1 (third-party) ---")
            compare_with_existing(timing, v1_path, surah, ayah)
            print("\n--- Comparison with V2 (our current) ---")
            compare_with_existing(timing, v2_path, surah, ayah)


if __name__ == "__main__":
    main()
