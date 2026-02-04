"""
Inject the new 2:282 timing from dual engine into Alafasy_128kbps.json
so user can compare V1 vs V2 by playing the audio.
"""

import json
import sys
import os

# Fix Windows console encoding
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_TIMING_FILE = r"C:\test\PlayGround\QA5\assets\quran_data\Alafasy_64kbps.json"  # V2 only!

# FFmpeg path
ffmpeg_path = r"C:\Users\Lapto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

# Import from dual_engine_timing
sys.path.insert(0, SCRIPT_DIR)
from dual_engine_timing import generate_dual_timing, get_audio_duration_ms
from batch_generate_timing import load_quran_data, get_verse_text, download_audio


def main():
    print("=" * 60)
    print("INJECTING NEW 2:282 TIMING INTO ALAFASY FILE")
    print("=" * 60)

    # Load Quran data
    quran_data = load_quran_data()

    # Generate timing for 2:282
    surah, ayah = 2, 282
    verse_text = get_verse_text(quran_data, surah, ayah)
    audio_path = download_audio('alafasy', surah, ayah)

    print(f"\nGenerating timing for {surah}:{ayah}...")
    timing = generate_dual_timing(audio_path, verse_text, surah, ayah)

    if not timing:
        print("ERROR: Failed to generate timing")
        return

    print(f"\nNew timing has {len(timing['segments'])} segments")

    # Load app timing file
    print(f"\nLoading {APP_TIMING_FILE}...")
    with open(APP_TIMING_FILE, 'r', encoding='utf-8') as f:
        app_data = json.load(f)

    # Find and replace 2:282
    found = False
    for i, verse in enumerate(app_data):
        if verse['surah'] == surah and verse['ayah'] == ayah:
            # Save old timing for comparison
            old_segments = verse['segments']
            print(f"\nOld timing: {len(old_segments)} segments")
            print(f"  First: {old_segments[0]}")
            print(f"  Last:  {old_segments[-1]}")

            # Replace with new timing
            verse['segments'] = timing['segments']

            print(f"\nNew timing: {len(timing['segments'])} segments")
            print(f"  First: {timing['segments'][0]}")
            print(f"  Last:  {timing['segments'][-1]}")

            found = True
            break

    if not found:
        print(f"ERROR: Could not find {surah}:{ayah} in app timing file")
        return

    # Save updated file
    print(f"\nSaving updated timing file...")
    with open(APP_TIMING_FILE, 'w', encoding='utf-8') as f:
        json.dump(app_data, f, separators=(',', ':'))

    print("\nDone! You can now compare the timing in the app.")
    print("Play 2:282 with word-by-word highlighting to see V2 timing.")


if __name__ == "__main__":
    main()
