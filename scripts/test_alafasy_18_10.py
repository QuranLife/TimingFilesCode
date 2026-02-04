"""
Test forced alignment with Alafasy 18:10 - known to have repetition
"""

import json
import os
import sys

# Fix Windows console encoding for Arabic
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add FFmpeg to PATH
ffmpeg_path = r"C:\Users\Lapto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

from forced_align_timing import generate_aligned_timing, get_verse_text

script_dir = os.path.dirname(os.path.abspath(__file__))

# Alafasy 18:10
audio_file = os.path.join(script_dir, "alafasy_018010.mp3")
output_file = os.path.join(script_dir, "Alafasy_test_18_10.json")

# Load verse text
quran_path = r"C:\test\PlayGround\QA5\assets\quran_data\quran_uthmani.json"
verse_text = get_verse_text(18, 10, quran_path)

if not verse_text:
    print("Could not load verse text from JSON, using known text for 18:10...")
    # Surah Al-Kahf, verse 10 (16 words)
    verse_text = "إِذْ أَوَى الْفِتْيَةُ إِلَى الْكَهْفِ فَقَالُوا رَبَّنَا آتِنَا مِن لَّدُنكَ رَحْمَةً وَهَيِّئْ لَنَا مِنْ أَمْرِنَا رَشَدًا"

print(f"Verse 18:10 text: {verse_text}")
print(f"Word count: {len(verse_text.split())}")

# Run forced alignment
result, whisper_words, verse_words = generate_aligned_timing(
    audio_file,
    verse_text,
    surah=18,
    ayah=10,
    output_path=output_file
)

print("\n" + "="*60)
print("COMPARISON: Original vs Generated")
print("="*60)

# Load original Alafasy timing
original_path = r"C:\test\PlayGround\QA5\assets\quran_data\Alafasy_128kbps.json"
with open(original_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Find 18:10 in original
original_segments = None
for v in original_data:
    if v.get('surah') == 18 and v.get('ayah') == 10:
        original_segments = v.get('segments', [])
        break

if original_segments:
    print(f"\nOriginal timing: {len(original_segments)} segments")
    print(f"Generated timing: {len(result[0]['segments'])} segments")

    print("\nWord-by-word comparison:")
    print(f"{'Word':<6} {'Original':^25} {'Generated':^25} {'Diff':^10}")
    print("-" * 70)

    for i in range(max(len(original_segments), len(result[0]['segments']))):
        orig = original_segments[i] if i < len(original_segments) else None
        gen = result[0]['segments'][i] if i < len(result[0]['segments']) else None

        orig_str = f"{orig[2]}-{orig[3]}ms" if orig else "N/A"
        gen_str = f"{gen[2]}-{gen[3]}ms" if gen else "N/A"

        # Calculate duration difference
        if orig and gen:
            orig_dur = orig[3] - orig[2]
            gen_dur = gen[3] - gen[2]
            diff = f"{gen_dur - orig_dur:+d}ms"
        else:
            diff = "N/A"

        print(f"{i+1:<6} {orig_str:^25} {gen_str:^25} {diff:^10}")

print("\n" + "="*60)
print("Whisper detected words:")
print("="*60)
for i, w in enumerate(whisper_words):
    print(f"  {i+1}: '{w['word']}' @ {w['start']:.2f}s - {w['end']:.2f}s")

print(f"\nTotal Whisper words: {len(whisper_words)}")
print(f"Expected verse words: {len(verse_words)}")
