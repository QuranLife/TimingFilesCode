"""
Convert Whisper timing output to QuranLife app timing format
"""

import json
import os

def convert_whisper_to_app_format(whisper_json_path, surah, ayah, output_path=None):
    """
    Convert Whisper word timing JSON to app's timing format.

    Whisper format:
    {
        "words": [
            {"word": " يا", "start": 0.06, "end": 0.26, "probability": 0.472},
            ...
        ]
    }

    App format:
    [
        {
            "surah": 2,
            "ayah": 282,
            "segments": [[0, 1, 60, 260], ...]  # [index, wordNum, startMs, endMs]
        }
    ]
    """

    with open(whisper_json_path, 'r', encoding='utf-8') as f:
        whisper_data = json.load(f)

    words = whisper_data.get('words', [])

    # Skip the first word if it looks like a verse number (e.g., "6.")
    if words and words[0]['word'].strip().replace('.', '').isdigit():
        words = words[1:]

    segments = []
    for i, word in enumerate(words):
        start_ms = int(word['start'] * 1000)
        end_ms = int(word['end'] * 1000)
        # Format: [index, wordNum (1-based), startMs, endMs]
        segments.append([i, i + 1, start_ms, end_ms])

    app_format = [
        {
            "surah": surah,
            "ayah": ayah,
            "segments": segments
        }
    ]

    if output_path is None:
        output_path = whisper_json_path.replace('_timing.json', '_app_timing.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(app_format, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(segments)} words")
    print(f"Saved to: {output_path}")

    return app_format


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Convert Dossary 2:282
    whisper_file = os.path.join(script_dir, "dossary_002282_timing.json")
    output_file = os.path.join(script_dir, "Yasser_Ad-Dussary_128kbps.json")

    result = convert_whisper_to_app_format(whisper_file, surah=2, ayah=282, output_path=output_file)

    print("\nFirst 5 segments:")
    for seg in result[0]['segments'][:5]:
        print(f"  Word {seg[1]}: {seg[2]}ms - {seg[3]}ms")
