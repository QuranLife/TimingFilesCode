"""
Quran Word Timing Generator using OpenAI Whisper
Generates word-level timestamps for Quran recitation audio files.
"""

import whisper
import json
import sys
import os

# Add FFmpeg to PATH for this session
ffmpeg_path = r"C:\Users\Lapto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

def generate_word_timings(audio_path, output_path=None, model_size="base"):
    """
    Generate word-level timestamps from an audio file using Whisper.

    Args:
        audio_path: Path to the MP3/audio file
        output_path: Path for output JSON (optional, defaults to audio_path.json)
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        dict with word timings
    """

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None

    if output_path is None:
        output_path = audio_path.rsplit('.', 1)[0] + '_timing.json'

    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_path}")
    print("This may take a while for the first run (downloading model)...")

    # Transcribe with word timestamps
    result = model.transcribe(
        audio_path,
        language="ar",  # Arabic
        word_timestamps=True,
        verbose=False  # Disable verbose to avoid Unicode encoding issues on Windows
    )

    # Extract word-level timing data
    timing_data = {
        "audio_file": os.path.basename(audio_path),
        "language": "ar",
        "full_text": result["text"],
        "segments": [],
        "words": []
    }

    # Process segments
    for segment in result["segments"]:
        seg_data = {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": []
        }

        # Process words within segment
        if "words" in segment:
            for word in segment["words"]:
                word_data = {
                    "word": word["word"],
                    "start": round(word["start"], 3),
                    "end": round(word["end"], 3),
                    "probability": round(word.get("probability", 0), 3)
                }
                seg_data["words"].append(word_data)
                timing_data["words"].append(word_data)

        timing_data["segments"].append(seg_data)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timing_data, f, ensure_ascii=False, indent=2)

    print(f"\nTiming data saved to: {output_path}")
    print(f"Total words detected: {len(timing_data['words'])}")

    # Print summary
    print("\n--- Word Timing Summary ---")
    for i, word in enumerate(timing_data['words'][:20]):  # First 20 words
        print(f"{i+1:3}. [{word['start']:6.2f}s - {word['end']:6.2f}s] {word['word']}")

    if len(timing_data['words']) > 20:
        print(f"... and {len(timing_data['words']) - 20} more words")

    return timing_data


if __name__ == "__main__":
    # Default test file
    audio_file = "dossary_002282.mp3"
    model = "base"  # Options: tiny, base, small, medium, large

    # Allow command line arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]

    # Get full path if relative
    if not os.path.isabs(audio_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_file = os.path.join(script_dir, audio_file)

    print("=" * 50)
    print("Quran Word Timing Generator")
    print("=" * 50)
    print(f"Audio: {audio_file}")
    print(f"Model: {model}")
    print("=" * 50)

    result = generate_word_timings(audio_file, model_size=model)

    if result:
        print("\nDone! Check the output JSON file for word timings.")
