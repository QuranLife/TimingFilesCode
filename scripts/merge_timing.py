"""Merge timing files and copy to app assets"""
import json

# Load existing 2:282 timing
with open(r"C:\test\PlayGround\QA5\assets\quran_data\Yasser_Ad-Dussary_128kbps.json", 'r') as f:
    existing = json.load(f)

# Load new Surah 67 timing
with open(r"C:\test\whisper_timing_experiment\output\dossary_timing.json", 'r') as f:
    new_timing = json.load(f)

# Merge
all_timing = existing + new_timing

# Sort by surah and ayah
all_timing.sort(key=lambda x: (x["surah"], x["ayah"]))

print(f"Merged: {len(existing)} + {len(new_timing)} = {len(all_timing)} verses")

# Save to app assets
output_path = r"C:\test\PlayGround\QA5\assets\quran_data\Yasser_Ad-Dussary_128kbps.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_timing, f, ensure_ascii=False, indent=2)

print(f"Saved to: {output_path}")
