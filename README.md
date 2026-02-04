# Quran Word Timing Files & Generation Scripts

**بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ**

This repository contains word-by-word timing data for Quran recitations and the Python scripts used to generate them. These timing files enable precise word highlighting synchronized with audio playback.

## Timing Files

Located in `/timing_files/`:

| File | Reciter | Audio Quality |
|------|---------|---------------|
| `Mishary_Alafasy_64kbps.json` | Mishary Rashid Alafasy | 64 kbps |
| `Yasser_Ad-Dussary_128kbps.json` | Yasser Ad-Dossari | 128 kbps |
| `MaherAlMuaiqly128kbps.json` | Maher Al-Muaiqly | 128 kbps |
| `Abdullah_Basfar_192kbps.json` | Abdullah Basfar | 192 kbps |

### Timing File Format

Each JSON file contains an array of verses with word-level timing segments:

```json
{
  "surah": 1,
  "ayah": 1,
  "segments": [
    [start_ms, end_ms, "word_text"],
    [start_ms, end_ms, "word_text"],
    ...
  ]
}
```

## Generation Scripts

Located in `/scripts/`:

| Script | Description |
|--------|-------------|
| `generate_timing.py` | Basic timing generator using Whisper |
| `batch_generate_timing.py` | Batch processing for full Quran |
| `dual_engine_timing.py` | Dual engine approach for better accuracy |
| `hybrid_timing_generator.py` | Hybrid timing generator combining methods |
| `forced_align_timing.py` | Forced alignment timing |
| `convert_to_app_format.py` | Convert output to app-compatible format |
| `merge_timing.py` | Merge timing files |
| `inject_282_timing.py` | Inject timing for specific verses |

### Requirements

- Python 3.8+
- OpenAI Whisper (`pip install openai-whisper`)
- FFmpeg (for audio processing)

### Usage

1. Download audio files from [EveryAyah.com](https://everyayah.com)
2. Run the timing generator:
   ```bash
   python scripts/batch_generate_timing.py --reciter <reciter_folder> --output <output.json>
   ```
3. Fine-tune results if needed (some manual adjustment may be required for optimal accuracy)

## Audio Sources

The timing files are designed to work with audio from [EveryAyah.com](https://everyayah.com). Ensure you use the matching audio quality (kbps) for accurate synchronization.

## Project Status

This project is a work in progress. The timing files and scripts require refinement and additional testing. In shaa Allah, once the [Quran Life App](https://github.com/QuranLife) reaches a stable state, I plan to return and improve this work further.

If you find these resources beneficial for your Quran application or Islamic project, please remember us in your duas.

## License

These timing files and scripts are provided free for use in Quran applications and Islamic educational projects.

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Please credit: **QuranLife App (github.com/QuranLife)**

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [EveryAyah.com](https://everyayah.com) - Quran audio source
- [quran-align (cpfair)](https://github.com/cpfair/quran-align) - Inspiration and methodology

## Contact

For questions, suggestions, or contributions:
- Email: quranlifeapp@gmail.com

---

رَبَّنَا تَقَبَّلْ مِنَّا إِنَّكَ أَنتَ السَّمِيعُ الْعَلِيمُ

*"Our Lord, accept from us. Indeed, You are the All-Hearing, the All-Knowing."*
