# Diya — Hindi Offline Voice Assistant

> **दीया** is a fully offline, Hindi-language voice assistant built for ARM-based Linux devices (e.g., Raspberry Pi). It listens for a wake word ("दीया"), understands spoken Hindi commands through on-device speech recognition, responds in natural Hindi speech using neural TTS, and can control music playback, report time/date, and set alarms — all without an internet connection.

---

## Features

| Feature | Details |
|---|---|
| Wake Word Detection | Listens for "दीया" (with fuzzy matching for Vosk spelling variants) |
| Hindi ASR | Offline speech recognition via [Vosk](https://alphacephei.com/vosk/) (`vosk-model-hi-0.22`) |
| Hindi TTS | Neural text-to-speech via [Piper](https://github.com/rhasspy/piper) (Priyamvada voice) |
| Music Playback | Play, pause, next, previous, and stop songs from a local `songs/` folder |
| Alarm | Set alarms by spoken Hindi time (e.g., "सात बजे का अलार्म लगाओ") |
| Time & Date | Queries for current time, date, and year answered in Hindi |
| Greetings | Responds to नमस्ते / हैलो |
| Graceful Exit | Say "धन्यवाद" to shut down the assistant politely |

---

## Project Structure

```
ASR/
├── voice_assistant.py          # Main assistant script
├── models/
│   └── vosk-model-hi-0.22/     # Vosk Hindi ASR model (download separately)
├── songs/                      # Place your local music files here
│   └── *.mp3 / *.wav / ...
└── README.md
```

---

## Requirements

### System Dependencies

| Dependency | Purpose |
|---|---|
| `piper` | Neural Hindi TTS engine |
| `paplay` (PulseAudio) | Audio playback for TTS output |
| `cvlc` (VLC) | Music playback |
| `espeak-ng` | Required by Piper on ARM |

Install system packages (Debian/Ubuntu/Raspberry Pi OS):

```bash
sudo apt install pulseaudio-utils vlc espeak-ng
```

Build or install [Piper](https://github.com/rhasspy/piper) separately, then update the paths in `voice_assistant.py`:

```python
PIPER_BIN   = "/home/pifive/voice_assistant/piper/build/piper"
PIPER_MODEL = "/home/pifive/voice_assistant/piper/voices/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx"
```

### Python Dependencies

```bash
pip install sounddevice vosk numpy scipy requests
```

---

## Setup & Running

### 1. Download the Vosk Hindi Model

```bash
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip
unzip vosk-model-hi-0.22.zip
cd ..
```

### 2. Add Music Files

Place your `.mp3`, `.wav`, `.ogg`, `.flac`, or `.m4a` files in the `songs/` folder. The assistant loads the first 3 songs alphabetically.

```bash
mkdir songs
cp /path/to/your/music/*.mp3 songs/
```

### 3. Run the Assistant

```bash
python voice_assistant.py
```

Diya will announce herself in Hindi and begin listening for her wake word.

---

## Supported Voice Commands (Hindi)

| Intent | Example Phrases |
|---|---|
| Greet | "नमस्ते", "हैलो" |
| Current Time | "समय क्या है?", "अभी कितने बजे हैं?" |
| Current Date | "आज की तारीख क्या है?" |
| Current Year | "इस साल क्या है?", "वर्ष बताओ" |
| Play Music | "गाना बजाओ", "म्यूज़िक चलाओ" |
| Next Song | "अगला गाना", "next" |
| Previous Song | "पिछला गाना", "previous" |
| Stop Music | "गाना बंद करो", "रोको" |
| Set Alarm | "सात बजे का अलार्म लगाओ" |
| Exit / Quit | "धन्यवाद", "थैंक्यू" |

> All commands must be preceded by the wake word **"दीया"**.
> Example: *"दीया, गाना बजाओ"* or *"दीया ... अभी कितने बजे हैं?"*

---

## Configuration

Key constants at the top of `voice_assistant.py`:

| Constant | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/vosk-model-hi-0.22` | Path to Vosk model |
| `DEVICE_RATE` | `48000` | Microphone sample rate (Hz) |
| `VOSK_RATE` | `16000` | Vosk input sample rate (Hz) |
| `WAKE_TIMEOUT` | `8` | Seconds to wait for a command after wake word |
| `PIPER_BIN` | *(see above)* | Path to the Piper binary |
| `PIPER_MODEL` | *(see above)* | Path to the Piper `.onnx` voice model |
| `SONG_DIR` | `./songs` | Directory for local music files |

---

## Architecture Overview

```
Microphone
    │
    ▼
sounddevice (48 kHz)
    │  resample_poly (48k → 16k)
    ▼
Vosk KaldiRecognizer (Hindi)
    │  text transcript
    ▼
Wake Word Detector  ──(no match)──► idle loop
    │  match
    ▼
Intent Parser (keyword rules)
    │  intent label
    ▼
Execute Handler
    │  Hindi response text
    ▼
Piper TTS  ──► paplay (PulseAudio)
```

- **Audio capture** is done via `sounddevice` at the device's native rate and resampled to 16 kHz for Vosk.
- **Music** is managed as background subprocesses (`cvlc`), paused while Diya speaks and resumed after.
- **Wake-word matching** uses both exact string matching and fuzzy Levenshtein ratio (>= 0.72) to handle Vosk transcription variants.

---

## Demo

| Recording | Description |
|---|---|
| `HINDI_VOICE_ASSISTANT.mp4` | Live demo of Diya in action |
| `ARM_Challenge_Recording.mp4` | ARM challenge submission recording |
| `Screen_Rec.mp4` | Screen recording of a session |

See also: [`REPORT_HINDI_VOICE_ASSISTANT_FINAL.pdf`](./REPORT_HINDI_VOICE_ASSISTANT_FINAL.pdf) for the full project report.

---

## Known Limitations

- Music playlist is limited to the first **3 songs** in the `songs/` folder.
- Alarm only supports **hour-level** precision (no minutes via voice).
- Requires a **PulseAudio** server running for TTS output.
- Designed and tested on **AArch64 (ARM64)** Linux; may need path changes on x86 systems.

---

## License

This project is intended for educational and research purposes.
