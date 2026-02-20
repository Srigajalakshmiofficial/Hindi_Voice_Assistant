import sounddevice as sd
import queue
import json
import numpy as np
import datetime
import re
import subprocess
import requests
import threading
from vosk import Model, KaldiRecognizer
from scipy.signal import resample_poly
from difflib import SequenceMatcher
import time
import os
import glob
import signal
import atexit
import unicodedata

os.environ["ESPEAK_DATA_PATH"] = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"

# ================= STATES =================
STATE_IDLE = 0
STATE_LISTENING = 1
STATE_PROCESSING = 2
STATE_SPEAKING = 3

state = STATE_IDLE
speech_buffer = ""

# Wake words — many phonetic variants so Vosk spelling errors still trigger
WAKE_WORDS = [
    "दीया", "दिया", "दिये", "दिए", "दिओ", "दया",
    "डिया", "जिया", "दिया", "दीयो", "दियो",
]

# ================= FUZZY WAKE WORD =================
def _fuzzy_match(word, text, threshold=0.72):
    """Return True if any token in text is phonetically close to word."""
    for token in text.split():
        ratio = SequenceMatcher(None, word, token).ratio()
        if ratio >= threshold:
            return True
    return False

def _wake_word_in(text):
    """Check exact OR fuzzy match for any wake word in text."""
    # Exact first (fast)
    if any(w in text for w in WAKE_WORDS):
        return True
    # Fuzzy fallback — catches Vosk spelling variants
    return any(_fuzzy_match(w, text) for w in WAKE_WORDS)

def _strip_wake_word(text):
    """Remove wake word token(s) from text, exact and fuzzy."""
    tokens = text.split()
    result = []
    for token in tokens:
        matched = False
        for w in WAKE_WORDS:
            if token == w or SequenceMatcher(None, w, token).ratio() >= 0.72:
                matched = True
                break
        if not matched:
            result.append(token)
    return " ".join(result).strip()

# ================= AUDIO =================
MODEL_PATH = "models/vosk-model-hi-0.22"
DEVICE_RATE = 48000
VOSK_RATE   = 16000

model = Model(MODEL_PATH)
rec   = KaldiRecognizer(model, VOSK_RATE)

q = queue.Queue()
assistant_busy = False

# ================= MUSIC PLAYLIST =================
SONG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
os.makedirs(SONG_DIR, exist_ok=True)

def _load_songs():
    exts = ("*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a", "*.mpeg", "*.mp4")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(SONG_DIR, ext)))
    files.sort()
    return files[:3]

SONGS = _load_songs()

def _song_name(idx):
    if 0 <= idx < len(SONGS):
        return os.path.splitext(os.path.basename(SONGS[idx]))[0]
    return f"गाना {idx+1}"

_music_proc   = None
_playlist_idx = 0
_music_lock   = threading.Lock()


def _stop_music():
    global _music_proc
    with _music_lock:
        if _music_proc and _music_proc.poll() is None:
            _music_proc.terminate()
            try:
                _music_proc.wait(timeout=2)
            except Exception:
                _music_proc.kill()
        _music_proc = None


def _play_index(idx):
    global _music_proc, _playlist_idx
    if not SONGS:
        return False
    idx = idx % len(SONGS)
    _stop_music()
    with _music_lock:
        _playlist_idx = idx
        _music_proc = subprocess.Popen(
            ["cvlc", "--intf", "dummy", "--play-and-exit", SONGS[idx]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return True


def _pause_music():
    global _music_proc
    if _music_proc and _music_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_music_proc.pid), signal.SIGSTOP)
        except Exception:
            pass


def _resume_music():
    global _music_proc
    if _music_proc and _music_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_music_proc.pid), signal.SIGCONT)
        except Exception:
            pass


def music_play():
    if not SONGS:
        return "माफ़ कीजिए, songs फ़ोल्डर में कोई गाना नहीं मिला।"
    ok = _play_index(_playlist_idx)
    return f"{_song_name(_playlist_idx)} बजा रही हूँ।" if ok else "गाना चलाने में समस्या आई।"


def music_next():
    global _playlist_idx
    if not SONGS:
        return "songs फ़ोल्डर में गाना नहीं मिला।"
    _playlist_idx = (_playlist_idx + 1) % len(SONGS)
    _play_index(_playlist_idx)
    return f"अगला गाना: {_song_name(_playlist_idx)}"


def music_prev():
    global _playlist_idx
    if not SONGS:
        return "songs फ़ोल्डर में गाना नहीं मिला।"
    _playlist_idx = (_playlist_idx - 1) % len(SONGS)
    _play_index(_playlist_idx)
    return f"पिछला गाना: {_song_name(_playlist_idx)}"


def music_stop():
    _stop_music()
    return "गाना बंद कर दिया।"


# ================= CLEANUP ON EXIT =================
def _cleanup():
    _stop_music()

atexit.register(_cleanup)

def _signal_handler(sig, frame):
    print("\nExiting Diya...")
    _stop_music()
    os._exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


PIPER_BIN   = "/home/pifive/voice_assistant/piper/build/piper"
PIPER_MODEL = "/home/pifive/voice_assistant/piper/voices/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx"
WAKE_TIMEOUT = 8

# ================= TTS =================
def speak_hi(text):
    global assistant_busy
    assistant_busy = True
    piper = subprocess.Popen(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    paplay = subprocess.Popen(
        ["paplay", "--raw", "--rate=22050", "--format=s16le", "--channels=1"],
        stdin=piper.stdout,
        stderr=subprocess.DEVNULL,
    )
    piper.stdin.write(text.encode("utf-8"))
    piper.stdin.close()
    piper.wait()
    paplay.wait()
    assistant_busy = False


# ================= INTENTS =================
# Keyword groups — lists cover common Vosk spelling variants
_STOP_KW  = ["बंद", "रोको", "रुको", "बंद करो", "stop", "रोक", "बंद कर"]
_NEXT_KW  = ["अगला", "next", "आगे", "अगले", "नेक्स्ट"]
_PREV_KW  = ["पिछला", "पिछले", "previous", "prev", "पीछे", "पिछला"]
_PLAY_KW  = ["गाना", "गीत", "म्यूज़िक", "music", "song", "बजाओ", "सुनाओ",
             "play", "बजा", "चलाओ"]
_TIME_KW  = ["समय", "टाइम", "वक्त", "घड़ी"]
_DATE_KW  = ["तारीख", "दिनांक", "डेट"]
_YEAR_KW  = ["साल", "वर्ष", "ईयर"]
_ALARM_KW = ["अलार्म", "alarm"]
_GREET_KW = ["नमस्ते", "हैलो", "हेलो", "नमस्कार"]
_THANKS_KW = ["धन्यवाद", "शुक्रिया", "थैंक्यू", "thanks", "thank",
               "धन्यवद", "धन्यबाद", "शुक्रिय"]   # common Vosk mis-spellings


def parse_intent(text):
    text = unicodedata.normalize("NFC", text)

    if any(w in text for w in _GREET_KW):
        return "GREET"

    # THANKS — exit command
    if any(w in text for w in _THANKS_KW):
        return "THANKS"

    if any(w in text for w in _TIME_KW) and \
       "अलार्म" not in text and "सेट" not in text:
        return "GET_TIME"

    if "बज" in text and any(w in text for w in ["कितने", "क्या", "अभी"]):
        return "GET_TIME"

    if any(w in text for w in _DATE_KW):
        return "GET_DATE"

    if any(w in text for w in _YEAR_KW):
        return "GET_YEAR"

    if any(w in text for w in _ALARM_KW):
        return "SET_ALARM"

    # Music: STOP before PLAY so "गाना बंद करो" picks STOP not PLAY
    if any(w in text for w in _STOP_KW):
        if any(w in text for w in _PLAY_KW) or _music_proc:
            return "STOP_SONG"

    if any(w in text for w in _NEXT_KW):
        return "NEXT_SONG"

    if any(w in text for w in _PREV_KW):
        return "PREV_SONG"

    if any(w in text for w in _PLAY_KW):
        return "PLAY_SONG"

    return "UNKNOWN"


def execute(intent, text=""):

    if intent == "GREET":
        return "नमस्ते! मैं दीया हूँ, आपकी सहायता के लिए तैयार हूँ।"

    if intent == "THANKS":
        # Speak goodbye, then exit
        speak_hi("आपका बहुत-बहुत धन्यवाद! अलविदा।")
        _stop_music()
        os._exit(0)

    if intent == "GET_TIME":
        now = datetime.datetime.now()
        h, m = now.hour, now.minute
        period = "सुबह" if h < 12 else ("दोपहर" if h < 16 else ("शाम" if h < 20 else "रात"))
        h12 = h % 12 or 12
        return f"अभी {period} के {h12} बजकर {m} मिनट हुए हैं।"

    if intent == "GET_DATE":
        d = datetime.date.today()
        MONTHS = ["","जनवरी","फ़रवरी","मार्च","अप्रैल","मई","जून",
                  "जुलाई","अगस्त","सितंबर","अक्टूबर","नवंबर","दिसंबर"]
        return f"आज {d.day} {MONTHS[d.month]} {d.year} है।"

    if intent == "GET_YEAR":
        return f"वर्तमान साल {datetime.datetime.now().year} है।"

    if intent == "SET_ALARM":
        HINDI_NUMS = {"एक":1,"दो":2,"तीन":3,"चार":4,"पाँच":5,"पांच":5,
                      "छह":6,"छः":6,"सात":7,"आठ":8,"नौ":9,"दस":10,
                      "ग्यारह":11,"बारह":12}
        hour = None
        for word, num in HINDI_NUMS.items():
            if word in text:
                hour = num
                break
        if hour is None:
            m = re.search(r'(\d{1,2})', text)
            if m:
                hour = int(m.group(1))
        if hour:
            if hour < 6:
                hour += 12
            _set_alarm(hour, 0)
            return f"{hour % 12 or 12} बजे का अलार्म लगा दिया।"
        return "कृपया बताइए कितने बजे का अलार्म चाहिए।"

    if intent == "PLAY_SONG":
        return music_play()

    if intent == "NEXT_SONG":
        return music_next()

    if intent == "PREV_SONG":
        return music_prev()

    if intent == "STOP_SONG":
        return music_stop()

    # UNKNOWN — instant offline reply (no Ollama)
    return "माफ़ कीजिए, यह मेरी समझ में नहीं आया।"


# ================= ALARM =================
def _set_alarm(hour, minute):
    def _ring():
        now = datetime.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        time.sleep((target - now).total_seconds())
        speak_hi(f"अलार्म! {hour} बज गए हैं।")
    threading.Thread(target=_ring, daemon=True).start()


# ================= AUDIO CALLBACK =================
def callback(indata, frames, time_, status):
    if assistant_busy:
        return
    q.put(indata.copy())


# ================= HELPERS =================
def _flush_queue():
    """Discard all buffered audio chunks from the queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _listen_for_command():
    """
    After wake word: listen fresh for up to WAKE_TIMEOUT seconds.
    Returns the transcribed command string, or '' if nothing heard.
    """
    rec.Reset()
    _flush_queue()

    deadline = time.time() + WAKE_TIMEOUT
    command_text = ""
    while time.time() < deadline:
        try:
            chunk = q.get(timeout=1.0)
        except queue.Empty:
            continue
        audio_c     = chunk[:, 0]
        audio_16k_c = resample_poly(audio_c, VOSK_RATE, DEVICE_RATE)
        pcm_c       = (audio_16k_c * 32767).astype(np.int16).tobytes()
        if rec.AcceptWaveform(pcm_c):
            res = json.loads(rec.Result())
            command_text = res.get("text", "").strip().replace("।", "").strip()
            if command_text:
                break
        else:
            part = json.loads(rec.PartialResult())
            if part.get("partial", ""):
                deadline = time.time() + WAKE_TIMEOUT   # extend while speaking
    return command_text


# ================= START =================
print("\n===== Diya Offline Assistant =====\n")
speak_hi("नमस्ते, मैं दीया हूँ। आप बोल सकते हैं।")

with sd.InputStream(
        samplerate=DEVICE_RATE,
        channels=1,
        dtype="float32",
        blocksize=1600,
        callback=callback):

    while True:

        data  = q.get()
        audio = data[:, 0]

        audio_16k = resample_poly(audio, VOSK_RATE, DEVICE_RATE)
        pcm16     = (audio_16k * 32767).astype(np.int16).tobytes()

        if rec.AcceptWaveform(pcm16):

            result = json.loads(rec.Result())
            text   = result.get("text", "").strip()

            if not text:
                continue

            text = text.replace("।", "").strip()
            print("\nYou said:", text)
            print("STATE =", state)

            # ================= WAKE WORD =================
            if state == STATE_IDLE:

                if _wake_word_in(text):
                    print("Wake word detected")

                    # Pause music + clear stale audio immediately
                    _pause_music()
                    _flush_queue()
                    rec.Reset()

                    # Strip wake word token from text
                    text = _strip_wake_word(text)

                    if text:
                        # Command came in the same utterance as the wake word
                        # e.g. "दिया गाना बजाओ" → text = "गाना बजाओ"
                        print("Inline command:", text)
                        state = STATE_LISTENING
                        # fall through to PROCESS block
                    else:
                        # Wake word only → ask for command
                        speak_hi("जी, बोलिए")
                        _flush_queue()   # flush audio buffered during TTS

                        command_text = _listen_for_command()

                        if not command_text:
                            print("No command heard — going back to sleep")
                            speak_hi("ठीक है।")
                            _resume_music()
                            state = STATE_IDLE
                            continue

                        print("Command heard:", command_text)
                        text  = command_text
                        state = STATE_LISTENING
                        # fall through to PROCESS block
                else:
                    continue

            # ================= PROCESS =================
            if state == STATE_LISTENING:

                speech_buffer = text.strip()
                print("Processing...")
                state = STATE_PROCESSING

                intent = parse_intent(speech_buffer)
                print("Intent:", intent)

                response = execute(intent, speech_buffer)
                print("Action:", response)

                state = STATE_SPEAKING
                speak_hi(response)

                # Resume music for non-music commands
                MUSIC_INTENTS = {"PLAY_SONG", "NEXT_SONG", "PREV_SONG", "STOP_SONG"}
                if intent not in MUSIC_INTENTS:
                    _resume_music()

                rec.Reset()
                _flush_queue()    # clear audio buffered while Diya was speaking
                state = STATE_IDLE
                print("\nReady for next command")
