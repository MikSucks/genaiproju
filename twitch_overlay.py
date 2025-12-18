import os
import time
import queue
import threading
import logging
import numpy as np
import sounddevice as sd
import json
import wave
from flask import Flask, Response, render_template_string
import whisper

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
BLOCKSIZE   = 2048
MAX_LINES   = 3
FADE_TIME   = 4

TEMP_DIR = "temp"
TEMP_FILE = os.path.join(TEMP_DIR, "audio.wav")

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("A1 Whisper LOCAL overlay starting (no OpenAI API, no soundfile, no Twitch bot)...")

# -----------------------------
# LOAD WHISPER
# -----------------------------
logging.info("Ladataan Whisper-malli (base)...")
model = whisper.load_model("base")
logging.info("Whisper-malli ladattu.")

# -----------------------------
# AUDIO BUFFER
# -----------------------------
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        logging.warning(status)
    audio_queue.put(indata.copy())

# -----------------------------
# FLASK PAGE
# -----------------------------
app = Flask(__name__)
last_lines = []
lock = threading.Lock()

HTML = """
<html><head><meta charset="UTF-8">
<style>
body { background: transparent; margin:0; overflow:hidden; }
.line {
  font-size: 40px; color:white; text-shadow:2px 2px 6px black;
  font-family: Arial; opacity:1; transition:opacity 1s linear;
}
#box { position:absolute; bottom:60px; left:60px; }
</style></head>
<body>
<div id="box">
{% for t,ts in lines %}
  <div class="line" id="l{{loop.index}}" data-ts="{{ts}}">{{t}}</div>
{% endfor %}
</div>
<script>
const evt = new EventSource("/stream");
evt.onmessage = e=>{
   const data = JSON.parse(e.data);
   data.forEach((line,i)=>{
     const el = document.getElementById("l"+(i+1));
     if (el){
        el.textContent = line.text;
        el.dataset.ts = line.ts;
        el.style.opacity = 1;
     }
   });
};
setInterval(()=>{
   const now = Date.now()/1000;
   document.querySelectorAll(".line").forEach(el=>{
      const ts = parseFloat(el.dataset.ts||0);
      if (now-ts > {{fade}}){
         el.style.opacity = 0;
      }
   });
},200);
</script>
</body></html>
"""

@app.route("/")
def index():
    with lock:
        return render_template_string(HTML, lines=last_lines, fade=FADE_TIME)

@app.route("/stream")
def stream():
    def event_stream():
        prev=None
        while True:
            with lock:
                cur=list(last_lines)
            if cur!=prev:
                yield "data: "+json.dumps(
                    [{"text":t,"ts":ts} for (t,ts) in cur]
                )+"\n\n"
                prev=cur
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

# -----------------------------
# MICROPHONE → WHISPER LOOP
# -----------------------------
def process_audio():
    logging.info("Mic → Whisper loop running...")
    buffer = np.zeros(0, dtype=np.float32)

    while True:
        block = audio_queue.get().reshape(-1)
        buffer = np.concatenate((buffer, block))

        if len(buffer) >= SAMPLE_RATE * 1.5:  # 1.5 s clip
            pcm = (buffer * 32767).astype(np.int16)

            try:
                with wave.open(TEMP_FILE, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm.tobytes())

                result = model.transcribe(TEMP_FILE, language="fi", task="translate")

                text = result["text"].strip()

                if text:
                    logging.info("Detected: " + text)
                    ts = time.time()
                    with lock:
                        last_lines.append((text, ts))
                        if len(last_lines)>MAX_LINES:
                            last_lines[:] = last_lines[-MAX_LINES:]

            except Exception as e:
                logging.error("Whisper error: %s", e)

            buffer = np.zeros(0, dtype=np.float32)

# -----------------------------
# START SYSTEM
# -----------------------------
def start_mic():
    logging.info("Starting microphone input...")
    stream = sd.InputStream(
        callback=audio_callback,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        channels=1
    )
    stream.start()
    while True:
        time.sleep(1)

def start_flask():
    logging.info("Overlay at: http://127.0.0.1:5000")
    app.run(port=5000, host="127.0.0.1", debug=False)

if __name__ == "__main__":
    threading.Thread(target=process_audio, daemon=True).start()
    threading.Thread(target=start_mic, daemon=True).start()
    start_flask()
