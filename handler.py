# RunPod Serverless handler — Nina voice chain (Qwen3-TTS Sohee → RVC Heo Ye-eun)
# Input:  {"input": {"text": "...", "emotion": "neutral"}}
# Output: streamed chunks {"pcm_b64": ..., "rate": 40000} per sentence group
import base64
import os
import re
import sys
import tempfile
import time

import numpy as np
import runpod
import soundfile as sf
import torch

# ── Qwen3-TTS ──
from qwen_tts import Qwen3TTSModel

MODEL_ID = os.environ.get("NINA_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
SPEAKER = os.environ.get("NINA_SPEAKER", "sohee")

EMOTION_INSTRUCT = {
    "neutral":   "차분하고 담담한, 약간 무심한 말투로",
    "caring":    "겉으론 무심한 척하지만 다정함이 묻어나는 부드러운 말투로",
    "scolding":  "한숨 섞인 짜증으로 차갑게 쏘아붙이는 말투로",
    "shy":       "수줍어서 작아지는 목소리로, 말끝을 흐리며",
    "pleased":   "옅은 웃음기가 섞인 흐뭇한 말투로",
    "worried":   "걱정이 배어나는 조심스러운 말투로",
    "nostalgic": "옛 생각에 잠긴 듯 그리움이 묻어나는 잔잔한 말투로",
}

print("[nina-cloud] loading TTS...", flush=True)
t0 = time.time()
tts = Qwen3TTSModel.from_pretrained(
    MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
)
print(f"[nina-cloud] TTS loaded in {time.time()-t0:.1f}s", flush=True)

# ── RVC (Applio infer core) ──
APPLIO_DIR = os.environ.get("APPLIO_DIR", "/app/applio")
sys.path.insert(0, APPLIO_DIR)
os.chdir(APPLIO_DIR)
from core import run_infer_script  # noqa: E402

RVC_PTH = os.environ.get("NINA_RVC_PTH", "/app/models/nina_heo_ko.pth")
RVC_INDEX = os.environ.get("NINA_RVC_INDEX", "/app/models/nina_heo_ko.index")
RVC_EMBEDDER = os.environ.get("NINA_RVC_EMBEDDER", "korean-hubert-base")
RVC_INDEX_RATE = float(os.environ.get("NINA_RVC_INDEX_RATE", "0.4"))


def rvc_convert(wav: np.ndarray, sr: int):
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "in.wav")
        op = os.path.join(td, "out.wav")
        sf.write(ip, np.asarray(wav, dtype=np.float32).squeeze(), sr)
        run_infer_script(
            pitch=0, index_rate=RVC_INDEX_RATE, volume_envelope=1.0, protect=0.5,
            f0_method="rmvpe", input_path=ip, output_path=op,
            pth_path=RVC_PTH, index_path=RVC_INDEX,
            split_audio=False, f0_autotune=False, f0_autotune_strength=1.0,
            proposed_pitch=False, proposed_pitch_threshold=155.0,
            clean_audio=False, clean_strength=0.2, export_format="WAV",
            embedder_model=RVC_EMBEDDER,
        )
        out, out_sr = sf.read(op, dtype="float32")
    return out, out_sr


def split_chunks(text: str):
    text = text.strip()
    if len(text) <= 26:
        return [text]
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    sents = [p.strip() for p in parts if p.strip()]
    if not sents:
        return [text]
    first = sents[0]
    if len(first) > 30:
        cut = max(first.rfind(",", 0, 26), first.rfind(" ", 0, 26))
        if cut >= 8:
            sents[0] = first[:cut + 1].strip()
            sents.insert(1, first[cut + 1:].strip())
    # 점진적 청크 크기: 20자 → 35자 → 70자… (앞은 빨리 소리내고, 재생이 생성을 따라잡게)
    limits = [35, 70]
    groups = [sents[0]]
    cur = ""
    li = 0
    for s in sents[1:]:
        lim = limits[min(li, len(limits) - 1)]
        if cur and len(cur) + len(s) + 1 > lim:
            groups.append(cur)
            cur = s
            li += 1
        else:
            cur = (cur + " " + s).strip()
    if cur:
        groups.append(cur)
    return [g for g in groups if g]


def to_s16le(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float32).squeeze(), -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


def handler(job):
    inp = job.get("input", {}) or {}
    text = (inp.get("text") or "").strip()[:1000]
    emotion = (inp.get("emotion") or "neutral").lower()
    if not text:
        yield {"error": "empty text"}
        return
    instruct = EMOTION_INSTRUCT.get(emotion, EMOTION_INSTRUCT["neutral"])
    for s in split_chunks(text):
        t = time.time()
        with torch.inference_mode():
            wavs, sr = tts.generate_custom_voice(
                text=s, language="Korean", speaker=SPEAKER, instruct=instruct,
            )
        wav, out_sr = rvc_convert(wavs[0], sr)
        print(f"[nina-cloud] [{emotion}] '{s[:20]}' {time.time()-t:.2f}s", flush=True)
        yield {"pcm_b64": base64.b64encode(to_s16le(wav)).decode(), "rate": int(out_sr)}


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
