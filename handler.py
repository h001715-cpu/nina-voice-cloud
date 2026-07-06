# RunPod Serverless handler v2 — Nina voice (CosyVoice3 SFT epoch_10 + zero-shot ref st_0030)
# Input:  {"input": {"text": "...", "emotion": "neutral"}}
# Output: streamed chunks {"pcm_b64": ..., "rate": 24000} per sentence group
#         (protocol identical to v1 — nina-tts.js works unchanged)
import base64
import os
import re
import sys
import time

sys.path.insert(0, '/app/cosyvoice')
sys.path.insert(0, '/app/cosyvoice/third_party/Matcha-TTS')

import numpy as np
import runpod
import torch

MODEL_DIR = os.environ.get("NINA_MODEL_DIR", "/app/pretrained/Fun-CosyVoice3-0.5B-2512")
FT_LLM = os.environ.get("NINA_FT_LLM", "/app/models/nina_llm_fp16.pt")
REF_WAV = os.environ.get("NINA_REF_WAV", "/app/ref/st_0030.wav")
REF_TEXT = 'You are a helpful assistant.<|endofprompt|>이 정도가 한계네요. 한심해 보이려나요? 미력하기나마.'

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

print("[nina-cloud] loading CosyVoice3...", flush=True)
t0 = time.time()
cv = CosyVoice3(MODEL_DIR, fp16=True)
ck = torch.load(FT_LLM, map_location='cpu')
cv.model.llm.load_state_dict(ck, strict=True)
cv.model.llm.cuda().eval()
del ck
cv.add_zero_shot_spk(REF_TEXT, REF_WAV, 'nina')  # 레퍼런스 특징 1회 추출 캐시
SR = cv.sample_rate
# warmup (콜드스타트에서 CUDA 커널/캐시 예열)
list(cv.inference_zero_shot('안녕.', REF_TEXT, REF_WAV, zero_shot_spk_id='nina', stream=False))
print(f"[nina-cloud] loaded+warm in {time.time()-t0:.1f}s (sr={SR})", flush=True)


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
    # 점진적 청크 크기: 앞은 빨리 소리내고, 재생이 생성을 따라잡게
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


def to_s16le(x: torch.Tensor) -> bytes:
    x = x.squeeze().cpu().numpy().astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()


def handler(job):
    inp = job.get("input", {}) or {}
    text = (inp.get("text") or "").strip()[:1000]
    if not text:
        yield {"error": "empty text"}
        return
    for s in split_chunks(text):
        t = time.time()
        for out in cv.inference_zero_shot(s, REF_TEXT, REF_WAV, zero_shot_spk_id='nina', stream=False):
            wav = out['tts_speech']
            print(f"[nina-cloud] '{s[:20]}' {time.time()-t:.2f}s ({wav.shape[1]/SR:.1f}s)", flush=True)
            yield {"pcm_b64": base64.b64encode(to_s16le(wav)).decode(), "rate": int(SR)}
            t = time.time()


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
