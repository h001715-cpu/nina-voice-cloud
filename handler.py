# RunPod Serverless handler v2 — Nina voice (CosyVoice3 SFT epoch_10 + zero-shot ref st_0030)
# Input:  {"input": {"text": "...", "emotion": "neutral"}}
# Output: streamed chunks {"pcm_b64": ..., "rate": 24000} per sentence group
#         (protocol identical to v1 — nina-tts.js works unchanged)
import base64
import faulthandler
import os
import re
import sys
import time

# C레벨 크래시(segfault/CUDA illegal memory) 스택을 stderr로 덤프 — 하드 크래시 원인 규명
faulthandler.enable()

sys.path.insert(0, '/app/cosyvoice')
sys.path.insert(0, '/app/cosyvoice/third_party/Matcha-TTS')

import numpy as np
import runpod
import torch

# 클라우드 Ampere GPU에서 추론 시 하드 크래시(CUDNN_STATUS_EXECUTION_FAILED) 방지.
# CosyVoice flow-matching/보코더의 conv가 GPU별 cuDNN 커널에서 실패 → cuDNN 자체를 끈다.
# (로컬 4070에선 되는데 클라우드 Ampere에선 터지던 그 크래시의 표준 워크어라운드)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# (CUDA_LAUNCH_BLOCKING은 warmup을 수십배 느리게 해 서버리스 워커 부팅을 막음 — 제거.
#  faulthandler가 하드 크래시의 C 스택을 이미 stderr로 덤프하므로 진단엔 충분)


def _diag():
    try:
        print(f"[nina-diag] torch {torch.__version__} cuda={torch.version.cuda} "
              f"avail={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}", flush=True)
    except Exception as e:
        print(f"[nina-diag] gpu info 실패: {e}", flush=True)


_diag()

MODEL_DIR = os.environ.get("NINA_MODEL_DIR", "/app/pretrained/Fun-CosyVoice3-0.5B-2512")
FT_LLM = os.environ.get("NINA_FT_LLM", "/app/models/nina_llm_fp16.pt")
REF_WAV = os.environ.get("NINA_REF_WAV", "/app/ref/st_0030.wav")
REF_TEXT = 'You are a helpful assistant.<|endofprompt|>이 정도가 한계네요. 한심해 보이려나요? 미력하기나마.'

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

# fp16은 일부 클라우드 GPU에서 하드 크래시(CUDA) 유발 — 안정성 위해 fp32 기본, env로 토글
USE_FP16 = os.environ.get("NINA_FP16", "0") == "1"
print(f"[nina-cloud] loading CosyVoice3 (fp16={USE_FP16})...", flush=True)
t0 = time.time()
cv = CosyVoice3(MODEL_DIR, fp16=USE_FP16)
ck = torch.load(FT_LLM, map_location='cpu')
if not USE_FP16:
    ck = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck.items()}  # fp16 파일→fp32 정합
cv.model.llm.load_state_dict(ck, strict=True)
cv.model.llm.cuda().eval()
del ck
torch.cuda.empty_cache()
cv.add_zero_shot_spk(REF_TEXT, REF_WAV, 'nina')  # 레퍼런스 특징 1회 추출 캐시
SR = cv.sample_rate
# warmup — 실패해도 워커는 살린다. faulthandler가 C레벨 크래시도 stderr로 덤프.
try:
    print("[nina-diag] warmup 추론 시작...", flush=True)
    sys.stderr.flush()
    _n = 0
    for _out in cv.inference_zero_shot('오늘도 잘 부탁해요 편하게 말 걸어요.', REF_TEXT, REF_WAV, zero_shot_spk_id='nina', stream=False):
        _n += 1
        print(f"[nina-diag] warmup 청크 {_n} shape={tuple(_out['tts_speech'].shape)}", flush=True)
    print(f"[nina-cloud] loaded+warm in {time.time()-t0:.1f}s (sr={SR}), 청크 {_n}", flush=True)
except Exception as _e:
    import traceback
    print(f"[nina-cloud] warmup 실패(무시하고 기동): {_e}", flush=True)
    traceback.print_exc()


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
    groups = [g for g in groups if g]
    # 초단문 병합: 너무 짧은 조각은 conv 커널 크기보다 작아 모델이 크래시 → 이웃과 합침
    merged = []
    for g in groups:
        if merged and (len(g) < 12 or len(merged[-1]) < 12):
            merged[-1] = (merged[-1] + " " + g).strip()
        else:
            merged.append(g)
    return merged


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
    yielded = False
    for s in split_chunks(text):
        t = time.time()
        try:
            # 문장 격리: 한 문장이 죽어도 워커가 죽지 않고 다음 문장으로 (커널 에러 등)
            for out in cv.inference_zero_shot(s, REF_TEXT, REF_WAV, zero_shot_spk_id='nina', stream=False):
                wav = out['tts_speech']
                print(f"[nina-cloud] '{s[:20]}' {time.time()-t:.2f}s ({wav.shape[1]/SR:.1f}s)", flush=True)
                yield {"pcm_b64": base64.b64encode(to_s16le(wav)).decode(), "rate": int(SR)}
                yielded = True
                t = time.time()
        except Exception as e:
            import traceback
            print(f"[nina-cloud] chunk 실패 '{s[:20]}': {e}", flush=True)
            traceback.print_exc()  # 정확한 원인을 워커 로그에 남김
            continue
    if not yielded:
        yield {"error": "synthesis failed"}


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
