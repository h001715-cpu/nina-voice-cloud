# Nina voice — RunPod Serverless image v2
# CosyVoice3 SFT (Heo Ye-eun epoch_10 fp16) + zero-shot ref st_0030
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CosyVoice (third_party Matcha-TTS 포함)
RUN git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git cosyvoice

# 로컬 검증 환경의 "실제 임포트" 목록에서 추출한 버전 고정
# (WeTextProcessing/pynini 대신 wetext 사용, torchaudio는 베이스에 없어 명시 설치)
RUN pip install --no-cache-dir \
    runpod \
    torchaudio==2.3.1 \
    transformers==4.51.3 modelscope==1.20.0 HyperPyYAML==1.2.3 \
    onnxruntime==1.18.0 onnx==1.16.0 wetext==0.0.4 \
    librosa==0.10.2 soundfile==0.12.1 numpy==1.26.4 \
    openai-whisper==20250625 pyarrow==18.1.0 matplotlib \
    omegaconf==2.3.0 inflect==7.3.1 \
    conformer==0.3.2 diffusers==0.29.0 lightning==2.2.4 \
    "huggingface_hub[hf_transfer]"

# 순수 모듈 임포트만 검사 (GPU/모델 불필요 — 빌드 머신에 GPU 없으므로 여기까지만).
# CosyVoice3 조립·추론 검증은 런타임 워밍업(비치명, traceback 로깅)이 담당한다.
ENV PYTHONPATH=/app/cosyvoice:/app/cosyvoice/third_party/Matcha-TTS
RUN python -c "import whisper, torchaudio, librosa, wetext, runpod, onnxruntime, pyarrow, conformer, diffusers, lightning; print('MODULE IMPORT OK')"

# 베이스 모델 (HF 공개 저장소, llm.rl.pt 제외)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/app/pretrained/Fun-CosyVoice3-0.5B-2512', ignore_patterns=['llm.rl.pt','speech_tokenizer_v3.batch.onnx'])"

# 파인튜닝 llm (fp16, GitHub 95MB 제한 분할 → 결합)
COPY models/ /app/models/
RUN cat /app/models/cv3_llm_fp16.pt.part* > /app/models/nina_llm_fp16.pt && rm /app/models/cv3_llm_fp16.pt.part*

# (조립/추론 검증은 런타임 warmup으로 이동 — 빌드 머신엔 GPU가 없어 여기서 하면 빌드가 깨짐)

# 제로샷 레퍼런스 + 핸들러
COPY ref/ /app/ref/
COPY handler.py /app/handler.py

ENV PYTHONPATH=/app/cosyvoice:/app/cosyvoice/third_party/Matcha-TTS
ENV NINA_MODEL_DIR=/app/pretrained/Fun-CosyVoice3-0.5B-2512
ENV NINA_FT_LLM=/app/models/nina_llm_fp16.pt
ENV NINA_REF_WAV=/app/ref/st_0030.wav

CMD ["python", "-u", "/app/handler.py"]
