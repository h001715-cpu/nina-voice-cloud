# Nina voice chain — RunPod Serverless image
# Qwen3-TTS (Sohee) + Applio RVC (Heo Ye-eun ko model)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Applio (RVC inference core)
RUN git clone --depth 1 https://github.com/IAHispano/Applio applio
RUN pip install --no-cache-dir -r applio/requirements.txt || pip install --no-cache-dir faiss-cpu librosa soundfile pedalboard noisereduce
RUN pip install --no-cache-dir runpod qwen-tts soundfile

# RVC predictors (rmvpe 등) 미리 다운로드
RUN cd applio && python core.py prerequisites --models True --pretraineds_hifigan False --exe False || true

# Qwen TTS 모델을 이미지에 미리 포함 (콜드스타트 단축)
ENV HF_HOME=/app/hf
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')"

# 니나 전용 모델 (허예은 ko)
COPY models/ /app/models/
COPY handler.py /app/handler.py

ENV APPLIO_DIR=/app/applio
ENV NINA_RVC_PTH=/app/models/nina_heo_ko.pth
ENV NINA_RVC_INDEX=/app/models/nina_heo_ko.index

CMD ["python", "-u", "/app/handler.py"]
