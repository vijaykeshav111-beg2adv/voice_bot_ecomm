FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY . .
# Server settings: listen on all interfaces inside the container, groq does the speech-to-text
# (no local Whisper model to download on a small server)
ENV HOST=0.0.0.0 PORT=8000 STT_BACKEND=groq PRELOAD_LOCAL_FALLBACK=0
EXPOSE 8000
CMD ["python", "server.py"]
