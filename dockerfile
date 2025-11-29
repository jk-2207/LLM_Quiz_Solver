# Use an official Python slim image
FROM python:3.11-slim

# Install system deps for Playwright, Pillow, ffmpeg, libsndfile, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential ffmpeg libsndfile1 \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libx11-6 libxkbcommon0 \
    libxss1 libasound2 libgbm1 libgtk-3-0 libxrandr2 libxcomposite1 libxi6 libxcursor1 \
    && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy requirements first (speeds rebuilds)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Install Playwright browsers (non-interactive)
RUN python -m playwright install --with-deps

# Copy app code
COPY . /app

# Environment defaults (can override in Render dashboard)
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Expose port (Render uses $PORT at runtime)
EXPOSE 10000

# Start command - use uvicorn/gunicorn in production. Render will set PORT env.
CMD exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
