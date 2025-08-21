FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

RUN useradd -m -u 10001 appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* \

WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir streamlit
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501


HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1


ENTRYPOINT ["python", "GUI/launch_GUI.py"]
