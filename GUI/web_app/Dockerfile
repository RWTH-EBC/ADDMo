FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# System dependencies (git for version info, curl for healthcheck, build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy your whole repo into the image (build context will be repo root)
COPY . /app

# Install your package and Streamlit
# If your project is pip-installable (has pyproject.toml), this will install it.
RUN python -m pip install --upgrade pip && \
    pip install . && \
    pip install streamlit

# Streamlit default port
EXPOSE 8501

# Healthcheck: Streamlit exposes _stcore/health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Use your launcher (it calls: streamlit run GUI/app.py)
ENTRYPOINT ["python", "GUI/launch_GUI.py"]
