# IntelliCredit-CreditAppraisal-v1 — HuggingFace Spaces Dockerfile
# Docs: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# HuggingFace Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (Docker cache layer)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files
COPY --chown=user . /app

# Set Python path so server module imports work
ENV PYTHONPATH="/app"

# Expose the HuggingFace required port
EXPOSE 7860

# Health check — uses Python (curl not available in python:3.11-slim)
HEALTHCHECK --interval=15s --timeout=10s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=8)" || exit 1

# Run the FastAPI server on port 7860 (HF Spaces requirement)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
