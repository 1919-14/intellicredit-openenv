# IntelliCredit-CreditAppraisal-v1 — HuggingFace Spaces Dockerfile
# Docs: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.10-slim

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
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose the HuggingFace required port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server on port 7860 (HF Spaces requirement)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
