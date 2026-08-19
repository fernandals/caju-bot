# One image per course: bakes in that course's pre-built knowledge base
# and student roster. Build a separate image per course rather than
# switching disciplines at runtime.
#
# Prepare the course BEFORE building this image:
#   1. Ingest the course PDFs with the existing app.py / pages/create_kb.py
#      flow (or scripts/create_kb.py) - produces data/knowledge_bases/<id>/
#   2. Write data/roster.txt - one valid enrollment ID per line
# See DEPLOY.md for the full walkthrough.

FROM python:3.12-slim

WORKDIR /app

# Some chromadb dependencies (onnxruntime, tokenizers) don't always ship
# prebuilt wheels for every platform; build-essential is a safety net for
# a from-source install. If pip installs cleanly without it on your build
# platform, it can be dropped to shrink the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY agent/ agent/
COPY rag/ rag/
COPY utils/ utils/
COPY pages/ pages/
COPY .streamlit/ .streamlit/
COPY app.py student_app.py main.py ./

# This course's pre-built knowledge base and roster, prepared beforehand
# (see DEPLOY.md) - not rebuilt at container start.
COPY data/knowledge_bases/ data/knowledge_bases/
COPY data/roster.txt data/roster.txt

# data/chats/ is intentionally NOT baked in - created at runtime and
# expected to be mounted as a volume so history survives restarts.
RUN mkdir -p data/chats

EXPOSE 8501

CMD ["streamlit", "run", "student_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
