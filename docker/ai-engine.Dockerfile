# AI engine CLI + Jupyter. Build context = repo root.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /work/AI-engine
# CPU torch first, or pip pulls the multi-GB CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY AI-engine/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt jupyterlab nbconvert
COPY AI-engine/ ./
RUN useradd -m -u 1000 app && chown -R app /work
USER app
EXPOSE 8888
CMD ["python", "app.py", "selftest"]
