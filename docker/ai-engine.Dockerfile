# AI engine CLI + Jupyter + UP2 evidence/test runner. Build context = repo root.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
# git: the evidence notebook prints the HEAD (absent .git -> it shows "unknown", harmless)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /work/AI-engine
# CPU torch first, or pip pulls the multi-GB CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY AI-engine/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt jupyterlab
# layout the notebook and the COE simulator expect: /work/{AI-engine,evidence,scripts}
COPY AI-engine/ ./
COPY evidence/ /work/evidence/
COPY scripts/coe_fault_scenario.py /work/scripts/coe_fault_scenario.py
RUN useradd -m -u 1000 app && chown -R app /work
USER app
EXPOSE 8888
CMD ["python", "app.py", "selftest"]
