# RetailPulse dashboard. Build: docker build -t retailpulse .   Run: docker run -p 8501:8501 retailpulse
FROM python:3.11-slim

WORKDIR /app
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Only what the dashboard reads (large raw files and caches are excluded in .dockerignore)
COPY app/ app/
COPY src/__init__.py src/segmentation.py src/
COPY data/processed/ data/processed/
COPY reports/figures/ reports/figures/
COPY retail_clean_dataset/store_master.csv retail_clean_dataset/sku_master.csv retail_clean_dataset/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
