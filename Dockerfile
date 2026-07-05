FROM mirror.gcr.io/library/python:3.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install numpy==1.26.4 scipy==1.10.1

RUN pip install scikit-surprise==1.1.4

RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu

RUN pip install \
        gdown==5.2.0 \
        matplotlib==3.5.1 \
        nltk==3.9.1 \
        pandas==1.4.2 \
        scikit-learn==1.0.2 \
        streamlit==1.46.1 \
        vaderSentiment==3.3.2

RUN pip install --no-deps xgboost==2.1.4

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
