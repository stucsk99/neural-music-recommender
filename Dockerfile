FROM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y \
libsndfile1 \
ffmpeg \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip==23.3.2 \
 && pip install --no-cache-dir --ignore-installed -r requirements.txt


COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
