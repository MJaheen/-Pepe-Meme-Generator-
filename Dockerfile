FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy app and dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

# Fix Streamlit permission issue
ENV HOME=/tmp
RUN mkdir -p /tmp/.streamlit
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
