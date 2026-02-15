FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch 2.2.0 (IMPORTANT)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]