FROM python:3.12-slim

WORKDIR /app

# Install dependencies before copying application code for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, package source, configuration, and champion model artifact
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
