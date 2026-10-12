FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["python", "-c", "from pathlib import Path; import uvicorn; from climate_ts.api import create_app; uvicorn.run(create_app(Path('/app')), host='0.0.0.0', port=8000)"]
