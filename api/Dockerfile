FROM python:3.13-slim

WORKDIR /app/api

COPY api/app.py /app/api/app.py
COPY models/best_model.pkl /app/models/best_model.pkl

RUN pip install --no-cache-dir fastapi uvicorn pandas joblib scikit-learn numpy

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]