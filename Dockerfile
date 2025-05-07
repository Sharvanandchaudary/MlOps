# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.deployment.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
