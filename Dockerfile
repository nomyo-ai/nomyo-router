FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install SQLite
RUN apt-get update && apt-get install -y sqlite3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create database directory and set permissions
RUN mkdir -p /app/data && chown -R www-data:www-data /app/data

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 12434

ENTRYPOINT ["/app/entrypoint.sh"]
