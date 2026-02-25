FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install SQLite
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y sqlite3

WORKDIR /app
COPY requirements.txt .
RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action=ignore --no-cache-dir -r requirements.txt

# Create database directory and set permissions
RUN mkdir -p /app/data && chown -R www-data:www-data /app/data

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 12434

ENTRYPOINT ["/app/entrypoint.sh"]
