import aiosqlite
import os
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

class TokenDatabase:
    def __init__(self, db_path: str = "token_counts.db"):
        self.db_path = db_path
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensure the directory for the database exists."""
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        """Initialize the database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS token_counts (
                    endpoint TEXT,
                    model TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    PRIMARY KEY(endpoint, model)
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS token_time_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    timestamp INTEGER,  -- Unix timestamp with approximate minute/hour precision
                    FOREIGN KEY(endpoint, model) REFERENCES token_counts(endpoint, model)
                )
            ''')
            await db.commit()

    async def update_token_counts(self, endpoint: str, model: str, input_tokens: int, output_tokens: int):
        """Update token counts for a specific endpoint and model."""
        total_tokens = input_tokens + output_tokens
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO token_counts (endpoint, model, input_tokens, output_tokens, total_tokens)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(endpoint, model) DO UPDATE SET
                    input_tokens = input_tokens + ?,
                    output_tokens = output_tokens + ?,
                    total_tokens = total_tokens + ?
            ''', (endpoint, model, input_tokens, output_tokens, total_tokens, input_tokens, output_tokens, total_tokens))
            await db.commit()

    async def add_time_series_entry(self, endpoint: str, model: str, input_tokens: int, output_tokens: int):
        """Add a time series entry with approximate timestamp."""
        total_tokens = input_tokens + output_tokens
        # Use current minute/hour as approximate timestamp
        now = datetime.now(tz=timezone.utc)
        timestamp = int(datetime(now.year, now.month, now.day, now.hour, now.minute).timestamp())

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO token_time_series (endpoint, model, input_tokens, output_tokens, total_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (endpoint, model, input_tokens, output_tokens, total_tokens, timestamp))
            await db.commit()

    async def update_batched_counts(self, counts: dict):
        """Update multiple token counts in a single transaction."""
        if not counts:
            return
        async with aiosqlite.connect(self.db_path) as db:
            for endpoint, models in counts.items():
                for model, (input_tokens, output_tokens) in models.items():
                    total_tokens = input_tokens + output_tokens
                    await db.execute('''
                        INSERT INTO token_counts (endpoint, model, input_tokens, output_tokens, total_tokens)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(endpoint, model) DO UPDATE SET
                            input_tokens = input_tokens + ?,
                            output_tokens = output_tokens + ?,
                            total_tokens = total_tokens + ?
                    ''', (endpoint, model, input_tokens, output_tokens, total_tokens,
                         input_tokens, output_tokens, total_tokens))
            await db.commit()

    async def add_batched_time_series(self, entries: list):
        """Add multiple time series entries in a single transaction."""
        async with aiosqlite.connect(self.db_path) as db:
            for entry in entries:
                await db.execute('''
                    INSERT INTO token_time_series (endpoint, model, input_tokens, output_tokens, total_tokens, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (entry['endpoint'], entry['model'], entry['input_tokens'],
                     entry['output_tokens'], entry['total_tokens'], entry['timestamp']))
            await db.commit()

    async def load_token_counts(self):
        """Load all token counts from database."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT endpoint, model, input_tokens, output_tokens, total_tokens FROM token_counts') as cursor:
                async for row in cursor:
                    yield {
                        'endpoint': row[0],
                        'model': row[1],
                        'input_tokens': row[2],
                        'output_tokens': row[3],
                        'total_tokens': row[4]
                    }

    async def get_latest_time_series(self, limit: int = 100):
        """Get the latest time series entries."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('''
                SELECT endpoint, model, input_tokens, output_tokens, total_tokens, timestamp
                FROM token_time_series
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,)) as cursor:
                async for row in cursor:
                    yield {
                        'endpoint': row[0],
                        'model': row[1],
                        'input_tokens': row[2],
                        'output_tokens': row[3],
                        'total_tokens': row[4],
                        'timestamp': row[5]
                    }

    async def get_token_counts_for_model(self, model):
        """Get token counts for a specific model."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT endpoint, model, input_tokens, output_tokens, total_tokens FROM token_counts WHERE model = ?', (model,)) as cursor:
                async for row in cursor:
                    return {
                        'endpoint': row[0],
                        'model': row[1],
                        'input_tokens': row[2],
                        'output_tokens': row[3],
                        'total_tokens': row[4]
                    }
            return None
