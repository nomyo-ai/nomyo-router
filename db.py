import aiosqlite, asyncio
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

class TokenDatabase:
    def __init__(self, db_path: str = "token_counts.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._db_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    def _ensure_db_directory(self):
        """Ensure the directory for the database exists."""
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

    async def _get_connection(self) -> aiosqlite.Connection:
        """Return a persistent connection with WAL mode and FK enforcement enabled."""
        if self._db is None:
            async with self._db_lock:
                if self._db is None:
                    self._ensure_db_directory()
                    self._db = await aiosqlite.connect(self.db_path)
                    # Enable WAL and foreign keys for reliability and integrity
                    await self._db.execute("PRAGMA journal_mode=WAL;")
                    await self._db.execute("PRAGMA foreign_keys = ON;")
                    await self._db.commit()
        return self._db

    async def close(self):
        """Close the persistent database connection, if open."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def init_db(self):
        """Initialize the database tables."""
        db = await self._get_connection()
        async with self._operation_lock:
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
                    timestamp INTEGER,
                    FOREIGN KEY(endpoint, model) REFERENCES token_counts(endpoint, model)
                )
            ''')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_token_time_series_timestamp ON token_time_series(timestamp)')
            await db.commit()

    async def update_token_counts(self, endpoint: str, model: str, input_tokens: int, output_tokens: int):
        """Update token counts for a specific endpoint and model."""
        total_tokens = input_tokens + output_tokens
        db = await self._get_connection()
        async with self._operation_lock:
            await db.execute('''
                INSERT INTO token_counts (endpoint, model, input_tokens, output_tokens, total_tokens)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (endpoint, model) DO UPDATE SET
                    input_tokens = input_tokens + ?,
                    output_tokens = output_tokens + ?,
                    total_tokens = total_tokens + ?
            ''', (endpoint, model, input_tokens, output_tokens, total_tokens, input_tokens, output_tokens, total_tokens))
            await db.commit()

    async def add_time_series_entry(self, endpoint: str, model: str, input_tokens: int, output_tokens: int):
        """Add a time series entry with approximate timestamp."""
        total_tokens = input_tokens + output_tokens
        # Use current minute/hour as approximate timestamp in UTC
        now = datetime.now(tz=timezone.utc)
        timestamp = int(datetime(now.year, now.month, now.day, now.hour, now.minute).timestamp())

        db = await self._get_connection()
        async with self._operation_lock:
            await db.execute('''
                INSERT INTO token_time_series (endpoint, model, input_tokens, output_tokens, total_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (endpoint, model, input_tokens, output_tokens, total_tokens, timestamp))
            await db.commit()

    async def update_batched_counts(self, counts: dict):
        """Update multiple token counts in a single transaction."""
        if not counts:
            return
        db = await self._get_connection()
        async with self._operation_lock:
            try:
                await db.execute('BEGIN')
                for endpoint, models in counts.items():
                    for model, (input_tokens, output_tokens) in models.items():
                        total_tokens = input_tokens + output_tokens
                        await db.execute('''
                            INSERT INTO token_counts (endpoint, model, input_tokens, output_tokens, total_tokens)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (endpoint, model) DO UPDATE SET
                                input_tokens = input_tokens + ?,
                                output_tokens = output_tokens + ?,
                                total_tokens = total_tokens + ?
                        ''', (endpoint, model, input_tokens, output_tokens, total_tokens,
                              input_tokens, output_tokens, total_tokens))
                await db.commit()
            except Exception:
                # Rollback on error to maintain consistency
                try:
                    await db.execute('ROLLBACK')
                except Exception:
                    pass
                raise

    async def add_batched_time_series(self, entries: list):
        """Add multiple time series entries in a single transaction."""
        db = await self._get_connection()
        async with self._operation_lock:
            try:
                await db.execute('BEGIN')
                for entry in entries:
                    await db.execute('''
                        INSERT INTO token_time_series (endpoint, model, input_tokens, output_tokens, total_tokens, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (entry['endpoint'], entry['model'], entry['input_tokens'],
                          entry['output_tokens'], entry['total_tokens'], entry['timestamp']))
                await db.commit()
            except Exception:
                try:
                    await db.execute('ROLLBACK')
                except Exception:
                    pass
                raise

    async def load_token_counts(self):
        """Load all token counts from database."""
        db = await self._get_connection()
        async with self._operation_lock:
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
        db = await self._get_connection()
        async with self._operation_lock:
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
        """Get token counts for a specific model, aggregated across all endpoints."""
        db = await self._get_connection()
        async with self._operation_lock:
            async with db.execute('''
                SELECT
                    'aggregated' as endpoint,
                    ? as model,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(total_tokens) as total_tokens
                FROM token_counts
                WHERE model = ?
            ''', (model, model)) as cursor:
                row = await cursor.fetchone()
                if row is not None:
                    return {
                        'endpoint': row[0],
                        'model': row[1],
                        'input_tokens': row[2],
                        'output_tokens': row[3],
                        'total_tokens': row[4]
                    }
        return None

    async def aggregate_time_series_older_than(self, days: int, trim_old: bool = False) -> int:
        """
        Aggregate time_series entries older than 'days' days into daily aggregates by
        endpoint, model and UTC date (YYYY-MM-DD). The results are stored in
        token_time_series_daily with a UNIQUE constraint on (endpoint, model, date).

        Returns the number of aggregated groups (distinct (endpoint, model, date) tuples)
        that were created/updated.
        """
        if not isinstance(days, int) or days <= 0:
            days = 30

        cutoff_ts = int(datetime.now(tz=timezone.utc).timestamp()) - (days * 86400)

        db = await self._get_connection()
        aggregated_count = 0

        async with self._operation_lock:
            # Ensure daily table exists
            await db.execute('''
                CREATE TABLE IF NOT EXISTS token_time_series_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT,
                    model TEXT,
                    date TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    UNIQUE(endpoint, model, date)
                )
            ''')
            await db.commit()

            cursor = await db.execute('''
                SELECT endpoint, model, date(timestamp, 'unixepoch') as day,
                       SUM(input_tokens) as in_sum,
                       SUM(output_tokens) as out_sum,
                       SUM(total_tokens) as tot_sum
                FROM token_time_series
                WHERE timestamp < ?
                GROUP BY endpoint, model, day
            ''', (cutoff_ts,))
            rows = await cursor.fetchall()

            for row in rows:
                endpoint, model, day, in_sum, out_sum, tot_sum = row
                await db.execute('''
                    INSERT INTO token_time_series_daily (endpoint, model, date, input_tokens, output_tokens, total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (endpoint, model, date) DO UPDATE SET
                        input_tokens = input_tokens + ?,
                        output_tokens = output_tokens + ?,
                        total_tokens = total_tokens + ?
                ''', (endpoint, model, day, int(in_sum or 0), int(out_sum or 0), int(tot_sum or 0),
                      int(in_sum or 0), int(out_sum or 0), int(tot_sum or 0)))
                aggregated_count += 1

            # Trim old entries if requested
            if trim_old:
                await db.execute('DELETE FROM token_time_series WHERE timestamp < ?', (cutoff_ts,))

            await db.commit()

        return aggregated_count
