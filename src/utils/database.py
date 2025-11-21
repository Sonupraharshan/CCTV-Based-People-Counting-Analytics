"""
Database Utilities for Event Logging
SQLite database management for counting events with CSV export.
"""

import sqlite3
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path
import threading
from datetime import datetime
from src.counter import CountEvent


class Database:
    """
    Thread-safe SQLite database manager for event logging.
    
    Stores counting events with timestamps, track IDs, directions, and metadata.
    """
    
    def __init__(self, db_path: str = "database/events.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connection per thread
        self.local = threading.local()
        
        # Initialize schema
        self._init_schema()
        
        print(f"Database initialized: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    def _init_schema(self):
        """Create database schema if not exists."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                direction TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                confidence REAL,
                crossing_x REAL,
                crossing_y REAL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_direction ON events(direction)
        """)
        
        conn.commit()
        print("Database schema initialized")
    
    def insert_event(self, event: CountEvent):
        """
        Insert a counting event into database.
        
        Args:
            event: CountEvent object to insert
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert timestamp to datetime string
        dt_str = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute("""
            INSERT INTO events (
                timestamp, datetime, track_id, direction,
                frame_number, confidence, crossing_x, crossing_y
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            dt_str,
            event.track_id,
            event.direction,
            event.frame_number,
            event.confidence,
            event.crossing_point[0],
            event.crossing_point[1]
        ))
        
        conn.commit()
    
    def insert_events_batch(self, events: List[CountEvent]):
        """Insert multiple events in a batch."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        data = [
            (
                event.timestamp,
                datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                event.track_id,
                event.direction,
                event.frame_number,
                event.confidence,
                event.crossing_point[0],
                event.crossing_point[1]
            )
            for event in events
        ]
        
        cursor.executemany("""
            INSERT INTO events (
                timestamp, datetime, track_id, direction,
                frame_number, confidence, crossing_x, crossing_y
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        print(f"Inserted {len(events)} events")
    
    def get_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        direction: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query events with optional filters.
        
        Args:
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            direction: Filter by direction ('IN' or 'OUT')
            limit: Maximum number of events to return
        
        Returns:
            DataFrame with event data
        """
        conn = self._get_connection()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if direction is not None:
            query += " AND direction = ?"
            params.append(direction)
        
        query += " ORDER BY timestamp DESC"
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    
    def get_count_by_time(
        self,
        interval: str = 'hour',
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get aggregated counts by time interval.
        
        Args:
            interval: Time interval ('minute', 'hour', 'day')
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            DataFrame with time-based counts
        """
        conn = self._get_connection()
        
        # SQL format strings for different intervals
        formats = {
            'minute': '%Y-%m-%d %H:%M',
            'hour': '%Y-%m-%d %H:00',
            'day': '%Y-%m-%d'
        }
        
        if interval not in formats:
            raise ValueError(f"Interval must be one of: {list(formats.keys())}")
        
        time_format = formats[interval]
        
        query = f"""
            SELECT
                strftime('{time_format}', datetime) as time_period,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as count_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as count_out,
                COUNT(*) as total_count
            FROM events
            WHERE 1=1
        """
        
        params = []
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " GROUP BY time_period ORDER BY time_period"
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    
    def get_statistics(self) -> dict:
        """Get overall statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                COUNT(*) as total_events,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                MIN(timestamp) as first_event_time,
                MAX(timestamp) as last_event_time
            FROM events
        """)
        
        row = cursor.fetchone()
        
        return {
            'total_events': row[0],
            'total_in': row[1] or 0,
            'total_out': row[2] or 0,
            'first_event_time': row[3],
            'last_event_time': row[4]
        }
    
    def export_csv(self, output_path: str, **filters):
        """
        Export events to CSV file.
        
        Args:
            output_path: Path to save CSV
            **filters: Optional filters (start_time, end_time, direction)
        """
        # Get events
        df = self.get_events(**filters)
        
        # Export
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} events to {output_path}")
    
    def clear_events(self):
        """Clear all events from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM events")
        conn.commit()
        
        print("All events cleared")
    
    def close(self):
        """Close database connection."""
        if hasattr(self.local, 'conn'):
            self.local.conn.close()
            delattr(self.local, 'conn')


def create_database_schema():
    """Standalone function to create database schema."""
    db = Database()
    db.close()
    print("Database created successfully")


if __name__ == '__main__':
    # Create database if run directly
    create_database_schema()
