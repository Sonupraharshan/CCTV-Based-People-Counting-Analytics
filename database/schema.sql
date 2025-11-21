"""
SQLite Database Schema for Event Logging
"""

-- Events table for storing counting events
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('IN', 'OUT', 'ENTERED', 'EXITED')),
    frame_number INTEGER NOT NULL,
    confidence REAL,
    crossing_x REAL,
    crossing_y REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_datetime ON events(datetime);
CREATE INDEX IF NOT EXISTS idx_direction ON events(direction);
CREATE INDEX IF NOT EXISTS idx_track_id ON events(track_id);
CREATE INDEX IF NOT EXISTS idx_frame_number ON events(frame_number);

-- Optional: Sessions table to track different video processing sessions
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT UNIQUE NOT NULL,
    video_source TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    total_frames INTEGER DEFAULT 0,
    count_in INTEGER DEFAULT 0,
    count_out INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Link events to sessions (optional)
-- ALTER TABLE events ADD COLUMN session_id INTEGER REFERENCES sessions(id);
