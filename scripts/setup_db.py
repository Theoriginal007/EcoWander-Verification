#!/usr/bin/env python3
"""
Initialize the EcoWander verification database.
"""
import sqlite3
from pathlib import Path
from ecowander.config import settings

DB_PATH = Path(__file__).parent.parent / "data" / "ecowander.db"

def init_database():
    """Initialize SQLite database with required tables."""
    try:
        DB_PATH.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Create verification records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    image_hash TEXT UNIQUE,
                    challenge_type TEXT,
                    confidence REAL,
                    location_score REAL,
                    fraud_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create eco_locations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eco_locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    latitude REAL,
                    longitude REAL,
                    radius_meters INTEGER,
                    challenge_types TEXT,
                    description TEXT
                )
            """)
            
            conn.commit()
            print(f"Database initialized at {DB_PATH}")
            
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    init_database()