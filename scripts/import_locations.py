#!/usr/bin/env python3
"""
Import eco-locations into the database.
"""
import json
import sqlite3
from pathlib import Path
from ecowander.config.eco_locations import KNOWN_ECO_LOCATIONS

DB_PATH = Path(__file__).parent.parent / "data" / "ecowander.db"

def import_locations():
    """Import known eco-locations into database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            for loc in KNOWN_ECO_LOCATIONS:
                cursor.execute("""
                    INSERT OR REPLACE INTO eco_locations (
                        name, latitude, longitude, radius_meters, 
                        challenge_types, description
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    loc.name,
                    loc.coordinates[0],
                    loc.coordinates[1],
                    loc.radius_meters,
                    json.dumps(loc.challenge_types),
                    loc.description
                ))
            
            conn.commit()
            print(f"Imported {len(KNOWN_ECO_LOCATIONS)} locations")
            
    except Exception as e:
        print(f"Location import failed: {str(e)}")
        raise

if __name__ == "__main__":
    import_locations()