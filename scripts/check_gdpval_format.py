#!/usr/bin/env python3
import sqlite3
import json
import struct

# Connect to the dataset
db_path = "/Users/joemiles/.cache/burn-dataset/openaigdpval.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get the first row
cursor.execute("SELECT * FROM train LIMIT 1")
columns = [desc[0] for desc in cursor.description]
row = cursor.fetchone()

print("Columns:", columns)
print("\nRaw data for first row:")

for col, val in zip(columns, row):
    print(f"\n{col}:")
    if isinstance(val, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(val)
            print(f"  JSON: {json.dumps(parsed, indent=2)[:200]}...")
        except:
            # Show raw string preview
            print(f"  String: {repr(val[:100])}...")
    elif isinstance(val, bytes):
        # Try to decode as string
        try:
            decoded = val.decode('utf-8')
            print(f"  Decoded: {decoded[:100]}...")
        except:
            print(f"  Binary ({len(val)} bytes): {val[:20].hex()}...")
    else:
        print(f"  {type(val).__name__}: {val}")

conn.close()
