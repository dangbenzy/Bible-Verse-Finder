"""
STEP 1 - Build Bible SQLite Database
=====================================
Run this once to convert your KJV JSON file into a SQLite database.

Usage:
    python step1_build_bible_db.py
"""

import json
import sqlite3
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
JSON_FILE = r"C:\Users\HP\Desktop\Bible\Data\en_kjv.json"   # path to your JSON file
DB_FILE   = r"C:\Users\HP\Desktop\Bible\Data\bible.db"      # output SQLite database
# ────────────────────────────────────────────────────────────────────────────


def build_database():
    print("📖 Loading JSON file...")
    with open(JSON_FILE, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    print(f"✅ Found {len(data)} books")

    # Create / connect to SQLite database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table
    cursor.execute("DROP TABLE IF EXISTS verses")
    cursor.execute("""
        CREATE TABLE verses (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            book    TEXT NOT NULL,
            abbrev  TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse   INTEGER NOT NULL,
            text    TEXT NOT NULL
        )
    """)

    # Insert all verses
    total = 0
    for book in data:
        book_name = book["name"]
        abbrev    = book["abbrev"]
        chapters  = book["chapters"]

        for chapter_idx, chapter in enumerate(chapters):
            chapter_num = chapter_idx + 1
            for verse_idx, verse_text in enumerate(chapter):
                verse_num = verse_idx + 1
                cursor.execute(
                    "INSERT INTO verses (book, abbrev, chapter, verse, text) VALUES (?, ?, ?, ?, ?)",
                    (book_name, abbrev, chapter_num, verse_num, verse_text)
                )
                total += 1

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM verses")
    count = cursor.fetchone()[0]
    print(f"✅ {count} verses inserted into database")

    # Show sample
    print("\n--- Sample verses ---")
    cursor.execute("SELECT book, chapter, verse, text FROM verses LIMIT 3")
    for row in cursor.fetchall():
        print(f"{row[0]} {row[1]}:{row[2]} → {row[3][:80]}...")

    # Show book list
    print("\n--- Books in database ---")
    cursor.execute("SELECT DISTINCT book FROM verses")
    books = [r[0] for r in cursor.fetchall()]
    print(", ".join(books))

    conn.close()
    print(f"\n✅ Database saved to: {DB_FILE}")


if __name__ == "__main__":
    build_database()
