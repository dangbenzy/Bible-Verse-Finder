"""
STEP 2 - Embed All Bible Verses
=================================
Run this ONCE after step1. It converts every verse into a vector and saves
them to a file so live search is instant.

Takes 5–15 minutes on CPU. Only needs to be done once ever.

Usage:
    pip install sentence-transformers
    python step2_embed_verses.py
"""

import sqlite3
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# ── CONFIG ──────────────────────────────────────────────────────────────────
DB_FILE      = r"C:\Users\HP\Desktop\Bible\Data\bible.db"         # from step 1
VECTORS_FILE = r"C:\Users\HP\Desktop\Bible\Data\bible_vectors.pkl" # output
MODEL_NAME   = "all-MiniLM-L6-v2"                                  # fast & free
# ────────────────────────────────────────────────────────────────────────────


def embed_verses():
    # Load all verses from DB
    print("📖 Loading verses from database...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, book, chapter, verse, text FROM verses")
    rows = cursor.fetchall()
    conn.close()

    print(f"✅ Loaded {len(rows)} verses")

    ids      = [r[0] for r in rows]
    books    = [r[1] for r in rows]
    chapters = [r[2] for r in rows]
    verses   = [r[3] for r in rows]
    texts    = [r[4] for r in rows]

    # Load embedding model (downloads ~80MB on first run)
    print(f"\n🤖 Loading model '{MODEL_NAME}' (downloads once ~80MB)...")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model loaded")

    # Embed all verses in batches (shows progress)
    print(f"\n⚙️  Embedding {len(texts)} verses... (this takes 5–15 mins on CPU)")
    print("Please wait...\n")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"\n✅ Embeddings shape: {embeddings.shape}")

    # Save everything to a pickle file
    data = {
        "ids":        ids,
        "books":      books,
        "chapters":   chapters,
        "verses":     verses,
        "texts":      texts,
        "embeddings": embeddings
    }

    with open(VECTORS_FILE, "wb") as f:
        pickle.dump(data, f)

    size_mb = os.path.getsize(VECTORS_FILE) / (1024 * 1024)
    print(f"✅ Vectors saved to: {VECTORS_FILE} ({size_mb:.1f} MB)")
    print("\n🎉 Done! You can now run step3_live_search.py")


if __name__ == "__main__":
    embed_verses()
