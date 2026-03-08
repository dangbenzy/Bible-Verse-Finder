"""
STEP 3 - Live Bible Verse Search (RAG Version)
================================================
Uses a full Retrieval-Augmented Generation pipeline:

  1. UNDERSTAND  → Groq (Llama 3.1 8B) extracts intent from your query
                   e.g. "What does the bible say in John chapter one verse one"
                   → detects it as a direct reference: John 1:1

  2. RETRIEVE    → Fetches verse(s) from SQLite (direct ref)
                   OR does vector similarity search (topic query)

  3. GENERATE    → Groq (Llama 3.1 8B) reads the retrieved verses
                   and returns a natural, grounded response

Setup:
    pip install sounddevice scipy openai-whisper sentence-transformers groq
    Get a free API key at: https://console.groq.com
    Paste your key in GROQ_API_KEY below.

Controls:
    [ENTER]     → Record 15 seconds of audio
    [t + ENTER] → Type a query manually
    [q + ENTER] → Quit
"""

import sqlite3
import numpy as np
import pickle
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
import re
import json
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── CONFIG ───────────────────────────────────────────────────────────────────
VECTORS_FILE  = r"C:\Users\HP\Desktop\Bible\Data\bible_vectors.pkl"
DB_FILE       = r"C:\Users\HP\Desktop\Bible\Data\bible.db"
WHISPER_MODEL = "base"                      # "small" or "medium" for better accuracy
TOP_K         = 5                           # verses to retrieve for topic searches
SAMPLE_RATE   = 16000                       # required by Whisper
GROQ_API_KEY  = "your_groq_api_key_here"   # ← paste your free key here
GROQ_MODEL    = "llama-3.1-8b-instant"     # fast + generous free tier
# ─────────────────────────────────────────────────────────────────────────────


# ── BOOK NAME MAPPING ─────────────────────────────────────────────────────────
BOOK_NAMES = {
    # Old Testament
    "genesis": "Genesis", "gen": "Genesis",
    "exodus": "Exodus", "ex": "Exodus", "exo": "Exodus",
    "leviticus": "Leviticus", "lev": "Leviticus",
    "numbers": "Numbers", "num": "Numbers",
    "deuteronomy": "Deuteronomy", "deut": "Deuteronomy", "deu": "Deuteronomy",
    "joshua": "Joshua", "josh": "Joshua",
    "judges": "Judges", "judg": "Judges",
    "ruth": "Ruth",
    "1 samuel": "1 Samuel", "1samuel": "1 Samuel", "first samuel": "1 Samuel",
    "2 samuel": "2 Samuel", "2samuel": "2 Samuel", "second samuel": "2 Samuel",
    "1 kings": "1 Kings", "1kings": "1 Kings", "first kings": "1 Kings",
    "2 kings": "2 Kings", "2kings": "2 Kings", "second kings": "2 Kings",
    "1 chronicles": "1 Chronicles", "first chronicles": "1 Chronicles",
    "2 chronicles": "2 Chronicles", "second chronicles": "2 Chronicles",
    "ezra": "Ezra",
    "nehemiah": "Nehemiah", "neh": "Nehemiah",
    "esther": "Esther", "est": "Esther",
    "job": "Job",
    "psalms": "Psalms", "psalm": "Psalms", "ps": "Psalms",
    "proverbs": "Proverbs", "prov": "Proverbs", "pro": "Proverbs",
    "ecclesiastes": "Ecclesiastes", "eccl": "Ecclesiastes",
    "song of solomon": "Song of Solomon", "song of songs": "Song of Solomon",
    "isaiah": "Isaiah", "isa": "Isaiah",
    "jeremiah": "Jeremiah", "jer": "Jeremiah",
    "lamentations": "Lamentations", "lam": "Lamentations",
    "ezekiel": "Ezekiel", "ezek": "Ezekiel",
    "daniel": "Daniel", "dan": "Daniel",
    "hosea": "Hosea", "hos": "Hosea",
    "joel": "Joel",
    "amos": "Amos",
    "obadiah": "Obadiah", "obad": "Obadiah",
    "jonah": "Jonah", "jon": "Jonah",
    "micah": "Micah", "mic": "Micah",
    "nahum": "Nahum", "nah": "Nahum",
    "habakkuk": "Habakkuk", "hab": "Habakkuk",
    "zephaniah": "Zephaniah", "zeph": "Zephaniah",
    "haggai": "Haggai", "hag": "Haggai",
    "zechariah": "Zechariah", "zech": "Zechariah",
    "malachi": "Malachi", "mal": "Malachi",
    # New Testament
    "matthew": "Matthew", "matt": "Matthew", "mat": "Matthew",
    "mark": "Mark",
    "luke": "Luke", "luk": "Luke",
    "john": "John", "joh": "John",
    "acts": "Acts", "act": "Acts",
    "romans": "Romans", "rom": "Romans",
    "1 corinthians": "1 Corinthians", "first corinthians": "1 Corinthians",
    "2 corinthians": "2 Corinthians", "second corinthians": "2 Corinthians",
    "galatians": "Galatians", "gal": "Galatians",
    "ephesians": "Ephesians", "eph": "Ephesians",
    "philippians": "Philippians", "phil": "Philippians",
    "colossians": "Colossians", "col": "Colossians",
    "1 thessalonians": "1 Thessalonians", "first thessalonians": "1 Thessalonians",
    "2 thessalonians": "2 Thessalonians", "second thessalonians": "2 Thessalonians",
    "1 timothy": "1 Timothy", "first timothy": "1 Timothy",
    "2 timothy": "2 Timothy", "second timothy": "2 Timothy",
    "titus": "Titus",
    "philemon": "Philemon", "phlm": "Philemon",
    "hebrews": "Hebrews", "heb": "Hebrews",
    "james": "James", "jam": "James",
    "1 peter": "1 Peter", "first peter": "1 Peter",
    "2 peter": "2 Peter", "second peter": "2 Peter",
    "1 john": "1 John", "first john": "1 John",
    "2 john": "2 John", "second john": "2 John",
    "3 john": "3 John", "third john": "3 John",
    "jude": "Jude",
    "revelation": "Revelation", "rev": "Revelation", "revelations": "Revelation",
}


# ── STEP 1: UNDERSTAND (Groq LLM intent extraction) ──────────────────────────

INTENT_SYSTEM_PROMPT = """
You are a Bible reference parser. Your only job is to analyze a user's query
and return a JSON object — nothing else. No explanation, no preamble.

Determine if the query is:
  A) A DIRECT reference to a specific Bible book/chapter/verse
  B) A TOPIC search (looking for verses about a theme or subject)

Return exactly one of these two JSON formats:

For a direct reference:
{
  "type": "direct",
  "book": "<full book name e.g. John>",
  "chapter": <integer>,
  "verse": <integer or null if only chapter mentioned>
}

For a topic search:
{
  "type": "topic",
  "query": "<clean search phrase extracted from the user's words>"
}

Rules:
- Understand any phrasing: "what does the bible say in John 1:1",
  "open to Romans chapter 8", "turn to first corinthians chapter thirteen verse four",
  "what does the bible say about worry", "verses about strength", etc.
- Number words like "one", "three", "sixteen" count as numbers.
- If you detect a book name + chapter (+ optional verse), always use type "direct".
- For topic searches, distill the core theme into a clean short phrase.
- Return ONLY the JSON object. No markdown, no explanation.
"""


def llm_extract_intent(groq_client: Groq, user_query: str) -> dict:
    """
    Call Groq LLM to understand the user's intent.
    Returns a dict with type='direct' or type='topic'.
    Falls back to topic search if LLM call fails.
    """
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_query}
            ],
            temperature=0,      # deterministic for structured output
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps in ```json ... ```
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()

        return json.loads(raw)

    except Exception as e:
        print(f"⚠️  LLM intent extraction failed ({e}), falling back to topic search.")
        return {"type": "topic", "query": user_query}


# ── STEP 2: RETRIEVE ──────────────────────────────────────────────────────────

def fetch_direct_verse(book: str, chapter: int, verse) -> list:
    """Fetch a specific verse or whole chapter from SQLite."""
    conn   = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if verse:
        cursor.execute(
            "SELECT book, chapter, verse, text FROM verses "
            "WHERE book=? AND chapter=? AND verse=?",
            (book, chapter, verse)
        )
    else:
        cursor.execute(
            "SELECT book, chapter, verse, text FROM verses "
            "WHERE book=? AND chapter=? ORDER BY verse",
            (book, chapter)
        )

    rows = cursor.fetchall()
    conn.close()
    return rows


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix   / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def search_verses(query_text: str, embed_model, bible_data: dict, top_k: int = 5) -> list:
    """Semantic vector search — find closest matching verses."""
    query_vec   = embed_model.encode([query_text], convert_to_numpy=True)[0]
    scores      = cosine_similarity(query_vec, bible_data["embeddings"])
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [{
        "reference": f"{bible_data['books'][idx]} {bible_data['chapters'][idx]}:{bible_data['verses'][idx]}",
        "text":      bible_data["texts"][idx],
        "score":     float(scores[idx])
    } for idx in top_indices]


def retrieve(intent: dict, embed_model, bible_data: dict) -> tuple:
    """
    Route to the right retrieval method based on LLM intent.
    Returns (retrieval_type, results).
    """
    if intent["type"] == "direct":
        book    = intent.get("book", "")
        chapter = intent.get("chapter")
        verse   = intent.get("verse")

        # Normalize book name — LLM returns full name, map via BOOK_NAMES if needed
        book_normalized = BOOK_NAMES.get(book.lower(), book)

        rows = fetch_direct_verse(book_normalized, chapter, verse)
        return "direct", rows

    else:
        query   = intent.get("query", "")
        results = search_verses(query, embed_model, bible_data, TOP_K)
        return "topic", results


# ── STEP 3: GENERATE (Groq LLM response) ─────────────────────────────────────

RESPONSE_SYSTEM_PROMPT = """
You are a knowledgeable and respectful Bible assistant.
You will be given the user's original question and one or more KJV Bible verses.

Your job is to give a clear, grounded, and concise response based ONLY on
the provided verses. Do not invent or paraphrase verses from memory.

Guidelines:
- Always cite the verse reference (e.g. John 3:16) when quoting.
- Keep your response to 3–6 sentences.
- If the user asked for a specific verse, read it and briefly explain it.
- If the user asked a topic question, explain how the retrieved verses address it.
- Be warm, respectful, and faithful to the text.
"""


def llm_generate_response(groq_client: Groq, user_query: str, retrieved_verses: str) -> str:
    """Call Groq LLM to generate a response grounded in the retrieved verses."""
    try:
        user_message = (
            f"User's question: {user_query}\n\n"
            f"Retrieved Bible verses:\n{retrieved_verses}"
        )
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️  Could not generate response ({e}). See retrieved verses above."


def format_verses_for_llm(retrieval_type: str, results: list) -> str:
    """Format retrieved verses into a clean string for the LLM prompt."""
    if not results:
        return "No verses found."

    if retrieval_type == "direct":
        return "\n".join(f"{r[0]} {r[1]}:{r[2]} — {r[3]}" for r in results)
    else:
        return "\n".join(f"{r['reference']} — {r['text']}" for r in results)


# ── FULL RAG PIPELINE ─────────────────────────────────────────────────────────

def process_query(user_query: str, groq_client: Groq, embed_model, bible_data: dict):
    """
    Full RAG pipeline:
      1. LLM extracts intent  (direct reference or topic search)
      2. Retrieve verse(s)    (SQLite lookup or vector search)
      3. LLM generates a grounded natural language response
    """
    print(f"\n📝 Query: \"{user_query}\"")
    print("─" * 60)

    # 1. UNDERSTAND
    print("🧠 Understanding intent...")
    intent = llm_extract_intent(groq_client, user_query)

    if intent["type"] == "direct":
        label = f"{intent.get('book')} {intent.get('chapter')}"
        if intent.get("verse"):
            label += f":{intent.get('verse')}"
        print(f"✅ Direct reference → {label}")
    else:
        print(f"🔍 Topic search → \"{intent.get('query')}\"")

    # 2. RETRIEVE
    print("📖 Retrieving verses...")
    retrieval_type, results = retrieve(intent, embed_model, bible_data)

    if not results:
        print("⚠️  No verses found in database.")
        return

    print("\n" + "=" * 60)
    print("📖 RETRIEVED VERSES:")
    print("=" * 60)
    if retrieval_type == "direct":
        for row in results:
            print(f"  {row[0]} {row[1]}:{row[2]}")
            print(f"  \"{row[3]}\"")
            print()
    else:
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['reference']}  (score: {r['score']:.3f})")
            print(f"     {r['text']}")
            print()

    # 3. GENERATE
    print("=" * 60)
    print("💬 RESPONSE:")
    print("=" * 60)
    verses_text = format_verses_for_llm(retrieval_type, results)
    response    = llm_generate_response(groq_client, user_query, verses_text)
    print(f"\n{response}\n")
    print("=" * 60)


# ── AUDIO ─────────────────────────────────────────────────────────────────────

def record_audio(duration_seconds: int = 15) -> str:
    print(f"\n🎙️  Recording for {duration_seconds} seconds... Speak now!")
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("✅ Recording done.")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("🚀 Loading Bible RAG System...\n")

    if GROQ_API_KEY == "your_groq_api_key_here":
        print("❌ Please set your GROQ_API_KEY in the CONFIG section at the top of this file.")
        print("   Get a free key at: https://console.groq.com")
        return

    print("📦 Loading Bible vectors...")
    with open(VECTORS_FILE, "rb") as f:
        bible_data = pickle.load(f)
    print(f"✅ {len(bible_data['texts'])} verses loaded\n")

    print("🤖 Loading Whisper model...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    print("✅ Whisper ready\n")

    print("🤖 Loading sentence transformer...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Sentence transformer ready\n")

    print("🔗 Connecting to Groq...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"✅ Groq ready ({GROQ_MODEL})\n")

    print("🎉 RAG System ready!\n")

    while True:
        print("\nOptions:")
        print("  [ENTER]     → Record 15 seconds of audio")
        print("  [t + ENTER] → Type a query manually")
        print("  [q + ENTER] → Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            print("👋 Goodbye!")
            break

        elif choice == "t":
            query = input("Type your query: ").strip()
            if query:
                process_query(query, groq_client, embed_model, bible_data)

        else:
            audio_file = record_audio(duration_seconds=15)
            print("📝 Transcribing...")
            result      = whisper_model.transcribe(audio_file, language="en")
            transcribed = result["text"].strip()
            os.unlink(audio_file)

            if transcribed:
                process_query(transcribed, groq_client, embed_model, bible_data)
            else:
                print("⚠️  No speech detected. Please try again.")


if __name__ == "__main__":
    main()
