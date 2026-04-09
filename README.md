# 🌐 Offline Internet Capsule (OIC)

> A portable, self-contained layer of the internet that works even when the internet doesn't.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd capsulenet
pip install -r requirements.txt
```

### 2. Ingest Knowledge Base
```bash
python -m core.ingest
```
This generates `knowledge.db` (SQLite) and `knowledge.faiss` (vector index).

### 3. Run the Server
```bash
uvicorn core.main:app --reload
```

### 4. Open the App
Navigate to **http://localhost:8000** in your browser.

## 📁 Project Structure

```
capsulenet/
├── core/
│   ├── main.py          # FastAPI app + endpoints
│   ├── search.py        # Hybrid BM25 + FAISS search engine
│   ├── formatter.py     # Structured response templates
│   ├── ingest.py        # Data ingestion pipeline
│   ├── models.py        # Pydantic models
│   └── data/
│       ├── knowledge.json   # Curated knowledge base
│       ├── knowledge.db     # SQLite database (generated)
│       └── knowledge.faiss  # FAISS index (generated)
├── frontend/
│   └── index.html       # Single-file web app
├── requirements.txt
└── README.md
```

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + document count |
| `/ask` | POST | Ask a question (hybrid search) |
| `/categories` | GET | List all categories |
| `/emergency` | GET | Emergency quick-access cards |

## 🧠 How It Works

1. **Hybrid Search**: Combines BM25 keyword matching (40%) with FAISS semantic similarity (60%)
2. **Category Filtering**: Medical, Survival, Navigation, Education
3. **Structured Responses**: Templates with warnings, steps, confidence scores
4. **Emergency Cards**: One-tap access to CPR, burns, snake bites, and more

## 📊 Knowledge Categories

- 🏥 **Medical** — CPR, burns, snake bites, wound care, dehydration, choking
- 🏕️ **Survival** — Water purification, fire starting, shelter, signaling, disaster prep
- 🧭 **Navigation** — Compass, stars, maps, GPS alternatives, river crossing
- 📚 **Education** — Science, math, geography, literacy, nutrition, hygiene

## 📄 License

MIT License — Free to use, modify, and deploy.
