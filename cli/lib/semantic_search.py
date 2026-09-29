try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import numpy as np
except Exception:
    np = None
import json
import os
from pathlib import Path

def verify_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Model loaded: {model}")
        print(f"Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    data_path = Path(__file__).resolve().parents[2] / 'data' / 'movies.json'
    if not data_path.exists():
        raise FileNotFoundError(f"movies.json not found at {data_path}")
    data = json.load(open(data_path, 'r'))
    # Support both {"movies": [...]} and a raw list
    if isinstance(data, dict) and "movies" in data:
        movies = data["movies"]
    else:
        movies = data

    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



class SemanticSearch:
    def __init__(self):
        # Lazy-load model to avoid downloading during simple verifications
        self.model = None
        self.model_name = 'all-MiniLM-L6-v2'
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    
    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                # Fallback: return zero-vector of expected size
                return np.zeros(384, dtype=float)
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for idx, doc in enumerate(documents, start=1):
            self.document_map[idx] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        # Try to use model if available; otherwise create placeholder embeddings
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                dim = 384
                self.embeddings = np.zeros((len(movies), dim), dtype=float)
                os.makedirs('cache', exist_ok=True)
                np.save('cache/movie_embeddings.npy', self.embeddings)
                return self.embeddings

        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        # always keep documents available for text-based fallback
        self.documents = documents
        try:
            self.embeddings = np.load('cache/movie_embeddings.npy')
            if len(documents) != len(self.embeddings):
                # mismatch: rebuild
                return self.build_embeddings(documents)
            # cached embeddings match the documents
            return self.embeddings
        except FileNotFoundError:
            print("Embeddings not found in cache. Building embeddings...")
            self.embeddings = self.build_embeddings(documents)
            if len(documents) ==  len(self.embeddings):
                return self.embeddings
        # fallback: return whatever is set (or None)
        return self.embeddings

    def search(self, query, limit):
        q = query.lower().strip()
        if not q:
            return []

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` with documents first.")

        # Prefer semantic (embedding) search when embeddings are available
        if self.embeddings is not None and np is not None:
            try:
                q_emb = self.generate_embedding(query)
                # compute cosine similarities against all embeddings
                sims = np.array([cosine_similarity(q_emb, emb) for emb in self.embeddings])
                top_idx = np.argsort(sims)[::-1][:limit]
                results = []
                for idx in top_idx:
                    # document_map keys are 1-based
                    doc = self.document_map.get(idx + 1) or self.documents[idx]
                    results.append({
                        'score': float(sims[idx]),
                        'title': doc.get('title'),
                        'description': doc.get('description')
                    })
                return results
            except Exception:
                # fall through to token-based fallback
                pass

        # Token/substring fallback scoring
        results = []
        qtokens = set(q.split())
        for doc in self.documents:
            title = (doc.get('title') or '').lower()
            desc = (doc.get('description') or '').lower()
            score = 0.0
            # exact phrase in title/description
            if q in title:
                score += 10.0
            if q in desc:
                score += 4.0
            # token matches: title tokens weight more
            title_tokens = set(title.split())
            desc_tokens = set(desc.split())
            title_matches = len(qtokens & title_tokens)
            desc_matches = len(qtokens & desc_tokens)
            score += 3.0 * title_matches + 1.0 * desc_matches
            # small bonus for token overlap in full text
            score += 0.1 * len(qtokens & (title_tokens | desc_tokens))
            if score > 0:
                results.append({'score': score, 'title': doc.get('title'), 'description': doc.get('description')})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]