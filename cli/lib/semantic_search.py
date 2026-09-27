from sentence_transformers import SentenceTransformer



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

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        return self.model.encode([text])[0]
    