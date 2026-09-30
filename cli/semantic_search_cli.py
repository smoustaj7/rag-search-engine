import argparse
import json
from pathlib import Path
from lib.semantic_search import verify_embeddings, embed_query_text, SemanticSearch
import re

if __package__:
    from .lib.semantic_search import embed_text, verify_model
else:
    from lib.semantic_search import embed_text, verify_model


def semantic_chunk(text: str, chunk_size: int = 4, overlap: int = 0) -> list[str]:
    words = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start += chunk_size - overlap
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed a text string")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed a query string")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    verify_emb = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text string")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a text string semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            # Load movies from the repo data directory
            data_path = Path(__file__).resolve().parents[1] / 'data' / 'movies.json'
            if not data_path.exists():
                data_path = Path(__file__).resolve().parents[2] / 'data' / 'movies.json'
            if not data_path.exists():
                raise FileNotFoundError(f"movies.json not found at {data_path}")
            data = json.load(open(data_path, 'r'))
            movies = data.get('movies') if isinstance(data, dict) and 'movies' in data else data
            ss = SemanticSearch()
            ss.load_or_create_embeddings(movies)
            results = ss.search(args.query, limit=args.limit)
            for idx, doc in enumerate(results, start=1):
                print(f"{idx}. {doc['title']}")
        case "verify_embeddings":
            verify_embeddings()
        case "chunk":
            words = args.text.split()
            if args.overlap > 0:
                chunks = []
                start = 0
                while start < len(words):
                    end = min(start + args.chunk_size, len(words))
                    chunk = words[start:end]
                    chunks.append(' '.join(chunk))
                    start += args.chunk_size - args.overlap
            else:
                chunks = [words[i:i + args.chunk_size] for i in range(0, len(words), args.chunk_size)]
                chunks = [' '.join(chunk) for chunk in chunks]
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, chunk_size=args.max_chunk_size, overlap=args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()