import argparse
import json
import math
import os
import pickle
import string
from collections.abc import Iterable
from collections import Counter
import math

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"
def _resource_path(*parts: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *parts)


def load_movies():
    with open(_resource_path("data", "movies.json"), "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize_text(text: str):
    if text is None:
        return []

    punctuation_table = str.maketrans("", "", string.punctuation)
    stopwords_path = _resource_path("data", "stopwords.txt")

    with open(stopwords_path, "r", encoding="utf-8") as stop_file:
        stop_text = stop_file.read()
        stop_list = stop_text.splitlines()
        stop_words = {
            sw.translate(punctuation_table).lower()
            for sw in stop_list
            if sw.strip()
        }

    tokens = text.translate(punctuation_table).split()
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token and token not in stop_words]
    return [stemmer.stem(token) for token in tokens]


def tokenize(text, data):
    query_tokens = tokenize_text(text)
    results = []
    for movie in data["movies"]:
        title_tokens = tokenize_text(movie["title"])
        title_text = " ".join(title_tokens)
        if query_tokens and any(q in title_text for q in query_tokens):
            results.append(movie)
    return results


def single_term_tokenize(text):
    tokens = tokenize_text(text)
    if len(tokens) > 1:
        raise ValueError("Input text must contain only a single term.")
    return tokens[0] if tokens else None

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequency = Counter()
        self.doc_length = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        self.doc_length[doc_id] = len(tokenize_text(text))
        for token in set(tokenize_text(text)):
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequency[token] += 1
    
    def get_tf(self, doc_id, term):
        doc_id = int(doc_id)
        if doc_id not in self.docmap:
            return 0
        text = f"{self.docmap[doc_id]['title']} {self.docmap[doc_id]['description']}"
        tokens = tokenize_text(text)
        return tokens.count(term)

    def get_documents(self, term):
        if term is None:
            return []

        normalized = tokenize_text(term)
        if normalized:
            term = normalized[0]
        return sorted(self.index.get(term, set()))

    def build(self):
        movies_data = load_movies()
        for movie in movies_data.get("movies", []):
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
        return self

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as index_file:
            pickle.dump(self.index, index_file)

        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open("cache/term_frequency.pkl", "wb") as term_frequency_file:
            pickle.dump(self.term_frequency, term_frequency_file)

        with open(self.doc_lengths_path, "wb") as doc_lengths_file:
            pickle.dump(self.doc_length, doc_lengths_file)

    def load(self):
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        term_frequency_path = "cache/term_frequency.pkl"
        doc_lengths_path = self.doc_lengths_path

        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError("Index files not found. Run the build command first.")

        with open(index_path, "rb") as index_file:
            self.index = pickle.load(index_file)

        with open(docmap_path, "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)

        if os.path.exists(term_frequency_path):
            with open(term_frequency_path, "rb") as term_frequency_file:
                self.term_frequency = pickle.load(term_frequency_file)
        else:
            self.term_frequency = Counter()

        if os.path.exists(doc_lengths_path):
            with open(doc_lengths_path, "rb") as doc_lengths_file:
                self.doc_length = pickle.load(doc_lengths_file)
        else:
            self.doc_length = {}

    def tf(self, doc_id, term):
        tokenized_term = single_term_tokenize(term)
        if tokenized_term is None:
            return 0
        return self.get_tf(doc_id, tokenized_term)
    
    def idf(self, term):
        doc_map = len(self.docmap)
        if doc_map == 0:
            return 0.0
        tokenized_term = single_term_tokenize(term)
        if tokenized_term is None:
            return 0.0
        doc_count = len(self.index.get(tokenized_term, set()))
        if doc_count == 0:
            return 0.0
        idf = math.log((doc_map + 1) / (doc_count + 1))
        print(f"Inverse document frequency of '{tokenized_term}': {idf:.2f}")
        return idf
    
    def tf_idf(self, doc_id, term):
        tf_value = self.tf(doc_id, term)
        idf_value = self.idf(term)
        tf_idf_value = tf_value * idf_value
        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf_value:.2f}")
        return tf_idf_value
    
    def get_bm25_idf(self, term: str) -> float:
        doc_count = len(self.index.get(term, set()))
        if doc_count == 0:
            return 0.0
        total_docs = len(self.docmap)
        idf = math.log((total_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)
        return idf
    
    def bm25_idf_command(self, term: str):
        tokenized_term = single_term_tokenize(term)
        if tokenized_term is None:
            return 0.0
        idf_value = self.get_bm25_idf(tokenized_term)
        return idf_value
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf_value = self.get_tf(doc_id, term)
        if tf_value == 0:
            return 0.0
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length == 0:
            return 0.0
        bm25_tf = (tf_value * (k1 + 1)) / (tf_value + k1 * (1 - b + b * self.doc_length.get(doc_id, 0) / avg_doc_length))
        return bm25_tf
    
    def bm25_tf_command(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tokenized_term = single_term_tokenize(term)
        if tokenized_term is None:
            return 0.0
        bm25_tf_value = self.get_bm25_tf(doc_id, tokenized_term, k1, b)
        return bm25_tf_value
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_length:
            return 0.0
        total_length = sum(self.doc_length.values())
        avg_length = total_length / len(self.doc_length)
        return avg_length
    
    def bm25(self, doc_id, term):
        tokenized_query = single_term_tokenize(term)
        if tokenized_query is None:
            return 0.0
        scores = {}
        for token in [tokenized_query]:
            idf = self.get_bm25_idf(token)
            tf = self.get_bm25_tf(doc_id, token)
            scores[doc_id] = idf * tf
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[0][1] if sorted_scores else 0.0


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()

    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to search for")
    tf_idf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tf_idf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to search for")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
        )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
        )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter"
        )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
        )
    bm25search_parser.add_argument("query", type=str, help="Search query")


    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)

            query_tokens = tokenize_text(args.query)
            results = []
            seen = set()
            for term in query_tokens:
                for doc_id in index.get_documents(term):
                    if doc_id not in seen:
                        seen.add(doc_id)
                        results.append(doc_id)
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break

            print(f"Searching for: {args.query}")
            for i, doc_id in enumerate(results[:5], start=1):
                title = index.docmap.get(doc_id, {}).get("title", "Unknown")
                print(f"{i}. ID {doc_id}: {title}")
        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)
            term_frequency = index.tf(args.doc_id, args.term)
            print(f"Term frequency for term '{args.term}' in document '{args.doc_id}': {term_frequency}")
        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)
            idf_value = index.idf(args.term)
            print(f"Inverse document frequency for term '{args.term}': {idf_value}")
        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)
            tf_idf_value = index.tf_idf(args.doc_id, args.term)
        case "build":
            build_command()
        case "bm25idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)
            bm25_idf_value = index.bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf_value:.2f}")
        case "bm25tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)
            bm25_tf_value = index.bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf_value:.2f}")
        case "bm25search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index not found. Run the build command first.")
                exit(1)
            query_tokens = tokenize_text(args.query)
            scores = {}
            for term in query_tokens:
                for doc_id in index.get_documents(term):
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += index.bm25(doc_id, term)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for i, (doc_id, score) in enumerate(sorted_scores[:5], start=1):
                title = index.docmap.get(doc_id, {}).get("title", "Unknown")
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
