from sentence_transformers import SentenceTransformer
import chromadb


class Retriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(self, db_dir="data/chromadb", collection_name="agri_knowledge"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=db_dir)
        self.collection = client.get_collection(collection_name)
        print(f"Retriever ready. Collection has {self.collection.count()} chunks.")

    def retrieve(self, query, top_k=3):
        """Find the top_k most relevant chunks for a query."""
        # Embed the query (same model that embedded the documents)
        query_embedding = self.model.encode(query).tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Package results cleanly
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i]  # convert distance to similarity
            })

        return retrieved


if __name__ == "__main__":
    retriever = Retriever()

    # Test queries - try different types
    test_queries = [
        "How much nitrogen does wheat need?",
        "What causes rice blast disease?",
        "How do I improve soil organic matter?",
        "What is the best way to control weeds?",
        "How should I water my wheat crop?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=2)

        for j, r in enumerate(results):
            print(f"\n  Result {j+1}:")
            print(f"  Source: {r['source']}")
            print(f"  Similarity: {r['similarity']:.4f}")
            print(f"  Text: {r['text'][:150]}...")