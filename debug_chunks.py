import chromadb

client = chromadb.PersistentClient(path="data/chromadb")
collection = client.get_collection("agri_knowledge")

# See ALL chunks from rice_cultivation.txt
all_data = collection.get(
    where={"source": "rice_cultivation.txt"},
    include=["documents", "metadatas"]
)

print(f"Total chunks from rice_cultivation.txt: {len(all_data['documents'])}")
for i, doc in enumerate(all_data["documents"]):
    print(f"\n--- Rice Chunk {i} ---")
    print(f"First 200 chars: {doc[:200]}")
    print(f"Total length: {len(doc)} chars")
    has_blast = "blast" in doc.lower()
    print(f"Contains 'blast': {has_blast}")

# See all unique sources in the database
all_data = collection.get(include=["metadatas"])
sources = set(m["source"] for m in all_data["metadatas"])
print(f"\nAll sources in database: {sources}")