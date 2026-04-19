import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import chromadb
from sentence_transformers import SentenceTransformer


def load_documents(data_dir):
    """Load all .txt files from the data directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks, db_dir="data/chromadb"):
    """Embed chunks and store them in ChromaDB."""
    # Load the embedding model (same one from our learning exercise)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create ChromaDB client (stores data on disk)
    client = chromadb.PersistentClient(path=db_dir)

    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection("agri_knowledge")
    except Exception:
        pass

    # Create a new collection
    collection = client.create_collection(
        name="agri_knowledge",
        metadata={"description": "Agricultural knowledge base",
                  "hnsw:space": "cosine"}
    )

    # Process each chunk
    texts = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        texts.append(chunk.page_content)
        metadatas.append({
            "source": os.path.basename(chunk.metadata.get("source", "unknown")),
            "chunk_index": i
        })
        ids.append(f"chunk_{i}")

    # Create embeddings for all chunks at once
    embeddings = model.encode(texts)
    embeddings_list = [emb.tolist() for emb in embeddings]

    # Add everything to ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings_list,
        metadatas=metadatas
    )

    print(f"Stored {len(texts)} chunks in ChromaDB at '{db_dir}'")
    return collection

if __name__ == "__main__":
    # Load
    documents = load_documents("data/raw")

    # Chunk
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

    # Print first 2 chunks to see what they look like
    print("\n--- Sample Chunks ---")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i}:")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")
        print(f"Length: {len(chunk.page_content)} characters")

    # Embed and store
    create_vector_store(chunks)

    print("\nIngestion complete!")