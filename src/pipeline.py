from src.retrieval.retriever import Retriever
from src.llm.generator import Generator


def run_query(query, retriever, generator, strategy="zero_shot", top_k=3):
    """Run the full RAG pipeline: retrieve → generate."""
    # Step 1: Retrieve relevant chunks
    chunks = retriever.retrieve(query, top_k=top_k)

    # Step 2: Generate answer
    result = generator.generate(query, chunks, prompt_strategy=strategy)

    return result, chunks


if __name__ == "__main__":
    # Load once, reuse for all queries
    retriever = Retriever()
    generator = Generator()

    query = "My wheat leaves are turning yellow. What could be the problem and what should I do?"

    print(f"QUERY: {query}\n")

    for strategy in ["zero_shot", "few_shot", "chain_of_thought"]:
        result, chunks = run_query(query, retriever, generator, strategy=strategy)

        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*60}")
        print(f"SOURCES: {result['sources']}")
        print(f"\nANSWER:\n{result['answer']}")

# Try a query our knowledge base can answer
    print(f"\n\n{'#'*60}")
    query2 = "How much nitrogen should I apply to my wheat crop and when?"
    print(f"QUERY: {query2}\n")

    for strategy in ["zero_shot", "few_shot", "chain_of_thought"]:
        result, chunks = run_query(query2, retriever, generator, strategy=strategy)

        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*60}")
        print(f"\nANSWER:\n{result['answer']}")