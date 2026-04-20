from src.retrieval.retriever import Retriever
from src.llm.generator import Generator
from src.explainability.explainer import Explainer


if __name__ == "__main__":
    retriever = Retriever()
    generator = Generator()
    explainer = Explainer()

    query = "How much nitrogen should I apply to my wheat crop and when?"

    # Step 1: Retrieve
    chunks = retriever.retrieve(query, top_k=3)

    # Step 2: Source Attribution
    print("=" * 60)
    print("SOURCE ATTRIBUTION")
    print("=" * 60)
    attribution = explainer.source_attribution(chunks)
    for attr in attribution:
        print(f"\n  Rank {attr['rank']}:")
        print(f"  File: {attr['source']}")
        print(f"  Similarity: {attr['similarity']:.4f}")
        print(f"  Confidence: {attr['confidence']}")
        print(f"  Preview: {attr['preview'][:100]}...")

    # Step 3: Generate answer
    result = generator.generate(query, chunks, prompt_strategy="zero_shot")
    print(f"\n{'='*60}")
    print("GENERATED ANSWER")
    print("=" * 60)
    print(result["answer"])

    # Step 4: Faithfulness Check
    context = "\n".join([c["text"] for c in chunks])
    print(f"\n{'='*60}")
    print("FAITHFULNESS CHECK")
    print("=" * 60)
    faithfulness = explainer.check_faithfulness(query, context, result["answer"])
    print(faithfulness)

    # Step 5: Strategy Comparison
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print("=" * 60)
    comparison, _ = explainer.compare_strategies(query, retriever, generator)
    for strategy, data in comparison.items():
        print(f"\n  {strategy}:")
        print(f"  Prompt length: {data['prompt_length']} chars")
        print(f"  Answer length: {data['answer_length']} chars")
        print(f"  Answer: {data['answer'][:150]}...")