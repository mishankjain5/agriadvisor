import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class Explainer:
    """Provides explainability for RAG pipeline outputs."""

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = "gemini-3.1-flash-lite-preview"

    def source_attribution(self, chunks):
        """Show which sources contributed to the answer and how relevant they were."""
        attribution = []
        for i, chunk in enumerate(chunks):
            attribution.append({
                "rank": i + 1,
                "source": chunk["source"],
                "similarity": chunk["similarity"],
                "preview": chunk["text"][:200],
                "confidence": self._similarity_to_confidence(chunk["similarity"])
            })
        return attribution

    def _similarity_to_confidence(self, similarity):
        """Convert similarity score to a human-readable confidence level."""
        if similarity >= 0.7:
            return "HIGH - Strong match to query"
        elif similarity >= 0.5:
            return "MEDIUM - Partial match to query"
        elif similarity >= 0.3:
            return "LOW - Weak match to query"
        else:
            return "VERY LOW - Likely irrelevant"

    def check_faithfulness(self, query, context, answer):
        """Use LLM to verify if the answer is grounded in the context."""
        prompt = (
            "You are a fact-checking assistant. Your job is to verify whether "
            "an answer is faithfully grounded in the provided context.\n\n"
            "Rules:\n"
            "- A claim is SUPPORTED if the context contains information that directly supports it\n"
            "- A claim is NOT SUPPORTED if the context does not contain relevant information\n"
            "- A claim is CONTRADICTED if the context says something different\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer to verify:\n{answer}\n\n"
            "Respond in this exact format:\n"
            "VERDICT: [FAITHFUL / PARTIALLY FAITHFUL / NOT FAITHFUL]\n"
            "SCORE: [0.0 to 1.0]\n"
            "REASONING: [Explain which claims are supported and which are not]"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text

    def compare_strategies(self, query, retriever, generator):
        """Run all three prompt strategies and compare results."""
        chunks = retriever.retrieve(query, top_k=3)

        comparison = {}
        for strategy in ["zero_shot", "few_shot", "chain_of_thought"]:
            result = generator.generate(query, chunks, prompt_strategy=strategy)
            comparison[strategy] = {
                "answer": result["answer"],
                "prompt_length": len(result["prompt_used"]),
                "answer_length": len(result["answer"])
            }

        return comparison, chunks