import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class Generator:
    """Generates answers using an LLM with retrieved context."""

    def __init__(self, model_name="gemini-3.1-flash-lite-preview"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name

    def generate(self, query, retrieved_chunks, prompt_strategy="zero_shot"):
        """Generate an answer using the query and retrieved context."""

        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Build prompt based on strategy
        prompt = self._build_prompt(query, context, prompt_strategy)

        # Call the LLM
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return {
            "answer": response.text,
            "prompt_used": prompt,
            "strategy": prompt_strategy,
            "sources": [chunk["source"] for chunk in retrieved_chunks]
        }

    def _build_context(self, chunks):
        """Format retrieved chunks into a context string."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source: {chunk['source']} | Relevance: {chunk['similarity']:.2f}]\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query, context, strategy):
        """Build prompt based on the chosen strategy."""

        if strategy == "zero_shot":
            return (
                f"You are an agricultural advisor. Answer the farmer's question "
                f"based ONLY on the provided context. If the context does not "
                f"contain enough information, say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Farmer's Question: {query}\n\n"
                f"Answer:"
            )

        elif strategy == "few_shot":
            return (
                f"You are an agricultural advisor. Answer the farmer's question "
                f"based ONLY on the provided context. If the context does not "
                f"contain enough information, say so clearly.\n\n"
                f"Here are examples of good answers:\n\n"
                f"Example Question: What type of soil is best for growing crops?\n"
                f"Example Answer: Most crops grow best in well-drained loamy soils "
                f"with a pH between 6.0 and 7.0. Loamy soils provide a good balance "
                f"of drainage and water retention. However, specific requirements "
                f"vary by crop type.\n\n"
                f"Example Question: When should I apply fertilizer?\n"
                f"Example Answer: Fertilizer timing depends on the crop and nutrient. "
                f"Generally, nitrogen should be split into multiple applications "
                f"aligned with key growth stages rather than applied all at once, "
                f"as this improves uptake efficiency and reduces losses.\n\n"
                f"Now answer the following:\n\n"
                f"Context:\n{context}\n\n"
                f"Farmer's Question: {query}\n\n"
                f"Answer:"
            )

        elif strategy == "chain_of_thought":
            return (
                f"You are an agricultural advisor. Answer the farmer's question "
                f"based ONLY on the provided context. If the context does not "
                f"contain enough information, say so clearly.\n\n"
                f"Think step by step:\n"
                f"1. Identify which parts of the context are relevant\n"
                f"2. Extract the key facts\n"
                f"3. Formulate a clear answer\n\n"
                f"Context:\n{context}\n\n"
                f"Farmer's Question: {query}\n\n"
                f"Step-by-step reasoning and answer:"
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")