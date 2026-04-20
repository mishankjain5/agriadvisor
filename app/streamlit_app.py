import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time
from src.retrieval.retriever import Retriever
from src.llm.generator import Generator
from src.explainability.explainer import Explainer


@st.cache_resource
def load_components():
    """Load models once and cache them."""
    retriever = Retriever()
    generator = Generator()
    explainer = Explainer()
    return retriever, generator, explainer


def main():
    st.set_page_config(
        page_title="AgriAdvisor",
        page_icon="🌾",
        layout="wide"
    )

    st.title("🌾 AgriAdvisor")
    st.markdown("*Explainable AI-powered Agricultural Advisory System*")

    # Load components
    retriever, generator, explainer = load_components()

    # Sidebar - settings
    st.sidebar.header("Settings")
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Farmer View", "Researcher View"]
    )
    strategy = st.sidebar.selectbox(
        "Prompt Strategy",
        ["zero_shot", "few_shot", "chain_of_thought"],
        format_func=lambda x: {
            "zero_shot": "Zero-Shot (Direct)",
            "few_shot": "Few-Shot (With Examples)",
            "chain_of_thought": "Chain-of-Thought (Step-by-Step)"
        }[x]
    )
    top_k = st.sidebar.slider("Number of sources to retrieve", 1, 5, 3)

    # Main input
    query = st.text_input(
        "Ask your agricultural question:",
        placeholder="e.g., How much nitrogen should I apply to wheat?"
    )

    if query:
        with st.spinner("Searching knowledge base..."):
            chunks = retriever.retrieve(query, top_k=top_k)

        with st.spinner("Generating answer..."):
            result = generator.generate(query, chunks, prompt_strategy=strategy)

        # ---- FARMER VIEW ----
        if view_mode == "Farmer View":
            st.header("Answer")
            st.write(result["answer"])

            # Simple source display
            st.subheader("Sources")
            for i, chunk in enumerate(chunks):
                confidence = explainer._similarity_to_confidence(chunk["similarity"])
                if chunk["similarity"] >= 0.5:
                    st.success(f"📄 {chunk['source']} — {confidence}")
                elif chunk["similarity"] >= 0.3:
                    st.warning(f"📄 {chunk['source']} — {confidence}")
                else:
                    st.error(f"📄 {chunk['source']} — {confidence}")

        # ---- RESEARCHER VIEW ----
        else:
            col1, col2 = st.columns([3, 2])

            with col1:
                st.header("Generated Answer")
                st.write(result["answer"])

                # Faithfulness check
                st.header("Faithfulness Check")
                with st.spinner("Verifying answer against sources..."):
                    context = "\n".join([c["text"] for c in chunks])
                    try:
                        faithfulness = explainer.check_faithfulness(
                            query, context, result["answer"]
                        )
                        st.code(faithfulness, language=None)
                    except Exception as e:
                        st.error(f"Faithfulness check failed: {str(e)[:100]}")

            with col2:
                st.header("Source Attribution")
                attribution = explainer.source_attribution(chunks)

                for attr in attribution:
                    with st.expander(
                        f"Rank {attr['rank']}: {attr['source']} "
                        f"(Similarity: {attr['similarity']:.4f})"
                    ):
                        st.markdown(f"**Confidence:** {attr['confidence']}")
                        st.markdown(f"**Content Preview:**")
                        st.text(attr["preview"])

                st.header("Retrieval Details")
                st.json({
                    "query": query,
                    "strategy": strategy,
                    "top_k": top_k,
                    "sources_retrieved": [c["source"] for c in chunks],
                    "similarity_scores": [
                        round(c["similarity"], 4) for c in chunks
                    ]
                })

            # Strategy Comparison
            st.header("Prompt Strategy Comparison")
            if st.button("Compare All Strategies"):
                with st.spinner("Running all three strategies..."):
                    try:
                        comparison, _ = explainer.compare_strategies(
                            query, retriever, generator
                        )

                        for strat, data in comparison.items():
                            label = {
                                "zero_shot": "Zero-Shot",
                                "few_shot": "Few-Shot",
                                "chain_of_thought": "Chain-of-Thought"
                            }[strat]

                            with st.expander(
                                f"{label} — {data['answer_length']} chars"
                            ):
                                st.write(data["answer"])
                                st.caption(
                                    f"Prompt: {data['prompt_length']} chars | "
                                    f"Answer: {data['answer_length']} chars"
                                )
                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)[:100]}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**AgriAdvisor** v1.0\n\n"
        "Built with LangChain, ChromaDB,\n"
        "HuggingFace, and Gemini API"
    )
    st.sidebar.markdown(
        "MSc Data Science\n"
        "University of Potsdam"
    )


if __name__ == "__main__":
    main()