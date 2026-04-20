import json
import time
import os
from datetime import datetime
from src.retrieval.retriever import Retriever
from src.llm.generator import Generator
from src.explainability.explainer import Explainer
from src.evaluation.benchmark import BENCHMARK_QUESTIONS


class Evaluator:
    """Evaluates the RAG pipeline against benchmark questions."""

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.explainer = Explainer()
        self.results_file = "docs/results/evaluation_results.json"

    def evaluate_retrieval(self, query, expected_source, top_k=3):
        """Check if the correct source document was retrieved."""
        chunks = self.retriever.retrieve(query, top_k=top_k)

        sources_found = [c["source"] for c in chunks]
        top_source = sources_found[0] if sources_found else "none"
        top_similarity = chunks[0]["similarity"] if chunks else 0.0

        return {
            "correct_at_1": top_source == expected_source,
            "correct_in_top_k": expected_source in sources_found,
            "top_source": top_source,
            "top_similarity": top_similarity,
            "all_sources": sources_found,
            "chunks": chunks
        }

    def evaluate_answer(self, answer, expected_keywords):
        """Check if the answer contains expected keywords."""
        if not expected_keywords:
            return {"keyword_recall": None, "found": [], "missing": []}

        answer_lower = answer.lower()
        found = []
        missing = []

        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                found.append(kw)
            else:
                missing.append(kw)

        recall = len(found) / len(expected_keywords) if expected_keywords else 0
        return {
            "keyword_recall": recall,
            "found": found,
            "missing": missing
        }

    def run_full_evaluation(self):
        """Run evaluation on all benchmark questions across all strategies."""
        results = []
        total = len(BENCHMARK_QUESTIONS)

        # Load existing progress if any
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                existing = json.load(f)
                results = existing.get("results", [])
                completed_queries = {r["query"] for r in results}
                print(f"Resuming: {len(results)} results already saved.")
        else:
            completed_queries = set()

        for i, q in enumerate(BENCHMARK_QUESTIONS):
            if q["query"] in completed_queries:
                print(f"[{i+1}/{total}] Skipping (already done): {q['query'][:50]}...")
                continue

            print(f"\n[{i+1}/{total}] Evaluating: {q['query'][:50]}...")

            # Evaluate retrieval
            retrieval = self.evaluate_retrieval(q["query"], q["expected_source"])

            question_result = {
                "query": q["query"],
                "expected_source": q["expected_source"],
                "difficulty": q["difficulty"],
                "retrieval": {
                    "correct_at_1": retrieval["correct_at_1"],
                    "correct_in_top_k": retrieval["correct_in_top_k"],
                    "top_source": retrieval["top_source"],
                    "top_similarity": retrieval["top_similarity"]
                },
                "strategies": {}
            }

            # Evaluate each prompt strategy
            for strategy in ["zero_shot", "few_shot", "chain_of_thought"]:
                try:
                    result = self.generator.generate(
                        q["query"], retrieval["chunks"],
                        prompt_strategy=strategy
                    )

                    answer_eval = self.evaluate_answer(
                        result["answer"], q["expected_keywords"]
                    )

                    question_result["strategies"][strategy] = {
                        "answer": result["answer"],
                        "keyword_recall": answer_eval["keyword_recall"],
                        "found_keywords": answer_eval["found"],
                        "missing_keywords": answer_eval["missing"],
                        "answer_length": len(result["answer"])
                    }

                    print(f"  {strategy}: keyword_recall={answer_eval['keyword_recall']}")
                    time.sleep(4)  # respect rate limits

                except Exception as e:
                    print(f"  {strategy}: ERROR - {str(e)[:80]}")
                    question_result["strategies"][strategy] = {
                        "answer": f"ERROR: {str(e)[:200]}",
                        "keyword_recall": None,
                        "found_keywords": [],
                        "missing_keywords": q["expected_keywords"],
                        "answer_length": 0
                    }
                    time.sleep(10)

            results.append(question_result)

            # Save progress after each question
            self._save_results(results)

        self._print_summary(results)
        return results

    def _save_results(self, results):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        output = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(BENCHMARK_QUESTIONS),
            "completed": len(results),
            "results": results
        }

        with open(self.results_file, "w") as f:
            json.dump(output, f, indent=2)

    def _print_summary(self, results):
        """Print a summary of evaluation results."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        # Retrieval accuracy
        total = len(results)
        correct_at_1 = sum(1 for r in results if r["retrieval"]["correct_at_1"])
        correct_in_top_k = sum(1 for r in results if r["retrieval"]["correct_in_top_k"])

        print(f"\nRetrieval Accuracy:")
        print(f"  Correct source at rank 1: {correct_at_1}/{total} ({100*correct_at_1/total:.1f}%)")
        print(f"  Correct source in top 3:  {correct_in_top_k}/{total} ({100*correct_in_top_k/total:.1f}%)")

        # Per-strategy metrics
        for strategy in ["zero_shot", "few_shot", "chain_of_thought"]:
            recalls = []
            lengths = []
            for r in results:
                if strategy in r["strategies"]:
                    kr = r["strategies"][strategy]["keyword_recall"]
                    if kr is not None:
                        recalls.append(kr)
                    lengths.append(r["strategies"][strategy]["answer_length"])

            avg_recall = sum(recalls) / len(recalls) if recalls else 0
            avg_length = sum(lengths) / len(lengths) if lengths else 0

            print(f"\n  {strategy}:")
            print(f"    Avg keyword recall: {avg_recall:.3f}")
            print(f"    Avg answer length:  {avg_length:.0f} chars")


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_full_evaluation()