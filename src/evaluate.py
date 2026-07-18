"""
Step 5: Evaluate the RAG pipeline using RAGAs metrics - faithfulness,
answer relevancy, and context precision. Uses our own local Ollama
model as the judge, so this stays fully free/offline like the rest
of the project.
"""

from datasets import Dataset
from ragas import evaluate
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, answer_correctness
except ImportError:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from embed_store import get_embedding_model, load_vector_store
from retrieve import get_relevant_chunks
from generate import get_llm, answer_question

# A handful of test questions with ground truth answers for checking pipeline performance.
TEST_QUESTIONS = [
    {
        "question": "What vector databases does this project use?",
        "ground_truth": "The project supports two vector databases: FAISS (a lightweight local index for prototyping) and ChromaDB (a persistent database for scalable production-like setups)."
    },
    {
        "question": "What is RAGAs used for in this project?",
        "ground_truth": "RAGAs is used in this project to evaluate the performance and reliability of the RAG system using metrics like faithfulness, answer relevancy, context precision, and answer correctness."
    },
    {
        "question": "Does this project support voice-based interaction right now?",
        "ground_truth": "The project supports voice-based interaction as an enhancement, specifically offering a Read Aloud text-to-speech option in the user interface to read responses aloud."
    }
]


def generate_ground_truth_for_question(question, vector_store, llm, k=4):
    """
    Retrieves context for a question and asks the LLM to generate a factual,
    perfect ground truth answer strictly based on the retrieved context.
    """
    from retrieve import get_relevant_chunks, format_context
    chunks = get_relevant_chunks(question, vector_store, k=k)
    context = format_context(chunks)

    prompt = f"""You are an expert system. Generate a clear, direct, and factually complete reference answer (ground truth) to the question below based ONLY on the provided context.
If the context doesn't contain the answer, write: "I don't have enough information in the provided documents to answer this."

Context:
{context}

Question: {question}

Reference Answer:"""

    gt_answer = llm.invoke(prompt)
    if hasattr(gt_answer, "content"):
        gt_answer = gt_answer.content
    return gt_answer.strip()


def run_pipeline_for_eval(questions_or_records, vector_store, llm, k=4):
    """
    Runs our actual RAG pipeline for each question, collects the retrieved context,
    the generated answer, and generates a ground truth reference answer if one is not provided.
    """
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in questions_or_records:
        if isinstance(item, dict):
            q = item.get("question", "")
            gt = item.get("ground_truth", "")
        else:
            q = str(item)
            gt = ""

        if not q.strip():
            continue

        print(f"Running pipeline for evaluation: {q}")
        # Run standard RAG pipeline
        answer, chunks = answer_question(q, vector_store, llm, k=k)

        # If no ground truth was provided, generate one using the LLM with context
        if not gt.strip():
            print("  -> Generating reference ground truth via LLM...")
            gt = generate_ground_truth_for_question(q, vector_store, llm, k=k)

        records["question"].append(q)
        records["answer"].append(answer)
        records["contexts"].append([c.page_content for c in chunks])
        records["ground_truth"].append(gt)

    return records


def run_evaluation(records, judge_llm, judge_embeddings):
    """
    Hands our collected records to RAGAs and gets back a scored dataset.
    Configures the RAGAs wrappers to use our active LLM judge and embeddings.
    """
    dataset = Dataset.from_dict(records)

    wrapped_llm = LangchainLLMWrapper(judge_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )
    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    embedding_model = get_embedding_model()
    store = load_vector_store(embedding_model)
    llm = get_llm()

    print("Generating answers for test questions...\n")
    records = run_pipeline_for_eval(TEST_QUESTIONS, store, llm)

    print("\nRunning RAGAs evaluation (this calls the judge LLM several")
    print("times per question as a judge - can take a few minutes)...\n")
    results = run_evaluation(records, llm, embedding_model)

    print("\n=== EVALUATION RESULTS ===")
    print(results)

    df = results.to_pandas()
    print("\n=== PER-QUESTION BREAKDOWN ===")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]])

    df.to_csv("eval_results.csv", index=False)
    print("\nSaved detailed results to eval_results.csv")