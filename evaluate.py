from rag_backend import get_rag_retriever
from eval_set import EVAL_QUESTIONS
from sentence_transformers import SentenceTransformer, util

# Load models
retriever = get_rag_retriever(persist_dir="../data/vector_store")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_match(a, b, threshold=0.6):
    emb_a = embedder.encode(a, convert_to_tensor=True)
    emb_b = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item() >= threshold


def evaluate(eval_questions, top_k=5):
    stats = {
        "hit@1": 0,
        "hit@3": 0,
        "hit@5": 0,
        "mrr": 0,
        "negatives_correct": 0,
    }

    for q in eval_questions:
        retrieved = retriever.retrieve(q["question"], top_k)

        # NEGATIVE QUESTIONS
        if q["source_hint"] is None:
            if len(retrieved) == 0:
                stats["negatives_correct"] += 1
            continue

        hit_rank = None

        for idx, doc in enumerate(retrieved, start=1):
            if semantic_match(doc["content"], q["source_hint"]):
                hit_rank = idx
                break

        if hit_rank:
            stats["mrr"] += 1 / hit_rank
            if hit_rank <= 1:
                stats["hit@1"] += 1
            if hit_rank <= 3:
                stats["hit@3"] += 1
            if hit_rank <= 5:
                stats["hit@5"] += 1
        else:
            print("MISSED:", q["question"])

    total_positive = len([q for q in eval_questions if q["source_hint"] is not None])

    print("\n=== Retrieval Evaluation ===")
    print(f"Hit@1: {stats['hit@1']/total_positive:.2%}")
    print(f"Hit@3: {stats['hit@3']/total_positive:.2%}")
    print(f"Hit@5: {stats['hit@5']/total_positive:.2%}")
    print(f"MRR: {stats['mrr']/total_positive:.3f}")
    print(f"Negatives correct: {stats['negatives_correct']}")

if __name__ == "__main__":
    evaluate(EVAL_QUESTIONS)