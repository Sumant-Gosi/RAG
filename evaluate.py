import os
from rag_backend import get_rag_retriever, rag_with_memory, get_query_rewriter
from eval_set import EVAL_QUESTIONS
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
import json
from typing import List, Dict

# Load models
retriever = get_rag_retriever(persist_dir="../data/vector_store")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

query_rewriter = get_query_rewriter(GROQ_API_KEY)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)


def semantic_match(a, b, threshold=0.6):
    """Check if two texts are semantically similar"""
    emb_a = embedder.encode(a, convert_to_tensor=True)
    emb_b = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item() >= threshold


def evaluate_retrieval(eval_questions: List[Dict], initial_k: int = 20, top_k: int = 5):
    """Evaluate retrieval performance"""
    stats = {
        "hit@1": 0,
        "hit@3": 0,
        "hit@5": 0,
        "mrr": 0,
        "negatives_correct": 0,
        "total_positive": 0,
        "total_negative": 0,
    }
    
    missed_queries = []

    for q in eval_questions:
        retrieved = retriever.retrieve(q["question"], initial_k=initial_k, top_k=top_k)

        # NEGATIVE QUESTIONS (should return nothing)
        if q["source_hint"] is None:
            stats["total_negative"] += 1
            if len(retrieved) == 0:
                stats["negatives_correct"] += 1
            else:
                print(f"FALSE POSITIVE: '{q['question']}' returned {len(retrieved)} docs")
            continue

        stats["total_positive"] += 1
        hit_rank = None

        # Check if correct document is in results
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
            missed_queries.append({
                "question": q["question"],
                "expected_hint": q["source_hint"][:100]
            })

    # Calculate percentages
    total_positive = stats["total_positive"]
    total_negative = stats["total_negative"]

    print("\n" + "="*50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*50)
    print(f"\nPositive queries (should retrieve): {total_positive}")
    print(f"Negative queries (should not retrieve): {total_negative}")
    print(f"\n--- Retrieval Metrics ---")
    print(f"Hit@1: {stats['hit@1']}/{total_positive} ({stats['hit@1']/total_positive:.2%})")
    print(f"Hit@3: {stats['hit@3']}/{total_positive} ({stats['hit@3']/total_positive:.2%})")
    print(f"Hit@5: {stats['hit@5']}/{total_positive} ({stats['hit@5']/total_positive:.2%})")
    print(f"MRR (Mean Reciprocal Rank): {stats['mrr']/total_positive:.3f}")
    print(f"\n--- Negative Handling ---")
    print(f"Negatives correct: {stats['negatives_correct']}/{total_negative} ({stats['negatives_correct']/total_negative:.2%})")
    
    if missed_queries:
        print(f"\n--- Missed Queries ({len(missed_queries)}) ---")
        for miss in missed_queries[:5]:  # Show first 5
            print(f"  • {miss['question']}")
            print(f"    Expected: {miss['expected_hint']}")
    
    return stats


def evaluate_end_to_end(eval_questions: List[Dict], num_samples: int = 10):
    """
    Evaluate end-to-end answer quality using LLM-as-judge.
    Only evaluates a sample due to API costs.
    """
    print("\n" + "="*50)
    print("END-TO-END EVALUATION (LLM-as-Judge)")
    print("="*50)
    
    # Sample queries
    import random
    sample = random.sample([q for q in eval_questions if q["source_hint"] is not None], 
                          min(num_samples, len(eval_questions)))
    
    scores = []
    
    judge_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=512,
    )
    
    for i, q in enumerate(sample, 1):
        # Generate answer
        answer, _, citations = rag_with_memory(
            query=q["question"],
            retriever=retriever,
            llm=llm,
            chat_history=[],
            query_rewriter=query_rewriter
        )
        
        # Judge answer quality
        judge_prompt = f"""You are evaluating a RAG system's answer quality.

Question: {q["question"]}

Generated Answer: {answer}

Rate the answer on a scale of 1-5:
5 = Perfect: Accurate, complete, well-sourced
4 = Good: Accurate with minor issues
3 = Acceptable: Partially correct or incomplete
2 = Poor: Significant inaccuracies
1 = Bad: Wrong or unhelpful

Provide your rating as a single number (1-5) followed by a brief explanation.
Format: RATING: X
REASON: [explanation]"""
        
        try:
            from langchain_core.messages import HumanMessage
            judgment = judge_llm.invoke([HumanMessage(content=judge_prompt)]).content
            
            # Extract rating
            if "RATING:" in judgment:
                rating_line = [line for line in judgment.split("\n") if "RATING:" in line][0]
                rating = int(rating_line.split(":")[1].strip()[0])
                scores.append(rating)
                
                print(f"\n[{i}/{num_samples}] Q: {q['question'][:60]}...")
                print(f"   Rating: {rating}/5")
                print(f"   Citations: {len(citations)}")
            else:
                print(f"\n[{i}/{num_samples}] Failed to parse judgment")
        
        except Exception as e:
            print(f"\n[{i}/{num_samples}] Error: {e}")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n--- Answer Quality ---")
        print(f"Average Rating: {avg_score:.2f}/5.0")
        print(f"Samples Evaluated: {len(scores)}")
        print(f"Distribution: {dict(zip(*zip(*[(s, scores.count(s)) for s in set(scores)])))}") 
    
    return scores


if __name__ == "__main__":
    print("Starting RAG Evaluation...\n")
    
    # 1. Retrieval evaluation
    retrieval_stats = evaluate_retrieval(EVAL_QUESTIONS, initial_k=20, top_k=5)
    
    # 2. End-to-end evaluation (smaller sample)
    # Uncomment if you want to run LLM-as-judge (costs API credits)
    # answer_scores = evaluate_end_to_end(EVAL_QUESTIONS, num_samples=5)
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)