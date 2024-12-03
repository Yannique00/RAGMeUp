def compute_relevance_score(bm25_score, feedback_score, alpha=0.7, beta=0.3):
    """
    Compute the relevance score based on BM25 score and user feedback.
    """
    return alpha * bm25_score + beta * feedback_score

def rerank_documents_with_feedback(query, documents, feedback_db):
    """
    Rerank documents based on BM25 scores and user feedback.
    """
    # Retrieve documents with BM25 scores
    retrieved_docs = retrieve_with_bm25(query)

    # Fetch feedback scores for the documents
    for doc in retrieved_docs:
        feedback = feedback_db.get_feedback(query, doc["id"])  # Fetch cumulative thumbs feedback
        doc["feedback_score"] = feedback if feedback is not None else 0

    # Compute new relevance scores
    for doc in retrieved_docs:
        doc["relevance_score"] = compute_relevance_score(
            bm25_score=doc["bm25_score"], 
            feedback_score=doc["feedback_score"]
        )

    # Sort documents by relevance score
    reranked_docs = sorted(retrieved_docs, key=lambda x: x["relevance_score"], reverse=True)
    return reranked_docs
