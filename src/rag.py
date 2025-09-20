def build_prompt(question, results, system_prompt, recommend_top_k=3):
    """
    Build a prompt for the LLM using retrieved product chunks.

    Args:
        question (str): Customer question.
        results (list): List of retrieved product chunks with metadata.
        system_prompt (str): System instruction for the LLM.
        recommend_top_k (int): Number of products to suggest as recommendations.

    Returns:
        system_prompt (str), user_prompt (str)
    """
    # Build context from top results
    context = "\n\n".join(
        [f"[{i+1}] Product: {r.get('product_name', 'N/A')}\nDescription: {r['text']}" 
         for i, r in enumerate(results)]
    )

    # Include optional product recommendations
    recommendations = ""
    if len(results) > recommend_top_k:
        recommendations_list = [r.get('product_name', 'N/A') for r in results[recommend_top_k:]]
        recommendations = f"\n\nYou may also suggest similar products: {', '.join(recommendations_list)}"

    # Build user prompt
    user_prompt = (
        f"Answer the customer question using ONLY the context below:\n\n{context}"
        f"{recommendations}\n\nQuestion: {question}"
    )

    return system_prompt, user_prompt


def format_sources(results):
    """
    Format retrieved product chunks for display.

    Args:
        results (list): List of retrieved product chunks with metadata.

    Returns:
        str: Markdown-formatted string of sources.
    """
    return "\n".join(
        [
            f"[{i+1}] File: {r['source'].split('/')[-1]}, "
            f"Product: {r.get('product_name', 'N/A')}, "
            f"Score: {r['score']:.3f}"
            for i, r in enumerate(results)
        ]
    )
