from resources import SharedResources
from model_utils import rerank, score

def demo_rerank():
    query = "What is the capital of China?"
    documents = [
        "Shanghai is a large city in China.",
        "The capital of China is Beijing.",
    ]

    ranked, scores_aligned, raw = rerank(
        query=query,
        documents=documents,
        top_n=2,
    )

    print("raw payload:")
    print({"query": query, "documents": documents, "top_n": 2})
    print("raw result:")
    print(raw)

    print("Rerank results:")
    for item in ranked:
        print(f"- score={item.score:.4f} | idx={item.index} | doc={item.doc}")

    print("scores aligned:", scores_aligned)

def demo_score():
    model = "qwen3-reranker-0.6b"
    query = "What is the capital of China?"
    documents = [
        "Shanghai is a large city in China.",
        "The capital of China is Beijing.",
        "My best city is Beijing.",
    ]

    items, scores_aligned, raw = score(
        model=model,
        query=query,
        documents=documents,
    )

    print("raw payload:")
    print({"model": model, "text_1": query, "text_2": documents})

    print("raw result:")
    print(raw)

    print("Score results (not sorted):")
    for x in items:
        print(f"- score={x.score:.4f} | idx={x.index} | doc={x.doc}")

    print("scores aligned:", scores_aligned)


if __name__ == "__main__":
    demo_rerank()
    demo_score()
