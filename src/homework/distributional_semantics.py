from __future__ import annotations

import numpy as np


def read_word_embeddings(path: str) -> dict[str, np.ndarray]:
    word_embeddings: dict[str, np.ndarray] = {}

    with open(path, encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            word = fields[0]
            embedding = np.array([float(value) for value in fields[1:]], dtype=float)
            word_embeddings[word] = embedding

    return word_embeddings


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denominator)


def similar_words(
    word_embeddings: dict[str, np.ndarray],
    target_word: str,
    threshold: float,
) -> list[tuple[str, float]]:
    if target_word not in word_embeddings:
        return []

    target_embedding = word_embeddings[target_word]
    similar = []

    for word, embedding in word_embeddings.items():
        if word == target_word:
            continue

        similarity = _cosine_similarity(target_embedding, embedding)
        if similarity >= threshold:
            similar.append((word, similarity))

    similar.sort(key=lambda item: item[1], reverse=True)
    return similar


def document_similarity(
    word_embeddings: dict[str, np.ndarray],
    document1: str,
    document2: str,
) -> float:
    embeddings1 = [word_embeddings[word] for word in document1.split() if word in word_embeddings]
    embeddings2 = [word_embeddings[word] for word in document2.split() if word in word_embeddings]

    if not embeddings1 or not embeddings2:
        return 0.0

    document_embedding1 = np.mean(np.array(embeddings1), axis=0)
    document_embedding2 = np.mean(np.array(embeddings2), axis=0)

    return _cosine_similarity(document_embedding1, document_embedding2)
