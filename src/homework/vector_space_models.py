from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"(?u)\b[\w']+\b")
OPTIMAL_K = 91
EXTRA_K = 81


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _read_training_document(document: str | tuple[int, str]) -> tuple[int, str]:
    if isinstance(document, tuple):
        return int(document[0]), str(document[1])

    label, text = document.rstrip("\n").split("\t", 1)
    return int(label), text


def _read_test_document(document: str | tuple[int, str]) -> str:
    if isinstance(document, tuple):
        return str(document[1])

    parts = document.rstrip("\n").split("\t", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return document


def _build_vocabulary(documents: list[list[str]]) -> dict[str, int]:
    vocab = set()
    for document in documents:
        vocab.update(document)
    return {word: i for i, word in enumerate(sorted(vocab))}


def _document_frequencies(vocab: dict[str, int], documents: list[list[str]]) -> dict[int, int]:
    counts = Counter()
    for document in documents:
        counts.update(set(document))
    return {vocab[word]: count for word, count in counts.items() if word in vocab}


def _tf_idf(
    vocab: dict[str, int],
    dfs: dict[int, int],
    document_count: int,
    document: list[str],
    *,
    sublinear_tf: bool = False,
) -> dict[int, float]:
    counts = Counter(document)
    vector: dict[int, float] = {}
    length = len(document)

    if length == 0:
        return vector

    for word, count in counts.items():
        if word not in vocab:
            continue
        term_id = vocab[word]
        tf = 1.0 + math.log(count) if sublinear_tf else count / length
        idf = math.log(document_count / dfs[term_id])
        vector[term_id] = tf * idf

    return vector


def _cosine_similarity(v1: dict[int, float], v2: dict[int, float]) -> float:
    numerator = sum(value * v2.get(key, 0.0) for key, value in v1.items())
    denominator = math.sqrt(sum(value * value for value in v1.values()))
    denominator *= math.sqrt(sum(value * value for value in v2.values()))

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def sentiment_analyzer(
    training_documents: list[str | tuple[int, str]],
    test_documents: list[str | tuple[int, str]],
) -> tuple[list[int], list[float]]:
    return _sentiment_analyzer(training_documents, test_documents, k=OPTIMAL_K, sublinear_tf=False)


def sentiment_analyzer_extra(
    training_documents: list[str | tuple[int, str]],
    test_documents: list[str | tuple[int, str]],
) -> tuple[list[int], list[float]]:
    return _sentiment_analyzer(training_documents, test_documents, k=EXTRA_K, sublinear_tf=True)


def _sentiment_analyzer(
    training_documents: list[str | tuple[int, str]],
    test_documents: list[str | tuple[int, str]],
    *,
    k: int,
    sublinear_tf: bool,
) -> tuple[list[int], list[float]]:
    training_data = [_read_training_document(document) for document in training_documents]
    training_tokens = [_tokenize(text) for _, text in training_data]

    vocab = _build_vocabulary(training_tokens)
    dfs = _document_frequencies(vocab, training_tokens)
    document_count = len(training_tokens)

    training_vectors = []
    for label, text in training_data:
        vector = _tf_idf(vocab, dfs, document_count, _tokenize(text), sublinear_tf=sublinear_tf)
        training_vectors.append((label, vector))

    predicted_labels: list[int] = []
    similarity_scores: list[float] = []

    for document in test_documents:
        test_text = _read_test_document(document)
        test_vector = _tf_idf(vocab, dfs, document_count, _tokenize(test_text), sublinear_tf=sublinear_tf)

        similarities = []
        for label, train_vector in training_vectors:
            score = _cosine_similarity(test_vector, train_vector)
            similarities.append((label, score))

        similarities.sort(key=lambda item: item[1], reverse=True)
        top_k = similarities[:k]

        top_labels = [label for label, _ in top_k]
        label_similarity_sums = Counter()
        for label, score in top_k:
            label_similarity_sums[label] += score

        label_counts = Counter(top_labels)
        predicted_label = label_counts.most_common(1)[0][0]

        predicted_labels.append(predicted_label)
        similarity_scores.append(label_similarity_sums[predicted_label] / label_counts[predicted_label])

    return predicted_labels, similarity_scores
