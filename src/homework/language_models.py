from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Tuple

UNKNOWN = ""  
INIT = "[INIT]" 


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)
_PUNCT_SET = set(string.punctuation)


def _tokenize(line: str) -> List[str]:
    """
    Tokenize a line into word and punctuation tokens.
    Keeps punctuation as its own token.
    """
    return _TOKEN_RE.findall(line.strip())


def bigram_model(path: str) -> Dict[str, Dict[str, float]]:
    """
    Build a Laplace-smoothed bigram language model with normalization.
    Each line in the input file is treated as an independent sentence:
    [INIT] precedes the first token of each line.

    Returns:
        model[prev][curr] = P(curr | prev)

    Notes:
        - UNKNOWN probabilities are accessed using UNKNOWN for both prev and curr.
        - For unseen prev tokens, callers should use model[UNKNOWN].
    """
    # First pass: collect vocabulary
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = _tokenize(line)
            vocab.update(toks)

    # Ensure INIT and UNKNOWN are not in the normal vocab outcomes
    # (we will add UNKNOWN explicitly as an outcome key during smoothing)
    vocab.discard(INIT)
    vocab.discard(UNKNOWN)

    vocab_list = sorted(vocab)
    V = len(vocab_list)

    # Second pass: count bigrams
    bigram_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    context_totals: Counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = _tokenize(line)
            prev = INIT
            for curr in tokens:
                bigram_counts[prev][curr] += 1
                context_totals[prev] += 1
                prev = curr

    model: Dict[str, Dict[str, float]] = {}

    outcomes = vocab_list + [UNKNOWN]  # include UNKNOWN as an explicit outcome
    denom_add = V + 1 

    def _make_dist(prev: str) -> Dict[str, float]:
        counts = bigram_counts.get(prev, Counter())
        total = context_totals.get(prev, 0)
        denom = total + denom_add
        dist = {}
        for curr in outcomes:
            c = counts.get(curr, 0)
            dist[curr] = (c + 1) / denom
        return dist

    # Distributions for all observed contexts and for INIT
    all_prev = set(bigram_counts.keys()) | {INIT}
    for prev in all_prev:
        model[prev] = _make_dist(prev)

    model[UNKNOWN] = _make_dist(UNKNOWN)

    return model


def _get_prob(model: Dict[str, Dict[str, float]], prev: str, curr: str) -> float:
    """
    Fetch P(curr|prev) using UNKNOWN backoff for both prev and curr.
    """
    prev_dist = model.get(prev, model[UNKNOWN])
    return prev_dist.get(curr, prev_dist.get(UNKNOWN, 1e-12))


def _is_punct(tok: str) -> bool:
    return tok in _PUNCT_SET


def sequence_generator(
    model: Dict[str, Dict[str, float]],
    initial_word: str,
    length: int,
) -> Tuple[List[str], float]:
    """
    Greedy sequence generation:
    At each step choose the next token maximizing P(next|current),
    subject to:
      - exact sequence length
      - punctuation tokens <= floor(length/5)
      - excluding punctuation, no repeated tokens
    Returns:
      (sequence_tokens, log_likelihood) with natural log via math.log().
    """
    if length <= 0:
        return ([], float("-inf"))

    max_punct = length // 5

    seq: List[str] = [initial_word]
    used_nonpunct = set()
    punct_count = 0

    if _is_punct(initial_word):
        punct_count = 1
    else:
        used_nonpunct.add(initial_word)


    fallback_dist = model.get(UNKNOWN, {})
    fallback_order = sorted(
        (t for t in fallback_dist.keys() if t != UNKNOWN),
        key=lambda t: fallback_dist.get(t, 0.0),
        reverse=True,
    )

    while len(seq) < length:
        prev = seq[-1]
        dist = model.get(prev, model[UNKNOWN])

    
        candidates = [t for t in dist.keys() if t != UNKNOWN]
        candidates.sort(key=lambda t: dist.get(t, 0.0), reverse=True)

        chosen = None

        def ok(tok: str) -> bool:
            nonlocal punct_count
            if _is_punct(tok):
                return punct_count < max_punct
            return tok not in used_nonpunct

        for tok in candidates:
            if ok(tok):
                chosen = tok
                break

        # If none satisfy, try global fallback order
        if chosen is None:
            for tok in fallback_order:
                if ok(tok):
                    chosen = tok
                    break

        if chosen is None:
            chosen = candidates[0] if candidates else UNKNOWN
            if chosen == UNKNOWN:
                # If we somehow have no real candidates, stop early.
                break

        # Update constraints bookkeeping
        if _is_punct(chosen):
            punct_count += 1
        else:
            used_nonpunct.add(chosen)

        seq.append(chosen)

    ll = 0.0
    if seq:
        p0 = _get_prob(model, INIT, seq[0])
        ll += math.log(p0)

        for i in range(1, len(seq)):
            p = _get_prob(model, seq[i - 1], seq[i])
            ll += math.log(p)

    return seq, ll


def sequence_generator_plus(
    model: Dict[str, Dict[str, float]],
    initial_word: str,
    length: int,
) -> Tuple[List[str], float]:
    """
    Extra credit: constrained beam search for higher-probability, more coherent sequences.
    Still respects:
      - exact length
      - punctuation <= floor(length/5)
      - no repeated non-punctuation tokens
    """
    if length <= 0:
        return ([], float("-inf"))

    max_punct = length // 5
    beam_width = 6
    expand_top_k = 25

    def init_state():
        seq = [initial_word]
        punct = 1 if _is_punct(initial_word) else 0
        used = set() if _is_punct(initial_word) else {initial_word}
        ll = math.log(_get_prob(model, INIT, initial_word))
        return (ll, seq, used, punct)

    beam = [init_state()]

    while True:
        # stop when all sequences are full length
        if all(len(s[1]) >= length for s in beam):
            break

        new_beam = []

        for ll, seq, used, punct in beam:
            if len(seq) >= length:
                new_beam.append((ll, seq, used, punct))
                continue

            prev = seq[-1]
            dist = model.get(prev, model[UNKNOWN])

            candidates = [t for t in dist.keys() if t != UNKNOWN]
            candidates.sort(key=lambda t: dist.get(t, 0.0), reverse=True)
            candidates = candidates[:expand_top_k]

            for tok in candidates:
                if _is_punct(tok):
                    if punct >= max_punct:
                        continue
                    new_used = used
                    new_punct = punct + 1
                else:
                    if tok in used:
                        continue
                    new_used = set(used)
                    new_used.add(tok)
                    new_punct = punct

                p = _get_prob(model, prev, tok)
                new_ll = ll + math.log(p)
                new_seq = seq + [tok]
                new_beam.append((new_ll, new_seq, new_used, new_punct))

        if not new_beam:
            # If constraints are too tight, fall back to greedy completion
            return sequence_generator(model, initial_word, length)

        # Keep best beam_width by log-likelihood
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]

    best = max(beam, key=lambda x: x[0])
    return best[1], best[0]