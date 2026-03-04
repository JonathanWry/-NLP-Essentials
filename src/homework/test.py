# src/homework/test_task1_random_norm.py
import random
from language_models import bigram_model, UNKNOWN, INIT

m = bigram_model("dat/chronicles_of_narnia.txt")

keys = list(m.keys())
sample = [INIT, UNKNOWN] + random.sample(keys, k=min(50, len(keys)))

bad = []
for prev in sample:
    s = sum(m.get(prev, m[UNKNOWN]).values())
    if abs(s - 1.0) > 1e-6:
        bad.append((prev, s))

if bad:
    print("FAILED normalization on:", bad[:10])
else:
    print("PASS: normalization holds for sampled contexts")