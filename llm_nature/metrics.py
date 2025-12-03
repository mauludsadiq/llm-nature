import math
from typing import Iterable, Tuple, Callable, Hashable

Pair = Tuple[Hashable, Hashable]
ProbFn = Callable[[Hashable, Hashable], float]

def cross_entropy(prob: ProbFn, pairs: Iterable[Pair]) -> float:
    total_logp = 0.0
    n = 0
    for x, y in pairs:
        p = prob(y, x)
        if p <= 0.0:
            raise ValueError(f"Non positive probability p={p} for pair ({x!r}, {y!r})")
        total_logp -= math.log(p)
        n += 1
    if n == 0:
        raise ValueError("No pairs provided to cross_entropy")
    return total_logp / n
