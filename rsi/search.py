"""Search policies over the synthetic terrain (CPU mechanism experiments).

`single_chain` is the original loop analogue.  `beam`, `ucb`, `thompson` and
`staged` are the Phase-3 candidates.  All policies share a fixed evaluation and
token budget so their results are comparable.

These policies are search-layer prototypes.  The production agent still emits
one JSON action per model turn; wiring beam/bandit into the LLM loop is a later
integration step.
"""
from __future__ import annotations

import math
import random

from .terrain import MODES, TILES, TOKENS_PER_PROPOSAL


class Budget:
    def __init__(self, max_evals: int, max_tokens: int, tokens_per: int = TOKENS_PER_PROPOSAL):
        self.max_evals = int(max_evals)
        self.max_tokens = int(max_tokens)
        self.tokens_per = int(tokens_per)
        self.evals = 0
        self.tokens = 0
        self.full_evals = 0
        self.cheap_evals = 0
        self.history = []

    def can_eval(self, cheap: bool = False) -> bool:
        cost = self.tokens_per // 4 if cheap else self.tokens_per
        return self.evals < self.max_evals and self.tokens + cost <= self.max_tokens

    def evaluate(self, terrain, candidate, cheap: bool = False):
        if not self.can_eval(cheap=cheap):
            return None
        cost = self.tokens_per // 4 if cheap else self.tokens_per
        reward = terrain.reward(candidate, self.evals, noise_scale=0.09 if cheap else 0.045)
        self.evals += 1
        self.tokens += cost
        if cheap:
            self.cheap_evals += 1
        else:
            self.full_evals += 1
        self.history.append({"candidate": list(candidate), "reward": round(reward, 4),
                             "cheap": bool(cheap)})
        return reward

    def as_dict(self):
        return {"evals": self.evals, "full_evals": self.full_evals,
                "cheap_evals": self.cheap_evals, "tokens": self.tokens,
                "history": self.history}


def _finish(terrain, budget, best, best_reward, threshold_hit, restarts=0):
    return {
        "best_candidate": list(best),
        "best_reward": round(float(best_reward), 4),
        "evals": budget.evals,
        "full_evals": budget.full_evals,
        "cheap_evals": budget.cheap_evals,
        "tokens": budget.tokens,
        "threshold_hit": bool(threshold_hit),
        "restarts": int(restarts),
        "history": budget.history,
    }


def single_chain(terrain, max_evals: int, token_budget: int, patience: int = 0):
    rng = random.Random(terrain.seed ^ 0x5EED)
    budget = Budget(max_evals, token_budget)
    current = (128, 1, 0)
    best, best_reward = current, budget.evaluate(terrain, current)
    no_gain = 0
    while budget.can_eval():
        proposal = terrain.move(current, rng.choice(("tile", "split", "mode")), rng)
        reward = budget.evaluate(terrain, proposal)
        if reward is None:
            break
        if reward > best_reward:
            best, best_reward = proposal, reward
            current = proposal
            no_gain = 0
        else:
            no_gain += 1
            if patience and no_gain >= patience:
                break
    return _finish(terrain, budget, best, best_reward,
                   best_reward >= terrain.threshold)


def beam_search(terrain, max_evals: int, token_budget: int, width: int = 8, patience: int = 0):
    """Best-first beam: expand the best unexpanded node, keep top-width frontier."""
    rng = random.Random(terrain.seed ^ 0xBEA3)
    budget = Budget(max_evals, token_budget)
    root = (128, 1, 0)
    root_reward = budget.evaluate(terrain, root)
    frontier = [(root_reward, root)]
    best, best_reward = root, root_reward
    seen = {root}
    expanded = set()
    restarts, no_gain = 0, 0
    while budget.can_eval():
        frontier.sort(key=lambda item: -item[0])
        chosen = None
        for reward, node in frontier:
            if node not in expanded:
                chosen = (reward, node)
                break
        if chosen is None:
            break
        _r, parent = chosen
        expanded.add(parent)
        added = 0
        for cand in terrain.neighbours(parent):
            if cand in seen:
                continue
            seen.add(cand)
            reward = budget.evaluate(terrain, cand)
            if reward is None:
                break
            frontier.append((reward, cand))
            added += 1
            if reward > best_reward:
                best, best_reward = cand, reward
                no_gain = 0
        if added == 0:
            no_gain += 1
        else:
            no_gain += 1
        frontier.sort(key=lambda item: -item[0])
        frontier = frontier[:max(1, int(width))]
        if patience and no_gain >= patience and budget.can_eval():
            cand = terrain.random_candidate(rng)
            if cand not in seen:
                seen.add(cand)
                reward = budget.evaluate(terrain, cand)
                if reward is not None:
                    frontier.append((reward, cand))
                    if reward > best_reward:
                        best, best_reward = cand, reward
            restarts += 1
            no_gain = 0
    return _finish(terrain, budget, best, best_reward,
                   best_reward >= terrain.threshold, restarts)


def _choose_ucb(counts, means, priors, total, c=1.2):
    best_arm, best_score = None, -1e9
    for arm in priors:
        n = counts.get(arm, 0)
        if n == 0:
            score = 10.0 * float(priors[arm])
        else:
            score = means[arm] + c * math.sqrt(math.log(max(2, total)) / n)
        if score > best_score:
            best_arm, best_score = arm, score
    return best_arm


def bandit_search(terrain, max_evals: int, token_budget: int, algorithm: str = "ucb",
                  pool_size: int = 5):
    rng = random.Random(terrain.seed ^ 0xA11CE)
    budget = Budget(max_evals, token_budget)
    priors = {"tile": 0.5, "split": 0.3, "mode": 0.2}
    counts = {a: 0 for a in priors}
    totals = {a: 0.0 for a in priors}
    means = {a: 0.0 for a in priors}
    root = (128, 1, 0)
    root_reward = budget.evaluate(terrain, root)
    pool = [(root_reward, root)]
    best, best_reward = root, root_reward
    seen = {root}
    while budget.can_eval():
        total = sum(counts.values())
        if algorithm == "thompson":
            arm = max(priors, key=lambda a: (means[a] if counts[a] else 0.0)
                      + (0.25 / math.sqrt(1 + counts[a])) * rng.gauss(0.0, 1.0)
                      + (0.5 * priors[a] if counts[a] == 0 else 0.0))
        else:
            arm = _choose_ucb(counts, means, priors, total)
        pool.sort(key=lambda item: -item[0])
        parent = pool[rng.randrange(min(len(pool), max(1, pool_size)))][1]
        cand = terrain.move(parent, arm, rng)
        if cand in seen:
            cand = terrain.random_candidate(rng)
        seen.add(cand)
        reward = budget.evaluate(terrain, cand)
        if reward is None:
            break
        counts[arm] += 1
        totals[arm] += reward
        means[arm] = totals[arm] / counts[arm]
        if reward > best_reward:
            best, best_reward = cand, reward
        pool.append((reward, cand))
        pool.sort(key=lambda item: -item[0])
        pool = pool[:max(1, pool_size)]
    return _finish(terrain, budget, best, best_reward,
                   best_reward >= terrain.threshold)


def map_elites_search(terrain, max_evals: int, token_budget: int, bins=None):
    """MAP-Elites archive over (tile bucket, split bucket, mode) descriptors.

    Each cell keeps its best candidate, so search pressure stays on diverse
    behaviours instead of collapsing onto one local optimum.
    """
    from .terrain import TILES

    rng = random.Random(terrain.seed ^ 0x3A9E)
    budget = Budget(max_evals, token_budget)
    bins = bins or len(TILES)
    archive = {}
    replacements = 0

    def descriptor(candidate):
        tile, split, mode = candidate
        return (min(bins - 1, int(TILES.index(tile) * bins / len(TILES))),
                min(3, (split - 1) // 2), mode)

    def place(candidate, reward):
        nonlocal replacements
        key = descriptor(candidate)
        if key not in archive or reward > archive[key][0]:
            if key in archive:
                replacements += 1
            archive[key] = (reward, candidate)

    root = (128, 1, 0)
    place(root, budget.evaluate(terrain, root))
    current = root
    while budget.can_eval():
        if archive and rng.random() < 0.35:
            _reward, current = archive[rng.choice(sorted(archive))]
        else:
            current = terrain.random_candidate(rng)
        for candidate in list(terrain.neighbours(current))[:3]:
            if not budget.can_eval():
                break
            place(candidate, budget.evaluate(terrain, candidate))
    best_reward, best = max(archive.values(), key=lambda item: item[0])
    return _finish(terrain, budget, best, best_reward,
                   best_reward >= terrain.threshold, replacements)


def staged_search(terrain, max_evals: int, token_budget: int, keep: int = 2):
    rng = random.Random(terrain.seed ^ 0x57A6E)
    budget = Budget(max_evals, token_budget)
    current = (128, 1, 0)
    best, best_reward = current, budget.evaluate(terrain, current)
    while budget.can_eval():
        candidates = list(terrain.neighbours(current))
        candidates.append(terrain.random_candidate(rng))
        scored = []
        for cand in candidates:
            r = budget.evaluate(terrain, cand, cheap=True)
            if r is None:
                break
            scored.append((r, cand))
        if not scored:
            break
        scored.sort(key=lambda item: -item[0])
        for _cheap, cand in scored[:max(1, keep)]:
            reward = budget.evaluate(terrain, cand)
            if reward is None:
                break
            if reward > best_reward:
                best, best_reward = cand, reward
                current = cand
    return _finish(terrain, budget, best, best_reward,
                   best_reward >= terrain.threshold)


POLICIES = {
    "single-chain": single_chain,
    "beam": beam_search,
    "ucb": lambda t, e, b: bandit_search(t, e, b, "ucb"),
    "thompson": lambda t, e, b: bandit_search(t, e, b, "thompson"),
    "staged": staged_search,
    "map-elites": map_elites_search,
}