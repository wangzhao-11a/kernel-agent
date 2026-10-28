"""Synthetic operator-search terrain used by meta-eval (CPU only).

The terrain models the qualitative shape of a kernel search: two broad peaks
whose location shifts with the mode, a split sweet spot, and noisy evaluations.
It is not a performance model of any real operator.
"""
from __future__ import annotations

import math
import random

TILES = (64, 128, 256, 512, 1024, 2048, 4096)
MODES = (0, 1, 2)
TOKENS_PER_PROPOSAL = 120


class SyntheticTerrain:
    def __init__(self, seed: int, threshold: float = 2.2):
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.rng = random.Random(seed)

    def reward(self, candidate, step: int = 0, noise_scale: float = 0.045) -> float:
        tile, split, mode = candidate
        x = math.log2(max(1, int(tile)))
        # Two peaks (512 / 2048) whose position shifts with mode.
        peak_a = 0.55 * math.exp(-((x - (9.0 + 0.45 * mode)) ** 2) / 0.55)
        peak_b = 0.40 * math.exp(-((x - (11.0 - 0.35 * mode)) ** 2) / 0.45)
        base = 1.0 + peak_a + peak_b
        split_f = 0.72 + 0.14 * split - 0.025 * max(0, split - 5) ** 2
        mode_f = 1.0 + 0.07 * mode
        noise = 1.0 + self.rng.gauss(0.0, noise_scale)
        return max(0.1, base * split_f * mode_f * noise)

    def neighbours(self, candidate):
        tile, split, mode = candidate
        out = []
        i = TILES.index(tile)
        if i > 0:
            out.append((TILES[i - 1], split, mode))
        if i + 1 < len(TILES):
            out.append((TILES[i + 1], split, mode))
        if split > 1:
            out.append((tile, split - 1, mode))
        if split < 8:
            out.append((tile, split + 1, mode))
        out.append((tile, split, (mode + 1) % len(MODES)))
        out.append((tile, split, (mode + len(MODES) - 1) % len(MODES)))
        return out

    def move(self, candidate, arm: str, rng: random.Random):
        tile, split, mode = candidate
        if arm == "tile":
            i = TILES.index(tile)
            j = max(0, min(len(TILES) - 1, i + rng.choice((-1, 1))))
            return (TILES[j], split, mode)
        if arm == "split":
            return (tile, max(1, min(8, split + rng.choice((-1, 1)))), mode)
        return (tile, split, (mode + rng.choice((-1, 1))) % len(MODES))

    def random_candidate(self, rng: random.Random):
        return (TILES[rng.randrange(len(TILES))], rng.randint(1, 8),
                MODES[rng.randrange(len(MODES))])