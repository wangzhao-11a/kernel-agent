"""Structure/parameter separation: render a template over a numeric grid.

The LLM owns the structural template; numeric axes (tile/block/split/threads)
are declared separately and searched by a bounded grid.  This keeps parameter
values out of the strategy-attribution path.

Template convention:

    # TUNABLE: BLOCK=1024,2048,4096; SPLIT=1,2,4
    BLOCK = {{BLOCK}}
    SPLIT = {{SPLIT}}

or pass the space explicitly through the agent's `tune` action.
"""
from __future__ import annotations

import itertools
import random
import re

TUNABLE_RE = re.compile(r"^\s*#\s*TUNABLE\s*:\s*(.+)$", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}")


class ParamError(ValueError):
    pass


def _coerce(value: str):
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_values(value) -> list:
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return []
    # range(start, stop, step) support
    m = re.fullmatch(r"range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?\)", text)
    if m:
        start, stop = int(m.group(1)), int(m.group(2))
        step = int(m.group(3) or 1)
        if step == 0:
            raise ParamError("range step cannot be 0")
        return list(range(start, stop, step))
    return [_coerce(x) for x in re.split(r"[,;]", text) if str(x).strip()]


def parse_space(spec) -> dict:
    if not isinstance(spec, dict) or not spec:
        raise ParamError("params must be a non-empty object")
    space = {}
    for name, raw in spec.items():
        key = str(name).strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", key):
            raise ParamError("invalid parameter name: %r" % name)
        values = parse_values(raw)
        if not values:
            raise ParamError("parameter %s has no values" % key)
        space[key] = values
    return space


def parse_tunable(source: str) -> dict:
    match = TUNABLE_RE.search(source or "")
    if not match:
        return {}
    spec = {}
    for part in match.group(1).split(";"):
        if "=" not in part:
            continue
        name, values = part.split("=", 1)
        spec[name.strip()] = values.strip()
    return parse_space(spec)


def render(source: str, values: dict) -> str:
    text = source or ""
    for name, value in (values or {}).items():
        text = re.sub(r"\{\{\s*%s\s*\}\}" % re.escape(str(name)), str(value), text)
    leftovers = PLACEHOLDER_RE.findall(text)
    if leftovers:
        raise ParamError("unresolved placeholders: %s" % ",".join(sorted(set(leftovers))))
    return text


def grid(space: dict, max_variants: int = 6, seed: int = 0) -> list:
    names = list(space)
    if not names:
        return []
    combos = list(itertools.product(*(space[n] for n in names)))
    cap = max(1, int(max_variants))
    if len(combos) > cap:
        rng = random.Random(int(seed))
        idx = sorted(rng.sample(range(len(combos)), cap))
        combos = [combos[i] for i in idx]
    return [dict(zip(names, combo)) for combo in combos]


def validate(source: str, space: dict) -> dict:
    placeholders = set(PLACEHOLDER_RE.findall(source or ""))
    declared = set(space)
    return {
        "ok": placeholders == declared and bool(placeholders),
        "placeholders": sorted(placeholders),
        "declared": sorted(declared),
        "missing_declaration": sorted(placeholders - declared),
        "unused": sorted(declared - placeholders),
    }