"""Strategy detectors: does a solution actually contain the declared practice?

A strategy may carry an optional ``detector`` (regex or AST).  Declarations that
cannot be found in the solution source are ``claimed_only``: they do not update
the strategy pool.  Detectors are deliberately conservative heuristics; the
ablation measurement in `rsi.distill` is the stronger attribution signal.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize


def code_only(source: str) -> str:
    """Strip comments and string literals: attribution must match code, not prose."""
    try:
        pieces = []
        for tok in tokenize.generate_tokens(io.StringIO(source or "").readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(tok.string)
        return " ".join(pieces).lower()
    except Exception:
        return (source or "").lower()


def _call_name(node) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        bits = []
        cur = f
        while isinstance(cur, ast.Attribute):
            bits.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            bits.append(cur.id)
        return ".".join(reversed(bits))
    return ""


def _match_ast(detector: dict, source: str) -> bool:
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return False
    calls = [str(c).lower() for c in detector.get("calls") or []]
    attrs = [str(a).lower() for a in detector.get("attrs") or []]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node).lower()
            if any(name == c or name.endswith("." + c) for c in calls):
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr.lower() in attrs:
                return True
    return False


def match(detector, source: str) -> bool:
    """True when the solution's code satisfies the detector, or no detector is set."""
    if not detector:
        return True
    if not isinstance(detector, dict):
        return True
    kind = str(detector.get("kind") or "regex").lower()
    if kind == "ast":
        return _match_ast(detector, source)
    code = code_only(source)
    flags = re.IGNORECASE
    try:
        all_rules = [re.compile(str(p), flags) for p in (detector.get("all") or [])]
        any_rules = [re.compile(str(p), flags) for p in (detector.get("any") or [])]
        none_rules = [re.compile(str(p), flags) for p in (detector.get("none") or [])]
    except re.error:
        return True
    if any(r.search(code) for r in none_rules):
        return False
    if all_rules and not all(r.search(code) for r in all_rules):
        return False
    if any_rules and not any(r.search(code) for r in any_rules):
        return False
    return True


# Detectors for the seed practices.  They are heuristics: "online_softmax" means
# the code computes max and exp itself (a direct torch.softmax call does not
# qualify), while the other practices look for the canonical code shape.
SEED_DETECTORS = {
    "online_softmax": {"kind": "regex", "all": [r"\bmax\s*\(", r"\bexp\s*\("]},
    "block_tiling": {"kind": "regex", "any": [r"tile", r"block_?size", r"block_m", r"chunk"]},
    "multi_core_split": {"kind": "regex", "any": [r"block_dim", r"num_cores", r"core_num",
                                                  r"\bgrid\b", r"\bcore\b", r"multi_?core"]},
    "double_buffer_pipeline": {"kind": "regex", "any": [r"double_?buffer", r"pipeline",
                                                        r"prefetch", r"async_copy"]},
    "vectorized_access": {"kind": "regex", "any": [r"vectoriz", r"aligned", r"contiguous",
                                                   r"coalesc"]},
    "fused_epilogue": {"kind": "regex", "any": [r"epilogue", r"fuse", r"fusion"]},
    "avoid_host_sync": {"kind": "regex", "any": [r"synchronize", r"item\s*\(", r"cpu\s*\(",
                                                 r"non_blocking"]},
    "mixed_precision_accum": {"kind": "regex", "any": [r"float16", r"bfloat16", r"fp16",
                                                       r"bf16", r"half\s*\("]},
    "mla_absorption": {"kind": "regex", "any": [r"absorb", r"latent", r"c_kv"]},
    "grouped_gemm_single_launch": {"kind": "regex", "any": [r"grouped", r"group_list"]},
    "permute_once": {"kind": "regex", "any": [r"permute", r"sort\s*\(", r"gather"]},
    "split_k": {"kind": "regex", "any": [r"split_k", r"atomic_add"]},
    "sparse_skip_blocks": {"kind": "regex", "any": [r"sparse", r"mask", r"skip"]},
}


def detector_for(key: str):
    return SEED_DETECTORS.get(str(key or ""))
