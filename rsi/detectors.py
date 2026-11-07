"""Strategy detectors: does a solution actually contain the declared practice?

A strategy may carry an optional ``detector`` (regex or AST).  Declarations that
cannot be found in the solution source are ``claimed_only``: they do not update
the strategy pool.  Detectors are deliberately conservative heuristics; the
ablation measurement in `rsi.distill` is the stronger attribution signal.
"""
from __future__ import annotations

import ast
import difflib
import io
import re
import tokenize


def code_only(source: str) -> str:
    """Blank out comments and string literals, keeping the rest verbatim.

    The code is masked in place (same characters, spaces where the comment or
    string was) instead of re-joining tokens, so punctuation-sensitive regexes
    such as ``torch.softmax(`` still match.
    """
    text = source or ""
    keep = "\r\n"
    try:
        rows = [list(line) for line in text.splitlines(keepends=True)]
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (sr, sc), (er, ec) = tok.start, tok.end
            for row in range(sr - 1, min(er, len(rows))):
                line = rows[row]
                start = sc if row == sr - 1 else 0
                end = ec if row == er - 1 else len(line)
                for index in range(start, min(end, len(line))):
                    if line[index] not in keep:
                        line[index] = " "
        return "".join("".join(row) for row in rows).lower()
    except Exception:
        return text.lower()


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


def diff_ratio(original: str, variant: str) -> float:
    """Share of changed lines between two sources (0 = identical, 1 = rewritten)."""
    a = (original or "").splitlines()
    b = (variant or "").splitlines()
    if not a and not b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b)
    same = sum(block.size for block in matcher.get_matching_blocks())
    return 1.0 - (same / max(1, max(len(a), len(b))))


def validate_ablation(strategies: dict, target_id: str, member_ids, original: str,
                      variant: str, max_diff: float = 0.75):
    """Is this variant a credible 'remove exactly this practice' experiment?

    Requires: the target practice is gone (its detector no longer matches), the
    other practices of the best solution are still present, and the edit is not a
    rewrite of the whole file (which would make attribution meaningless).
    Returns (valid, reason).
    """
    problems = []
    target = (strategies or {}).get(target_id) or {}
    detector = target.get("detector")
    if not detector:
        problems.append("target has no detector")
    elif match(detector, variant):
        problems.append("target practice still present in the variant")
    for other in member_ids or ():
        if other == target_id:
            continue
        other_detector = ((strategies or {}).get(other) or {}).get("detector")
        if other_detector and not match(other_detector, variant):
            problems.append("variant removed an unrelated practice: %s" % other)
    ratio = diff_ratio(original, variant)
    if ratio > max_diff:
        problems.append("variant rewrites %.0f%% of the file (cap %.0f%%)"
                        % (100 * ratio, 100 * max_diff))
    return (not problems), "; ".join(problems)
