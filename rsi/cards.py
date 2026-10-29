"""CANN knowledge cards (cannbot-knowledge) as a retrieve-on-demand knowledge source.

Cards are Markdown with YAML frontmatter (type/title/description/tags/status/platforms/
updated_at), laid out as knowledge/<domain>/<scope>/.../card.md (plus operators/, recipes/).

Protocol implemented here follows the CANN team's own guidance: filter first (candidates
with one-line applicability), read on demand (the agent's `knowledge` action loads a body),
verify against sources when needed. Matching is deterministic (tags + op-family keywords +
platform filter); no embeddings, so retrieval stays explainable and cheap.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

_REPO_KB_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge", "cards")
DEFAULT_KB_ROOT = os.environ.get("KERNEL_AGENT_KB_ROOT", _REPO_KB_ROOT)
if not os.path.isdir(DEFAULT_KB_ROOT) and os.path.isdir("/data2/kernel-agent/kb/cannbot-knowledge"):
    DEFAULT_KB_ROOT = "/data2/kernel-agent/kb/cannbot-knowledge"
INDEX_NAME = ".cards_index.json"
EXCERPT_CHARS = 1200   # body excerpt kept in the index so scoring sees card content, not just metadata
INDEX_TTL_SECONDS = 7 * 24 * 3600
SCAN_DIRS = ("knowledge", "operators", "recipes")

OP_FAMILY_HINTS = {
    "attention": ("attention", "flash", "softmax", "mla", "sparse", "decode", "rope"),
    "gemm": ("matmul", "gemm", "grouped", "quant", "mm", "bmm"),
    "moe": ("moe", "expert", "permute", "finalize", "routing", "gating", "swiglu", "moe_gating"),
    "reduction": ("rmsnorm", "layernorm", "reduce", "reduction", "sum", "norm", "softmax"),
    "elementwise": ("elementwise", "add", "mul", "silu", "gelu", "cast", "copy", "rope", "slice"),
}


def _parse_frontmatter(text: str) -> dict:
    """Tolerant YAML-frontmatter reader for the fields we index (no PyYAML dependency)."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    meta = {}
    for line in text[3:end].splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        else:
            meta[key] = value.strip("'\"")
    return meta


def _iter_card_paths(kb_root: str):
    for top in SCAN_DIRS:
        base = os.path.join(kb_root, top)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if name.endswith(".md") and name.lower() != "index.md":
                    yield os.path.join(root, name)


def build_index(kb_root: str) -> list:
    cards = []
    for path in _iter_card_paths(kb_root):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(4000)
        except OSError:
            continue
        meta = _parse_frontmatter(head)
        rel = os.path.relpath(path, kb_root).replace("\\", "/")
        body_at = head.find("\n---", 3)
        excerpt = head[body_at + 4:] if body_at >= 0 else head
        excerpt = " ".join(excerpt.split())[:EXCERPT_CHARS]
        cards.append({
            "id": rel,
            "excerpt": excerpt,
            "title": str(meta.get("title") or os.path.basename(rel)),
            "type": str(meta.get("type") or ""),
            "description": str(meta.get("description") or ""),
            "tags": [str(t) for t in (meta.get("tags") or [])],
            "status": str(meta.get("status") or ""),
            "platforms": [str(p).lower() for p in (meta.get("platforms") or [])],
            "updated_at": str(meta.get("updated_at") or ""),
        })
    return cards


def _git_head(kb_root: str) -> str:
    try:
        out = subprocess.run(["git", "-C", kb_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def load_index(kb_root: str, rebuild: bool = False) -> list:
    """Load (or build and cache) the card index.

    The cache is invalidated when the KB's git HEAD moves (so `git pull` refreshes it)
    and after INDEX_TTL_SECONDS otherwise.
    """
    cache = os.path.join(kb_root, INDEX_NAME)
    head = _git_head(kb_root)
    if not rebuild and os.path.isfile(cache):
        try:
            with open(cache, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
                fresh = time.time() - os.path.getmtime(cache) < INDEX_TTL_SECONDS
                same_head = (not head) or (payload.get("head") == head)
                if fresh and same_head:
                    return payload["cards"]
        except (OSError, ValueError):
            pass
    cards = build_index(kb_root)
    try:
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump({"head": head, "built_at": time.time(), "cards": cards}, fh, ensure_ascii=False)
    except OSError:
        pass
    return cards


def _platform_ok(card: dict, platforms) -> bool:
    card_platforms = card.get("platforms") or []
    if not card_platforms:
        return True  # unknown -> keep, scoring decides
    return any(p in card_platforms for p in platforms)


def _score(card: dict, feats: dict, required_form: str = "") -> float:
    text = " ".join([card.get("title", ""), card.get("description", ""),
                     " ".join(card.get("tags") or []), card.get("id", ""),
                     card.get("excerpt", "")]).lower()
    score = 0.0
    for tag in feats.get("tags") or []:
        if len(tag) > 3 and tag.lower() in text:
            score += 0.5
    for token in re.findall(r"[a-z0-9]+", (feats.get("name") or "").lower()):
        if len(token) > 3 and token in text:
            score += 0.6
    for keyword in OP_FAMILY_HINTS.get(feats.get("op_class") or "", ()):
        if keyword in text:
            score += 0.3
    if str(card.get("type", "")).lower() == "runbook" or "/runbooks/" in card.get("id", ""):
        score += 0.4
    if "/performance/" in card.get("id", ""):
        score += 0.4
    if card.get("status") == "stable":
        score += 0.2
    if required_form == "triton_kernel":
        # when the task wants a kernel, prefer triton/ops implementation material over
        # system-level architecture notes
        if "/ops/triton" in card.get("id", "") or "triton" in text:
            score += 0.8
        elif card.get("id", "").startswith("knowledge/ops/"):
            score += 0.3
    elif required_form == "ascendc":
        # AscendC tasks must not be steered toward triton-only recipes.
        if "ascendc" in text or "/ops/ascendc" in card.get("id", "").lower():
            score += 0.8
        if "triton" in text:
            score -= 0.6
    return score


def retrieve_cards(kb_root: str, feats: dict, top_k: int = 4, min_score: float = 1.0,
                   platforms=("a2", "a3"), required_form: str = "") -> list:
    if not kb_root or not os.path.isdir(kb_root) or top_k <= 0:
        return []
    scored = []
    for card in load_index(kb_root):
        if not _platform_ok(card, platforms):
            continue
        score = _score(card, feats, required_form=required_form)
        if score >= min_score:
            scored.append((card, score))
    scored.sort(key=lambda item: -item[1])
    return [card for card, _score_value in scored[:top_k]]


def search_cards(kb_root: str, query: str, top_k: int = 4,
                 platforms=("a2", "a3"), min_score: float = 0.0) -> list:
    """Agent-driven free-text search over the same card index.

    This is the tool behind `{"action":"kb_search","query":"..."}`.  It scores
    title/description/tags/id/excerpt by query-token coverage, so the agent can
    ask for "32B alignment SwiGLU" instead of only accepting harness-selected cards.
    """
    if not kb_root or not os.path.isdir(kb_root) or top_k <= 0:
        return []
    tokens = [t for t in re.split(r"[\s,;:/|]+", str(query or "").lower()) if len(t) >= 2]
    if not tokens:
        return []
    scored = []
    for card in load_index(kb_root):
        if not _platform_ok(card, platforms):
            continue
        text = " ".join([card.get("title", ""), card.get("description", ""),
                         " ".join(card.get("tags") or []), card.get("id", ""),
                         card.get("excerpt", "")]).lower()
        hits = sum(1 for token in tokens if token in text)
        if not hits:
            continue
        score = hits / float(len(tokens))
        if str(query).lower() in text:
            score += 1.0
        if card.get("status") == "stable":
            score += 0.1
        if score >= min_score:
            scored.append((card, score))
    scored.sort(key=lambda item: -item[1])
    return [card for card, _score in scored[:top_k]]


def render_candidates(cards: list, max_desc: int = 220) -> str:
    if not cards:
        return ""
    lines = [
        "## CANN knowledge cards — candidates (filter first, read on demand)",
        "Matching cards from cannbot-knowledge for this operator family. Load one with",
        "{\"action\": \"knowledge\", \"id\": \"<path>\"} when you need its recipe/details.",
    ]
    for card in cards:
        desc = card.get("description") or card.get("title") or ""
        if len(desc) > max_desc:
            desc = desc[:max_desc] + "…"
        bits = ["- %s" % card["id"]]
        meta = " | ".join(filter(None, [card.get("type") or "", card.get("status") or "",
                                        ",".join(card.get("platforms") or [])]))
        lines.append("%s\n    %s%s" % (bits[0], desc, ("  [" + meta + "]") if meta else ""))
    return "\n".join(lines)


def read_card(kb_root: str, card_id: str, max_chars: int = 6000) -> str:
    """Load a card body (frontmatter + content), budgeted; returns '' when not found."""
    card_id = str(card_id or "").strip().replace("\\", "/")
    if not card_id or ".." in card_id or card_id.startswith("/"):
        return ""
    path = os.path.join(kb_root, card_id)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read()
    except OSError:
        return ""
    if len(body) > max_chars:
        body = body[:max_chars] + "\n\n…(truncated; ask again or read the file on the host for the rest)"
    return body