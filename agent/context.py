"""Long-trajectory context management.

The task prompt and the most recent turns stay verbatim; everything in between
is replaced by a structured summary.  This keeps the current plan, the latest
failure and the best result in context without resending every source revision.
"""
from __future__ import annotations


def context_chars(messages) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def compress_messages(messages, summary: str, keep_recent: int = 4):
    """Return a shorter message list: first message + summary + recent tail.

    Roles are coalesced so providers that require user/assistant alternation keep
    working.  The summary is appended to the first user message.
    """
    messages = list(messages or [])
    if len(messages) <= keep_recent + 1:
        return messages
    keep_recent = max(2, int(keep_recent))
    head = dict(messages[0])
    omitted = messages[1:len(messages) - keep_recent]
    tail = [dict(m) for m in messages[-keep_recent:]]
    note = ("\n\n[compressed trajectory: %d earlier turn(s) omitted; use read_solution to re-read "
            "code and memory/profile actions for details]\n%s" % (len(omitted), summary.strip()))
    head["content"] = str(head.get("content") or "") + note

    out = [head]
    for message in tail:
        role = message.get("role") or "user"
        if out and out[-1].get("role") == role:
            out[-1]["content"] = (str(out[-1].get("content") or "") + "\n\n" +
                                  str(message.get("content") or ""))
        else:
            out.append(message)
    return out