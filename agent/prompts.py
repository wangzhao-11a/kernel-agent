"""Prompts for the minimal kernel-optimization agent loop (with RSI priors)."""
from __future__ import annotations

_SYSTEM = """You are a senior GPU/NPU kernel optimization engineer working as an autonomous agent.
Your goal: given an operator task with a reference implementation, write a FASTER implementation
that stays exactly correct, iterating on evaluation feedback. You also feed a shared strategy
pool: declare which known practices you applied and predict the speedup, so the pool can learn.

## Solution contract
- One self-contained Python file defining `run(*inputs)`; it must return the same values,
  shapes and dtypes as the reference implementation.
- Allowed imports: torch, plus triton only if it is already installed. Do NOT import the task
  file or the harness, do not hardcode expected outputs, and do not cache results across calls:
  the harness re-generates inputs per trial, detects input mutation, and flags task/harness
  references in your file.
- Never mutate the input tensors. Do not print more than a few lines.
- The solution must run on device `{device}`. Read `x.device` instead of hardcoding `.cuda()`,
  and guard accelerator-only code paths so the same file also works on cpu.
- Correctness is checked over multiple random trials with atol={atol}, rtol={rtol};
  shapes and dtypes must match exactly.
- Performance is the median of {rep} timed iterations after {warmup} warmup iterations,
  reported as speedup versus the `{baseline}` baseline; a second pass reports stability.

## Response protocol (STRICT)
Reply with exactly ONE JSON object, no prose outside it:
  {{"action": "write_solution", "content": "<complete python source>",
   "strategies": ["<names or ids of practices you applied>"],
   "expected_speedup": 1.25}}
      strategies / expected_speedup are optional but strongly encouraged: they feed the
      strategy pool (calibration + evidence) for this and later tasks.
  {{"action": "tune", "content": "<template with {{{{NAME}}}} placeholders>",
   "params": {{"NAME": [1024, 2048, 4096]}}}}
      Keep the structural template fixed and search only the declared numeric axes
      on a bounded grid (tile/block/split/threads). Use it after correctness is
      established, instead of rewriting the same code with different constants.
  {{"action": "run_eval"}}
      Re-evaluate the current solution unchanged.
  {{"action": "read_solution"}}
      Print the current solution source (useful after the trajectory was compressed).
  {{"action": "profile"}}
      Profile the current solution (op-level self-time breakdown) to guide the next change.
      Optional "case": "<shape_set case name>" and "trial": <int> profile one shape only.
  {{"action": "memory", "query": "<free text>"}}
      Search remembered facts/failures/skills/strategies on demand. Results are text matches,
      NOT scope-matched: check each record's scope before applying it.
  {{"action": "knowledge", "id": "<card path from the candidates list>"}}
      Load a CANN knowledge card (recipe + applicability conditions) when you need its details.
  {{"action": "finish", "summary": "<2-4 sentences: final approach, speedup, caveats>"}}

## Strategy
1. First get a simple, correct implementation accepted (clean correctness beats raw speed).
2. When code exemplars or priors are provided, treat them as verified clues for matching
   conditions — reuse their ideas, but re-check correctness and try to beat their speedup.
2. Then optimize one change at a time: prefer practices from the tuning priors that match this
   task's conditions; fuse elementwise chains, avoid extra memory passes, avoid host-device syncs,
   improve memory access patterns, use faster fused library calls where allowed. When a profiling
   breakdown is present, attack the op that dominates self time first. Re-check correctness after
   every change.
3. If evaluation fails, read the error/reason carefully and fix the root cause.
4. If timings look unstable, re-run. Keep your best version; never trade correctness for speed.
"""

_TASK = """# Task: {name} (level {level}) — device `{device}`
{description}

## Task file (reference implementation + input generation) — read-only
```python
{source}
```

{constraints}
{form}
{priors}

{memory}

{knowledge}

{exemplars}

## Current state
{state}

Iteration budget: up to {max_iters} evaluations. Reply with exactly one JSON action.
"""


def _constraints_block(task) -> str:
    meta = getattr(task, "meta", {}) or {}
    requires = [str(r).lower() for r in (meta.get("require") or [])]
    baseline = str(meta.get("baseline", "eager"))
    bits = []
    if "triton" in requires:
        bits.append("- HARD REQUIREMENT: this task needs a triton (triton-ascend) implementation. "
                    "The module must `import triton` and use at least one `@triton.jit` kernel for the "
                    "main compute; a torch-only solution is rejected with `fail_constraint`.")
    if baseline.startswith("op"):
        bits.append("- The baseline is the vendor's official fused op (correctness-checked first): "
                    "a plain torch reimplementation will not beat it — target the kernel path.")
    if requires:
        bits.append("- Harness-enforced requirements: %s" % ", ".join(requires))
    if not bits:
        return ""
    return "## HARD CONSTRAINTS\n" + "\n".join(bits) + "\n"


def system_prompt(device: str, atol: float, rtol: float, warmup: int, rep: int, baseline: str) -> str:
    return _SYSTEM.format(device=device, atol=atol, rtol=rtol, warmup=warmup, rep=rep, baseline=baseline)


def build_task_prompt(task, device: str, max_iters: int, state: str = "No solution submitted yet.",
                      priors: str = "", exemplars: str = "", knowledge: str = "",
                      form: str = "", memory: str = "") -> str:
    return _TASK.format(
        name=task.name,
        level=task.level,
        device=device,
        description=task.description or "(no description)",
        source=task.source,
        constraints=_constraints_block(task),
        form=form or "",
        priors=priors or "## Tuning priors\n(none retrieved for this task)",
        memory=memory or "## Operator memory\n(no facts/failures/skills matched this task)",
        knowledge=knowledge or "",
        exemplars=exemplars or "",
        state=state,
        max_iters=max_iters,
    )


def build_feedback(iteration: int, status: str, summary: str, speedup: float, best: dict,
                   hints: str = "", profiling: str = "") -> str:
    lines = ["Evaluation result (iteration %d):" % iteration, "```", summary, "```"]
    if profiling:
        lines.append("Profiling (op-level self time):")
        lines.append("```")
        lines.append(profiling)
        lines.append("```")
    if status == "pass":
        lines.append("This solution passed. Speedup vs baseline: %.3fx (best so far: %.3fx)."
                     % (speedup, best.get("speedup", 0.0)))
        lines.append("You may reply with finish, or another write_solution to push the speedup further.")
    else:
        lines.append("This solution did NOT pass. Fix the root cause and reply with a new write_solution.")
        lines.append("Best so far: %s at %.3fx." % (best.get("status"), best.get("speedup", 0.0)))
    if hints:
        lines.append(hints)
    lines.append("Reply with exactly one JSON action.")
    return "\n".join(lines)