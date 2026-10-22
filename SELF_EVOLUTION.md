# What "self-evolution" means in this project

> Object: `kernel-agent` — an operator optimization agent with an RSI strategy pool.
>
> Conclusion first: it can already **get better at a single task** and **reuse a
> strategy across tasks**. It does not yet **decide what to practice next**,
> **generate new tasks**, **keep diverse solutions**, or **compile a winning move
> into an executable skill**. Self-evolution is not fancier prompting; it is
> changing the distribution of future practice.

---

## 0. Start with the audit

```bash
python -m rsi.evolve audit
```

The audit scans `runs/agent/*`, the strategy pool, exemplars and the task suite,
and reports:

```
runs: N | valid performance runs: N | mechanics-only: N
tasks in suite: N | never run: N | runs without valid evidence: N
strategies: N | with evidence: N | with valid evidence: N
exemplars: N | combos: N
memory: facts N | open failures N | verified skills N
```

Two rules make the numbers meaningful:

1. A CPU or replay run is a **mechanics test**, not a performance result. If it
   claims `hardware=ascend-910b` while `device=cpu`, the audit flags it as
   mislabelled.
2. A strategy without evidence is a hypothesis, not knowledge. Confidence only
   moves after measured runs, and only in the direction the evidence supports.

The point of the audit is to prevent the most common failure mode of "self-
improving" agents: a knowledge base that grows while capability stays flat.

---

## 1. The four capabilities, and where this repo stands

| Capability | What it needs | Current state |
|---|---|---|
| Improve on one task | trajectory + feedback + iteration | yes (agent loop) |
| Reuse across tasks | conditions + retrieval + narrowing | yes (strategy pool) |
| Decide what to practice next | capability matrix + curriculum scoring | partial (`rsi.evolve plan`) |
| Generate new tasks | task synthesis from failure signatures | no |
| Keep diverse solutions | exemplar archive with novelty | partial (`rsi.exemplars`) |
| Compile wins into skills | template extraction + validation | partial (`rsi.skills`) |

The interesting part is not the rows marked "yes". It is the gap between "reuse"
and "decide": a system that only reuses priors will keep solving the tasks it
already knows. A curriculum that scores *family coverage*, *transfer* and
*evidence gaps* is what changes the practice distribution.

---

## 2. Curriculum scoring

`rsi.evolve plan` scores each task by:

- **family coverage** — how many distinct op families have evidence;
- **transfer** — whether a family is shared by several tasks, so a win
  transfers;
- **evidence gap** — tasks with no valid run or no verified strategy;
- **difficulty penalty** — level 2 costs more than level 1;
- **expected headroom** — prior speedup potential.

The output is a ranked list, not an automatic schedule. The human (or an outer
agent) still chooses. That boundary is deliberate: an agent that can silently
change its own practice distribution is hard to audit.

---

## 3. Provenance is the load-bearing wall

The RSI loop is only as good as its evidence rules. Three of them matter:

1. **Device provenance.** A run records the device it actually used. A CPU run
   cannot be filed under an NPU scope.
2. **Scope matching is hard.** `soc / cann / bisheng / triton / dtype / form /
   op / shape` must match before a prior is injected. No match means no prior.
3. **Lifecycle.** Facts, failures and skills decay if not re-verified. A stale
   fact is marked `stale`, not silently kept.

Shape features (`half_is_1`, `half_aligned_32b`, `outer_bucket`, `prime_outer`)
exist because the same tile size can have opposite effects on different shapes.
Without them, "block=4096 is faster" becomes folklore that is true half the time.

---

## 4. What the next iteration should add

If you want to take this further, the order that has the best return:

1. **Task synthesis from failure signatures.** Turn a minimal repro into a new
   task. This directly expands the practice distribution.
2. **Exemplar diversity.** Keep the top-k solutions per task by *behavioral*
   diversity, not only speed, so the agent sees more than one winning shape.
3. **Skill compilation.** Promote a verified solution into a parameterized
   template with preconditions; require two independent passes before
   `verified`.
4. **Calibration.** Track predicted vs measured speedup per strategy. A strategy
   that is consistently over-optimistic should lose retrieval weight even if it
   sometimes wins.

None of these require a bigger model. They require better bookkeeping.

---

## 5. Anti-goals

- Do not let the agent edit the evaluator.
- Do not record a mechanics run as a hardware result.
- Do not promote a skill from a single pass.
- Do not inject a prior whose scope does not match.
- Do not optimize the audit metrics instead of the task.

The last one is subtle: if the curriculum rewards "tasks with evidence", an agent
can generate easy tasks to fill the matrix. That is why the plan scores transfer
and headroom, not raw counts.