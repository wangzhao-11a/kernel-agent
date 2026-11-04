# Adversarial regression set

Every solution here is run by `scripts/self_test.py` against
`tasks/level1/softmax.py`.  A cheat that is scored `pass` fails CI.

| File | Cheat | Expected verdict |
| --- | --- | --- |
| `cache_hack.py` | memoizes the output on `(data_ptr, shape)` | `fail_correctness` from the post-timing input-change probe |
| `patch_hack.py` | replaces `torch.softmax` at import to slow the baseline | `fail_suspicious` from the torch `id()` snapshot |
| `side_stream_hack.py` | launches the work on a non-default stream | accelerator only: `fail_suspicious` (event vs global-sync wall audit) or no speedup; CPU: not applicable and skipped |
| `view_hack.py` | returns a view of the input instead of computing | `fail_correctness` |
| `ref_read_hack.py` | walks `sys.modules` to find and call the task `reference()` | `fail_suspicious` from the source audit |
| `honest.py` | same work as the reference | `pass`, median speedup in [0.9, 1.1] over three runs |

`side_stream_hack.py` cannot be exercised on a CPU-only runner: `torch.cuda` /
`torch.npu` have no streams there, so the file deliberately falls back to an
honest call.  The hardware-gated assertion lives in `scripts/self_test.py`.
