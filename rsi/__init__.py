"""RSI layer: trajectories, strategy pool, knowledge base, retrieval priors.

This is the recursive-self-improvement part of the project:

    run -> attempts.jsonl (trajectory) -> distill -> strategy pool + combos
        -> retrieve (condition match) -> tuning priors -> next run's agent

Nothing here is hardware-specific: strategies carry applicability conditions
(hardware / op class / dtypes) that get narrowed by measured evidence.
"""