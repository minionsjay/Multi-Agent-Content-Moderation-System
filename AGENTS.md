# Project Rules for Coding Agents

## Prototype Development Entry

For the content safety risk-control Agent prototype, do not use `SDD_PROTOTYPE_DEVELOPMENT_PLAN.md` as the direct task source.

Use `PROTOTYPE_SPEC_INDEX.md` as the primary execution entry. It defines:

- the two development modules;
- the exact Spec IDs;
- the implementation order;
- the shared data contracts;
- the integration checkpoints G0-G5.

`SDD_PROTOTYPE_DEVELOPMENT_PLAN.md` is background documentation only. Read it only when a specific Spec in `PROTOTYPE_SPEC_INDEX.md` asks for more detail, or when clarifying product intent.

## Task Granularity

Implement one Spec ID or one integration checkpoint at a time.

Good task examples:

- Implement `A2`: rule loading and rule detection.
- Implement `B8`: offline replay evaluator.
- Run integration checkpoint `G3`: human review feedback integration.

Avoid broad tasks such as:

- Implement the whole prototype.
- Follow the SDD document and build everything.

## Integration Rule

Modules must collaborate through the shared JSON / JSONL contracts listed in `PROTOTYPE_SPEC_INDEX.md`.

Do not introduce hidden coupling between Module A and Module B internals.

