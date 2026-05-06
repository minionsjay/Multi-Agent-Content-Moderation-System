# Project Rules for Coding Agents

## 1. Authoritative Docs

Use this root Markdown set only:

- `README.md`: project entry and unified architecture.
- `PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md`: product scope and acceptance.
- `SDD_PROTOTYPE_DEVELOPMENT_PLAN.md`: SDD planning, data contracts, detailed interface guidance.
- `AGENT_ARCHITECTURE.md`: Agent responsibilities, collaboration flow, and boundary constraints.
- `PROTOTYPE_SPEC_INDEX.md`: direct execution entry for Coding Agents.
- `AGENTS.md`: these rules.

Legacy architecture or overview content has been merged into the unified docs. Do not resurrect old entry points or conflicting decision enums.

## 2. Prototype Development Entry

For implementation tasks, use `PROTOTYPE_SPEC_INDEX.md` as the primary execution entry.

It defines:

- the two development modules;
- exact Spec IDs;
- implementation order;
- shared JSON / JSONL contracts;
- integration checkpoints G0-G6.

`SDD_PROTOTYPE_DEVELOPMENT_PLAN.md` is background and detail, not the direct task source. Read it only when the current Spec requires field-level detail or acceptance clarification.

## 3. Current Project Direction

- Keep the existing technical entry: `python -m src.api`.
- Preserve the already working POC capabilities unless a Spec explicitly asks to adapt them.
- Image moderation follows the current project approach: convert image content to text signals, then run text recognition/moderation logic. Do not introduce a multimodal LLM as the current core dependency.
- Online decisions use `pass / block / review`.
- `fraud` is the first Prototype example risk type; do not hard-code the whole system so only fraud can work.
- Business keyword/rule assets already exist. Do not invent a rule optimization product unless a Spec explicitly asks for a debug or replay interface.
- Model training is not implemented in this Prototype. Only reserve a manual or scheduled trigger interface.

## 4. Task Granularity

Implement one Spec ID or one integration checkpoint at a time.

Good task examples:

- Implement `A2`: rule loading and rule detection.
- Implement `A12`: image-to-text signal integration.
- Implement `B7`: offline replay evaluator.
- Implement `B9`: training trigger interface.
- Run integration checkpoint `G3`: human review feedback integration.

Avoid broad tasks such as:

- Implement the whole prototype.
- Follow the SDD document and build everything.
- Rewrite the existing POC architecture.

## 5. Integration Rule

Modules must collaborate through the shared JSON / JSONL contracts listed in `PROTOTYPE_SPEC_INDEX.md`.

Do not introduce hidden coupling between Module A and Module B internals.
