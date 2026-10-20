---
name: project-spec-workflow
description: Plan, document, review, and track software changes with traceable requirements, acceptance criteria, design decisions, and implementation tasks. Use when a repository maintains specification groups; adapt to its existing documentation and version-control conventions.
---

# Specification Workflow

Use this skill when work needs explicit scope, behavior, design choices, implementation tasks, or acceptance evidence. Fit the workflow to the repository’s existing documentation layout and engineering practices. If it has persistent specification groups, update the relevant group; otherwise locate the authoritative docs and use a suitable existing location. Do not impose a particular directory tree or document set when the repository already has a convention.

## Discover the context

1. Find the repository root and read applicable agent instructions and contributor documentation.
2. Identify the project’s authoritative sources for product goals, external evidence, architecture, shared contracts, local specifications, and implementation. Their names and paths vary; follow the repository’s own ownership rules.
3. Read the relevant existing specification and inspect the implementation. Treat plans as intended behavior until verified in code or runtime evidence.
4. Check the working tree and current version-control context before editing. Preserve unrelated changes and follow the repository’s branch, review, and commit conventions.

## Pick the smallest useful path

- **Light:** one low-risk, local, reversible adjustment. Record purpose, scope, unchanged behavior, and observable acceptance in the matching group’s `proposal.md`; do not manufacture a full document set.
- **Standard:** a feature or defect with user-visible behavior. Maintain the matching group’s `proposal.md`, `spec.md`, `design.md` when there are meaningful technical choices, and `tasks.md`.
- **High risk / cross-boundary:** include every affected domain, explicit requirement-to-module/task/acceptance traceability, failure and rollback behavior, and relevant security, compatibility, data, operations, and hardware constraints.
- **Prototype:** state the unknown, evidence sought, success/failure and stop criteria, and disposable boundary. Record findings; convert validated conclusions into the durable spec before treating the prototype as product behavior.

If no suitable group exists and a grouped specification layout fits the repository, use `scripts/init_spec_group.py` with the repository’s specification root to scaffold one from the included templates. It refuses to overwrite an existing group. For an existing group, edit authoritative files in place. Do not create one folder per task or duplicate the baseline just to record a code change.

## Draft and reconcile

1. Summarize the caller, problem, observable outcome, scope, and non-goals. Separate evidence, user decisions, recommendations, `[待确认]` product choices, and `[假设]` technical judgments. Ask only about decisions that block safe scope, authorization, compatibility, data, or hardware commitments; keep non-blocking uncertainty explicit with an owner or validation path.
2. Where traceability is useful, write behavior as stable `REQ-xx` requirements with testable `AC-xx.n` acceptance conditions. Cover normal paths and relevant errors, boundaries, permissions, and compatibility. Link each requirement to its upstream decision and responsible module.
3. Design only the decisions needed to build and review the change: boundaries, APIs/data, state/error handling, security, operations, compatibility, and rollback. Explain material alternatives; state when a separate design is unnecessary.
4. Make each `T-xx` task independently checkable and map it to requirements and acceptance IDs. Keep tasks in `tasks.md`; use Git history for implementation history and review.
5. Before coding, compare proposal, spec, design, tasks, repository baselines, and code. Resolve contradictions and blocking unknowns. The included validator checks document presence, unresolved template markers, ID uniqueness, and REQ/AC/task links; it does not replace engineering review.
6. Implement in task-sized steps. Update task status and the owning specification when evidence changes a decision. Keep each durable fact in the repository’s designated authoritative source; avoid maintaining competing copies.
7. Verify the applicable acceptance conditions and record only checks actually run, with outcomes and evidence. For device/model work, name the hardware and versions and distinguish simulation from real-device results. Reconcile the specification and implementation in the same Git review; do not claim completion when acceptance evidence is missing.

## Optional scripts

- Scaffold a group: `python <skill-dir>/scripts/init_spec_group.py --root <repository-root> --name <domain> --title "<title>" --mode standard`; pass `--spec-root <path>` when the specification directory is not discovered automatically.
- Check a group: `python <skill-dir>/scripts/validate_spec_group.py <path-to-group>`.
- Adapt the included templates to the repository’s terminology and rules. Script success is structural validation, not approval or proof that the implementation works.

## Report

Report the documentation locations and status, affected implementation scope, acceptance evidence, checks not performed, synchronized authoritative sources, and unresolved risks. Do not imply user approval, deployment, or hardware validation without evidence.
