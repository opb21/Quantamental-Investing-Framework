# CLAUDE.md

Guidance for Claude Code when working in this repository.

This project is a **systematic investing research framework** designed for long-horizon equity strategies. The codebase will evolve significantly over time, so guidance here focuses on **principles and invariants rather than specific file structures**.

---

# Core Principles

## Modularity

The system is designed as a modular research pipeline.

Typical stages include:

Config → Universe → Data → Signals → Portfolio → Backtest → Analytics → Outputs

Modules should remain loosely coupled and composable.

Avoid merging responsibilities across layers.

---

## Prefer Simple Functional Code

Unless explicitly requested:

- Prefer **pure functions** over classes
- Keep functions small and composable
- Avoid unnecessary abstractions
- Avoid introducing frameworks

This repository prioritizes **clarity and inspectability for research code**.

---

## Minimal Dependencies

Do not add new dependencies unless explicitly requested.

The framework intentionally uses a small core stack (e.g. pandas, numpy, pytest) to keep environments reproducible.

---

## Stable Interfaces

When modifying existing code:

- Preserve function signatures unless instructed otherwise
- Avoid renaming modules or moving files unless requested
- Maintain backward compatibility where possible

---

# Research Integrity Rules

This framework prioritizes **robust research practices over maximizing backtest performance**.

Claude must respect the following principles.

## No Lookahead Bias

Signals and portfolio decisions must only use information available at the time of decision.

In practice:

- Portfolio returns must use **lagged weights**
- Signals must use **historical data only**
- Rankings must not include future information

---

## Explicit Data Alignment

Financial pipelines often break due to silent index misalignment.

When manipulating pandas objects:

- ensure indices align explicitly
- avoid silent broadcasting errors
- validate shapes where appropriate

---

## Reproducibility

Backtests should be reproducible.

Runs may generate artifacts such as:

- configuration used
- performance metrics
- portfolio weights
- return series
- signal outputs

These artifacts may be used to compare strategies and support qualitative analysis.

---

# Testing

The project uses **pytest**.

Guidelines:

- Add tests for new logic where appropriate
- Do not remove tests unless instructed
- Update tests when modifying behavior

Run tests with:

pytest

---

# Working With This Codebase

When implementing changes:

1. Modify only the files requested.
2. Avoid large architectural refactors unless asked.
3. Explain assumptions in financial logic.
4. Prefer readable code over clever code.

---

# Non-Goals

Claude should **not** attempt to:

- optimise strategies
- tune parameters
- propose alpha signals
- redesign architecture without instruction

The human developer controls research decisions.

---

# Summary

This repository is a **long-term research framework**, not a short-term trading script.

Priorities:

- clarity
- modularity
- reproducibility
- research integrity

# Hook Behavior

The following behaviors should be treated as default workflow rules when making changes.

## Pre-change checks

Before implementing new code:

1. Search the repository for similar functions, modules, or logic.
   - Prefer extending or updating existing functionality over creating duplicates.
   - Avoid introducing parallel versions of the same concept.

2. Confirm change scope.
   - Only modify the files explicitly requested.
   - If additional files appear to require changes, explain why before proceeding.

3. Check interface consistency.
   - If a requested change touches an existing function, inspect its current signature and expected return shape before editing.
   - Preserve interfaces unless explicitly asked to change them.

## Post-change checks

After modifying code:

1. Search for usages of any changed function, class, or module.
   - Identify imports, call sites, and assumptions that may now be broken.
   - If task scope allows, update those usages.
   - Otherwise, report them clearly.

2. Run lightweight validation where feasible.
   - Prefer running pytest first.
   - If pipeline code changed, also consider whether the main entrypoint should be run.

3. Check research integrity.
   - If signal, portfolio, or backtest logic changed, explicitly verify:
     - no lookahead bias has been introduced
     - data alignment remains explicit
     - output reproducibility is preserved

4. Report risks clearly.
   - If any change could affect downstream behavior, note the likely impact rather than silently assuming everything is fine.