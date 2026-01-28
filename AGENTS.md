# Agent Instructions

These guidelines keep model code, data, and results cleanly separated.

## Structure rules
- **Model code lives only in model folders** (one folder per model/approach).
  Examples: `FASTcode/`, `DeepCAD/`, `legacy-custom-wizard-plugin-no-denoise-abandonded-for-now/`.
- **Our own training data and results live at repo root** in dedicated folders
  (e.g., `training_data/`, `long-batch/`, `short-test/`, `results-from-FASTcode/`,
  `results-from-DeepCAD/`, `result/`, `analysis/`, `figures/`).
- **Do not place our training data or results inside model folders.**
- **Original datasets distributed with a model repo stay within that repo.**

## Change discipline
- Model code changes should not move or modify data/results.
- If new outputs are generated, store them under a top-level results folder (e.g., `results-from-FASTcode/`).
- Keep `.gitignore` updated for large or transient outputs; preserve curated examples.

## When unsure
- Prefer adding a new top-level folder instead of nesting inside a model folder.
- Ask for confirmation before moving any existing data/results.
