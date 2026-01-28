# Calcium Analysis & Regeneration

This repo is organized to keep **model code**, **data**, and **results** separate so that changes to models never affect data storage or analysis artifacts.

## Repo layout (convention)
- **Model code (one folder per model/approach)**  
  - `FASTcode/`  
  - `legacy-custom-wizard-plugin-no-denoise-abandonded-for-now/`  
  - Add new models as their own top-level folders.

- **Data (inputs and curated datasets)**  
  - `training_data/`  
  - `long-batch/`  
  - `short-test/`

- **Results (outputs, reports, and figures)**  
  - Stored under dedicated repo-level folders (keep separate from model code).  
  - Current: `results-from-FASTcode/`  
  - Planned: `results-from-DeepCAD/`  
  - Examples: `result/`, `analysis/`, `figures/` (create as needed).

- **Tools / integrations**  
  - `Fiji/`

## Workflow notes
- Keep model code self-contained inside its folder.  
- Store data and results at the repo root in dedicated folders, such as `results-from-FASTcode/`.  
- Avoid mixing results inside model code folders.  
- Update `.gitignore` for large or generated outputs; keep curated examples checked in.

## Data & results policy
- Put new outputs under `results/` (preferred) or `analysis/` as appropriate.
- If outputs are large or transient, add them to `.gitignore` unless explicitly curated.
- Keep `results-from-FASTcode/` for archived FASTcode runs only.

## Getting started
See each model folder’s `README.md` for model-specific instructions.
