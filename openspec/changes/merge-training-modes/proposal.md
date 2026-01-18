# Change: Merge Training Modes

## Why

Currently, "Normal Training" (offline script/initial training) and "Rolling Training" (UI triggered update) are treated as separate features. Their core functionality is identical: training a model on a dataset. Merging them into a single "Model Training" feature simplifies the User Experience and consolidates implementation logic.

## What Changes

- Combine "Normal Training" and "Rolling Update" into a unified "Train Model" workflow.
- Update UI to provide a single entry point for training, allowing configuration of parameters (symbols, interval) that serves both use cases.
- Remove the distinction between "initial training" and "rolling" in the spec, treating both as training tasks.

## Impact

- Affected specs: `qlib-ml-strategy`
- Affected code: `autotrade/ui/templates/models.html`, `autotrade/web_server.py` (API endpoints), `autotrade/research/models/trainer.py`
