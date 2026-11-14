<!--
PR title format: <type>(<optional scope>): <summary>
Example: feat(train): log feature importances to MLflow
See CONTRIBUTING.md for details.
-->

## What

<!-- What does this PR change? -->

## Why

<!-- Motivation. -->

## Reviewer focus

<!-- What should the reviewer look at most carefully? -->

## Type of change

<!-- Keep the one that matches your branch type -->

- [ ] `feat` - new functionality
- [ ] `fix` - bug fix
- [ ] `data` - data download/cleaning
- [ ] `model` - training, features, hyperparameters, evaluation
- [ ] `docs` - documentation only
- [ ] `refactor` - no behavior change
- [ ] `test` - tests only
- [ ] `chore` - dependencies, config, tooling
- [ ] `hotfix` - urgent fix for the deployed service

## Changes

-

## How to test

<!-- Exact commands or steps a reviewer can run -->

```bash

```

## Results (for model/data changes)

<!-- Delete this section if not applicable -->

| Metric | Baseline | This PR |
|---|---|---|
| MAE | | |
| RMSE | | |
| R² | | |

- Dataset version / config used:
- MLflow run link or ID:
- Same data split as baseline? <!-- yes / no (explain) -->

## Screenshots (for UI changes)

<!-- Delete this section if not applicable. Before / after. -->

## Checklist

- [ ] Branch name follows `<type>/<short-kebab-description>`
- [ ] Branch is up to date with `main` and has no conflicts
- [ ] Code runs locally (pipeline, API, or app, whichever I touched)
- [ ] Docker image still builds (if I changed `Dockerfile`, `requirements.txt`, or API imports)
- [ ] Documentation (README, comments, `config.yaml` notes) is updated
- [ ] API changes are compatible with the Streamlit app, or both are updated here
- [ ] Settings live in `config.yaml`; new dependencies are pinned and explained
- [ ] No secrets, credentials, personal data, or large data files are included
- [ ] No finish-time-derived features are used as model inputs
- [ ] Raw data files were not hand-edited
- [ ] I reviewed my own diff
