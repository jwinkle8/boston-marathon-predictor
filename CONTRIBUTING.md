# Contributing

Thanks for working on the Boston Marathon Finish Time Predictor. This guide keeps our
history readable, our reviews fast, and our deployed API and app stable. Keywords like
**must** and **should** are meant literally: *must* is required, *should* is the default
unless you have a good reason.

## Table of contents

1. [Ground rules](#ground-rules)
2. [Getting set up](#getting-set-up)
3. [Branching](#branching)
4. [Commits](#commits)
5. [Pull requests](#pull-requests)
6. [Code review](#code-review)
7. [Merging](#merging)
8. [Code and ML conventions](#code-and-ml-conventions)
9. [Data, secrets, and reproducibility](#data-secrets-and-reproducibility)
10. [Issues](#issues)

## Ground rules

- `main` is always deployable. The Cloud Run API and Streamlit app are built from it.
- Nobody commits directly to `main`. All changes go through a pull request (PR).
- Nobody merges their own PR without at least one approval from a teammate.
- Keep changes small and focused: one purpose per branch and per PR.
- Be respectful and assume good intent. Critique code, not people.

## Getting set up

```bash
git clone https://github.com/jwinkle8/aipi-module-3.git
cd aipi-module-3
python -m venv .venv
# macOS/Linux: source .venv/bin/activate    Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The Streamlit app has its own dependency list in `app/requirements.txt`. See the
[README](README.md) for how to run the pipeline, API, and app.

## Branching

Always branch from an up-to-date `main`:

```bash
git switch main
git pull origin main
git switch -c feat/short-description
```

### Naming convention

```
<type>/<short-kebab-case-description>
```

or, when an issue exists, include its number:

```
<type>/<issue-number>-<short-kebab-case-description>
```

| Type | Use for | Example |
|---|---|---|
| `feat` | New functionality | `feat/age-group-feature` |
| `fix` | Bug fixes | `fix/12-api-null-split` |
| `data` | Data download/cleaning changes | `data/handle-missing-splits` |
| `model` | Training, features, hyperparameters, evaluation | `model/gradient-boosting` |
| `docs` | Documentation only | `docs/update-readme-metrics` |
| `refactor` | Restructuring without behavior change | `refactor/split-train-module` |
| `test` | Adding or fixing tests | `test/api-validation` |
| `chore` | Dependencies, config, tooling, CI | `chore/bump-fastapi` |
| `hotfix` | Urgent fix for the deployed service | `hotfix/cloud-run-timeout` |

Rules:

- Use lowercase letters, numbers, and hyphens only. No spaces, underscores, or personal
  names (`ben-stuff` is not acceptable).
- Keep names short (about 3-5 words) but descriptive.
- Delete the branch after it is merged.
- Branches should be short-lived. If a branch is more than about a week old, rebase on
  `main` and consider splitting the work.
- Do not reuse a branch after its PR is merged; start a new one.

## Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <summary>

<optional body: what and why, not how>

<optional footer: Closes #12>
```

- Types match the branch types above (`feat`, `fix`, `data`, `model`, `docs`,
  `refactor`, `test`, `chore`).
- Scope is optional and usually a module: `api`, `app`, `train`, `clean`, `features`,
  `docker`, `config`.
- The summary is imperative, lowercase, has no trailing period, and is 72 characters or
  fewer: `fix(api): reject negative split times`.
- Explain *why* in the body when it is not obvious from the diff.
- Make atomic commits: each one should leave the project in a working state.
- Never commit generated artifacts, large data files, or secrets (see
  [Data, secrets, and reproducibility](#data-secrets-and-reproducibility)).
- Avoid vague messages such as `update`, `fix stuff`, or `wip` in anything that goes to
  `main`.

Good examples:

```
feat(features): add pace-drop ratio between 10K and half
fix(clean): drop rows with non-numeric split values
chore(deps): bump scikit-learn to 1.9.1
```

## Pull requests

### Before you open one

- [ ] Branch is rebased on or merged with the latest `main` and has no conflicts.
- [ ] The code runs locally (pipeline, API, or app, whichever you touched).
- [ ] The Docker image still builds if you changed `Dockerfile`, `requirements.txt`, or
      anything the API imports.
- [ ] Documentation (README, comments, `config.yaml` notes) is updated where needed.
- [ ] No secrets, credentials, personal data, or large files are included.
- [ ] You have reviewed your own diff first.

### Title

Use the same format as a commit summary:

```
feat(train): log feature importances to MLflow
```

### Description

Every PR should include:

```markdown
## Summary
What does this change and why? Link the issue (Closes #N).

## Changes
- Bullet list of the main changes

## How to test
Exact commands or steps a reviewer can run.

## Results (for model/data changes)
Metrics before vs. after (MAE, RMSE, R²), dataset version, MLflow run link or ID.

## Screenshots (for UI changes)
Before / after.

## Notes for reviewers
Risky areas, trade-offs, follow-ups, or anything you want a closer look at.
```

### Size and scope

- Aim for under roughly 400 changed lines. Larger PRs should be split or explained.
- Do not mix refactors, formatting, and behavior changes in a single PR.
- Open a **Draft PR** early if you want feedback on direction. Mark it *Ready for
  review* only when the checklist above is done.
- Do not force-push to a branch while a reviewer is mid-review without telling them.
  Prefer new commits during review; they are squashed at merge.

## Code review

### Expectations for authors

- Request review from at least one teammate. Respond to review comments within one
  working day where possible.
- Reply to every comment, either with a change or a reason. Do not silently ignore one.
- The **reviewer** resolves a conversation when satisfied, unless it was a trivial
  suggestion you applied.
- Do not take feedback personally; it is about the code.

### Expectations for reviewers

- Aim to review within one working day.
- Check for: correctness, clarity, tests or manual verification, config and data
  impact, security, and consistency with this guide.
- Pull the branch and run it for anything touching the pipeline, API, or app.
- Approve when the change is good enough to merge. Perfection is not required, but
  `main` must stay healthy.

### Comment conventions

Prefix comments so intent and urgency are clear:

| Prefix | Meaning | Blocks merge? |
|---|---|---|
| `blocking:` | Bug, security issue, or break from our rules. Must be fixed. | Yes |
| `question:` | I need to understand this. Might lead to a change. | Until answered |
| `suggestion:` | Recommended improvement; author decides. | No |
| `nit:` | Minor style or preference. | No |
| `praise:` | Something done well. | No |
| `thought:` | Idea for later or a different approach. | No |

Writing good review notes:

- Be specific and actionable: point at the line, explain why, and suggest a fix.
- Ask questions instead of making demands: "What happens if `split_half` is 0?"
- Explain the reasoning behind your preference; link docs when helpful.
- Keep it kind and concise. Use `praise:` sincerely and often enough to matter.
- Use suggestion blocks for small fixes so the author can apply them in one click.

Example:

```
blocking: `predict()` will raise a KeyError if `gender` is lowercase. Can we
normalize with `.upper()` before the lookup, or validate in the request model?
```

## Merging

- Required: at least **1 approval**, all `blocking:` and `question:` comments resolved,
  and no merge conflicts.
- Use **Squash and merge** by default so `main` has one clean commit per PR. The squash
  commit message should follow the commit format above and include the PR number.
- Use a regular merge commit only when preserving the individual commits is meaningful
  (for example, a multi-part data migration), and agree on it in the PR.
- Do not merge feature branches into each other. Merge into `main`, then rebase.
- After merge, delete the branch and confirm any deployment (Cloud Run, Streamlit)
  still works.

### Hotfixes

For urgent production problems: branch `hotfix/<description>` from `main`, keep the fix
minimal, request an expedited review, and note the cause and follow-up work in the PR.

## Code and ML conventions

- **Python style:** follow [PEP 8](https://peps.python.org/pep-0008/), 4-space
  indentation, and type hints on new public functions. Use descriptive names:
  `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for constants.
- **Docstrings and comments:** short docstring on public functions; comment the *why*,
  not the *what*.
- **Configuration over constants:** pipeline settings belong in `config.yaml`, not
  hard-coded in scripts.
- **Dependencies:** pin exact versions in `requirements.txt`. Explain any new
  dependency in the PR, and keep `app/requirements.txt` limited to what the Streamlit
  app needs.
- **Model changes:** every model or feature change should be tracked in MLflow and
  report metrics against the current baseline in the PR. Do not claim an improvement
  without comparing on the same data split.
- **API changes:** keep the request/response schema backward compatible with the
  Streamlit app, or update both in the same PR (or coordinate the release order in the
  description).
- **Tests:** add or update tests for bug fixes and new logic where practical. Describe
  manual checks in the PR when automated tests are not feasible.

## Data, secrets, and reproducibility

- **Never commit secrets:** API keys, GCP service account JSON, tokens, `.env` files.
  Use environment variables or the platform's secret store. If you commit one by
  accident, tell the team immediately and rotate it; deleting the commit is not enough.
- **Do not commit raw or large data files.** Data lives in Google Cloud Storage and is
  referenced by version in `config.yaml`.
- **Respect privacy:** cleaned data must not include names, bib numbers, or cities.
  Do not reintroduce them.
- **Avoid leakage:** never use features derived from the finish time (pace, place,
  projected time) as model inputs.
- **Reproducibility:** set random seeds, and record the data version, config, and
  package versions used for any reported result.
- **Respect licenses:** note the source and license for any new dataset or
  third-party code you add.

## Issues

- Search existing issues before opening a new one.
- Use a clear title and include steps to reproduce, expected vs. actual behavior, and
  environment details for bugs.
- Reference issues from commits and PRs with `Closes #N` / `Refs #N`.
- Assign yourself when you start work, and pick up unassigned issues by commenting
  first.

## Questions

Not sure about something here? Open an issue or ask in the team channel. If the guide
itself is out of date or unclear, send a `docs/` PR to improve it.
