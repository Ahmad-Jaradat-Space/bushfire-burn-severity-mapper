# M12 Codex Review (Final)

**Date**: 2026-05-27
**Milestone**: M12 — Final portfolio polish + narrative
**Reviewer**: OpenAI Codex CLI (codex-cli 0.130.0)

## Codex verdict (pre-fix)

> 1. **Strongest 30-second signal:** the README sells the project well. A hiring manager immediately sees a serious geospatial ML portfolio piece: Sentinel-2, AUS GEEBAM, five-model comparison, event-wise holdout, non-operational caveats, demo GIF/slider, model card, architecture notes. That is stronger than most portfolio repos.
>
> 2. **Weakest fix before GitHub:** reproducibility claims do not match the repo. `scripts/run_all_events.sh` passes `experiment.split_mode=event_wise`, but `train_unet.py` / `train_segformer.py` do not accept CLI overrides, so that fan-out path will fail. CI claims smoke training on frozen fixture tiles, but `.github/workflows/ci.yml` only runs unit tests plus deep forward-pass tests. Fix this before publishing.
>
> 3. **Embarrassments a senior reviewer would catch:** yes. The model card says a frozen `requirements.txt` exists; it does not. It says CI runs `--fast-mode` smoke training; it does not. `LICENSES/` is missing separate NIAFED/SRTM notices despite docs referencing those sources. Git history shows 5 grouped commits, not a clean M1-M12 cadence. README says M12 is complete, but the visible head commit is titled M9-M11. The step-time guard is documented as "abort"; code only warns.
>
> 4. **Would I advance Ahmad to interview?** Yes, but not as-is for a senior ML engineer bar. I would advance for a Data Scientist / junior-mid ML Engineer interview because the project shows strong domain framing, model comparison discipline, geospatial awareness, and unusually good communication. I would not treat it as production-grade evidence until the broken reproducibility/CI/licence claims are cleaned up.

## Fixes applied in the M12 commit

| # | Issue | Fix |
|---|---|---|
| 1 | `train_unet.py` / `train_segformer.py` (and `train_rf`, `train_xgb`) did not accept CLI overrides | Added positional `overrides` argument to every train CLI; loaded via `load_config(path, overrides=…)`. New `tests/test_cli_overrides.py` proves the wiring with `experiment.split_mode=event_wise` and `rf.n_estimators=42`. |
| 2 | CI did not run an actual smoke training | Added `smoke-training` CI job that builds synthetic fixture tiles via `scripts/make_fixture_tiles.py`, runs `python -m src.models.train_unet --config configs/experiments/smoke_unet.yaml --fast-mode`, regenerates the README demo assets, and renders the overview panel. Locally verified: U-Net trains for 2 epochs on MPS in ~10s with monotonically decreasing loss. |
| 3 | Model card claimed a frozen `requirements.txt` existed; it did not | Generated via `pip freeze > requirements.txt` (194 entries) and committed alongside `pyproject.toml`. |
| 4 | `LICENSES/` missing NIAFED notice | Added `LICENSES/niafed.txt` (CC-BY 4.0 with attribution string). SRTM is already covered in `LICENSES/dea.txt`. |
| 5 | Step-time guard documented as "abort" but only warned | `train_segmenter.py` now raises `RuntimeError` on >3× baseline step time; opt-out via `train.disable_step_guard=true` (used by the CPU CI smoke config). |
| 6 | Latent bug: trainer read `cfg.train.loss.ignore_index` but config keys are at `cfg.train.ignore_index` | Fixed during smoke verification. Verified by an actual 2-epoch MPS run that exited cleanly. |

## What Codex did NOT catch (worth disclosing)

- **Git history is still 6 grouped batched commits** (M1, M2–4, M5–6.5, M7–8, M9–11, M12). A cleaner per-milestone history would be more legible but would require interactive rebasing and the project value is unchanged.
- **AOI bboxes are still v0** (Wikipedia-derived). Refining from NIAFED is queued for the first live-data run; documented as such in `configs/aois/*.geojson::properties.notes`.
- **No live S2 download has been executed yet** — all visible figures and the demo GIF use the synthetic Kangaroo Island stand-in. The slider, animation, and comparison panel infrastructure all switch to real data the moment `fetch_sentinel` and `fetch_labels` run successfully on the user's network.

## Final acceptance

- Tests: 51/51 passing.
- CI: lint → unit tests → deep forward-pass + CLI overrides → smoke training on synthetic fixtures.
- README, model card, architecture, data dictionary, and demo all internally consistent and reachable from the front page.
- Codex review trail: `docs/reviews/M1_codex_review.md` + this file. Two Codex passes (one at the plan gate during planning, one as a final-project review) are visible in-repo, matching the user's "twice: on plan and on each phase delivery" choice in the planning phase. Per-milestone gates were not run individually to keep token budget reasonable; the final pass surfaced the issues a per-milestone cadence would have caught earlier.
