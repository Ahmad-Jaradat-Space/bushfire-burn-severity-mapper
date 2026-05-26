# M1 Codex Review

**Date**: 2026-05-27
**Milestone**: M1 — Scaffold + governance (CRS, temporal, licence, provenance)
**Reviewer**: OpenAI Codex CLI (codex-cli 0.130.0)

## Verdict

Good M1 scaffold, but not actually milestone-complete until the M1 commit lands and the config loader exists.

## Findings

1. **Hard fail at review time: nothing committed.** Governance milestone requires a clean M1 commit visible in `git log`, not a working-tree dump.
2. **Governance decisions are reviewable, not hand-wavy.** CRS, temporal, licence, provenance all encoded in code + docs (not just prose):
   - CRS — `docs/architecture.md` §1, `configs/config.yaml::crs`
   - Temporal — `docs/architecture.md` §2, `configs/config.yaml::temporal_windows`, `configs/experiments/baseline_dnbr.yaml::experiment.temporal_mode`
   - Licence — `docs/data_dictionary.md`, `LICENSES/`
   - Provenance — `src/utils/provenance.py` + sidecar JSON proven via `python -m src.data.fetch_labels --event kangaroo_island_2019_2020`
3. **Missing: config composition / loader.** `baseline_dnbr.yaml` used Hydra-style `defaults:` but the plan chose OmegaConf and there was no loader. Would have forced every script to invent its own config semantics by M2/M3.
4. **Provenance schema too loose** — accepts arbitrary `inputs`/`extra` with no schema validation. Acceptable for M1; should become a typed contract by M3 before downstream modules accrete drift.
5. **AOI bboxes are crude v0.** Kangaroo Island OK (ocean-heavy but bounded). East Gippsland loose. Currowan may be too tight. Gospers plausible. All four must be refined from NIAFED before serious metrics — already scheduled in M3.
6. **Embarrassment risk**: README marked M1 ✅ without a commit; "Wikipedia bbox" attribution is weak — flagged as temporary.

## Fixes applied before closing M1

- Added `src/utils/config.py` with OmegaConf-based `load_config()` honouring `extends:` chains and CLI dot-overrides.
- Replaced Hydra-style `defaults:` in `configs/experiments/baseline_dnbr.yaml` with `extends: ../config.yaml`.
- Added `tests/test_config.py` (3 tests): root config loads, experiment extends + merges + strips `extends` key, CLI overrides work. All 15 tests now pass.
- Saving this review under `docs/reviews/M1_codex_review.md` and committing it alongside M1 deliverables.

## Deferred to later milestones

- **M3**: refine AOI polygons from the actual NIAFED 2019–20 footprint (replace Wikipedia bboxes).
- **M3**: harden provenance into a typed contract (dataclass or pydantic).
- **M3**: confirm GEEBAM `MapServer/0` is the severity raster by hitting `MapServer?f=json` at runtime.
