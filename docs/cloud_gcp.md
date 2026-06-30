# Deploying on Google Cloud

The local stack (`docker/docker-compose.yml`) maps one-to-one onto GCP. The
container is the unit of deployment; only the base image and the data/artifact
locations change.

## Mapping

| Local (compose)                     | GCP equivalent                                  |
|-------------------------------------|-------------------------------------------------|
| `app` image (`docker/Dockerfile`)   | Image in **Artifact Registry**                  |
| CPU torch wheel                     | CUDA base + cuda torch on a **GPU Vertex job**  |
| `data/interim/*.tif`, `outputs/`    | **Cloud Storage (GCS)** buckets (`gcsfuse` or `gs://` via rasterio `/vsigs/`) |
| `postgis` service                   | **Cloud SQL for PostgreSQL** + PostGIS extension |
| `mlflow` service (sqlite)           | **MLflow on Cloud Run** backed by Cloud SQL + GCS artifact store |
| `scripts/run_all_events.sh`         | **Vertex AI custom training job** (one per AOI) |
| `scripts/benchmark_inference.py`    | **Batch inference**: Cloud Run jobs or a Vertex batch-prediction over GCS tiles |

## Batch inference at scale

The benchmark (`docs/figures/14_scale_benchmark.png`) projects ~600 GPU-hours to
map an 80 Mha state at 0.5 m on one worker. The platform pattern:

1. **Tile** the imagery in GCS (`src/data/tiling.py` logic) into a work queue.
2. **Fan out** Cloud Run jobs / a Vertex batch-prediction; each worker pulls a
   tile, runs `src/inference/batch_infer.py:tiled_event_inference`, writes a
   prediction COG back to GCS. 100 workers → the state in ~6 h.
3. **Mosaic** (`src/geospatial/mosaic.py`) the tiles into a deliverable COG.
4. **Vectorise + load** to Cloud SQL/PostGIS (`src/geospatial/`) for the
   management-unit roll-up the client consumes.

## Reproducibility & monitoring

- Every raster already carries a provenance sidecar (git SHA, STAC IDs, CRS).
- `src/utils/tracking.py` logs params/metrics/artifacts to MLflow — point
  `MLFLOW_TRACKING_URI` at the Cloud Run MLflow server.
- The image is built and smoke-tested in CI; tag images by git SHA in Artifact
  Registry so every Vertex job is pinned to an exact commit.

## CI/CD

`.github/workflows/ci.yml` lints, tests, and (the added `docker-build` job)
builds the image. Extend with a `gcloud builds submit` + `gcloud run deploy`
step on tags to ship to Cloud Run / Vertex automatically.
