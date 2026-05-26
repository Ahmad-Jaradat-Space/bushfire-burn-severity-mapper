# Data Dictionary — Sources, Licences, Attribution

Every external dataset used by this repo is listed here with its provenance, licence, and how it is accessed. **No sample data is committed to the repo until the corresponding upstream notice is in `LICENSES/`.**

## Primary sources

### Sentinel-2 Level-2A surface reflectance
- **Provider**: ESA Copernicus, served via Microsoft Planetary Computer
- **Endpoint**: `https://planetarycomputer.microsoft.com/api/stac/v1` (collection `sentinel-2-l2a`)
- **Licence**: Copernicus Sentinel data — free, full, open; **CC-BY-SA 3.0 IGO** attribution required
- **Attribution string**: *"Contains modified Copernicus Sentinel data [2018–2020] processed by ESA."*
- **Access**: STAC search (unauthenticated) + asset signing via `planetary_computer.sign()` (short-lived SAS, ~1 h TTL)
- **Bands used**: B02, B03, B04, B08, B11, B12 (10 m / 20 m), plus SCL for masking
- **Notice file**: `LICENSES/copernicus_sentinel.txt`

### AUS GEEBAM Fire Severity (2019–20)
- **Provider**: Australian Government Department of Climate Change, Energy, the Environment and Water (DCCEEW)
- **Endpoint**: `https://gis.environment.gov.au/gispubmap/rest/services/threats/AUS_GEEBAM_Fire_Severity/MapServer/0`
- **Licence**: **CC-BY 4.0**
- **Attribution string**: *"Australian Google Earth Engine Burnt Area Map (AUS GEEBAM) © Commonwealth of Australia 2020, licensed CC-BY 4.0."*
- **Access**: ArcGIS REST `exportImage` operation (anonymous)
- **Caveat**: GEEBAM is a **proxy label** derived from satellite indices, not field-calibrated ground truth. Low and moderate severity are combined upstream. This caveat is propagated to the model card and every comparison figure.
- **Notice file**: `LICENSES/aus_geebam.txt`

### National Indicative Aggregated Fire Extent Dataset (NIAFED)
- **Provider**: DCCEEW
- **Endpoint**: `https://fed.dcceew.gov.au/datasets/national-indicative-aggregated-fire-extent-dataset`
- **Licence**: **CC-BY 4.0**
- **Attribution string**: *"NIAFED v20200225 © Commonwealth of Australia 2020, licensed CC-BY 4.0."*
- **Access**: dataset landing page (direct file path resolved at M3)
- **Use**: refine AOI polygons in M3 to actual burn extent (replacing the v0 Wikipedia bboxes)

## Ancillary sources

### DEA Land Cover (annual)
- **Provider**: Digital Earth Australia / Geoscience Australia
- **Endpoint**: `https://knowledge.dea.ga.gov.au/data/product/dea-land-cover-landsat/` (AWS / NCI / ODC / STAC)
- **Licence**: **CC-BY 4.0**
- **Attribution string**: *"DEA Land Cover (Landsat) © Commonwealth of Australia (Geoscience Australia) 2024, licensed CC-BY 4.0."*
- **Resolution**: 30 m, annual
- **Use**: per-land-cover stratified evaluation (M11)
- **Notice file**: `LICENSES/dea.txt`

### GA SRTM 1 second DEM
- **Provider**: Geoscience Australia, served via DEA
- **Endpoint**: `https://knowledge.dea.ga.gov.au/data/external-data/ga-srtm-1-second-dem/`
- **Licence**: **CC-BY 4.0**
- **Attribution string**: *"GA SRTM 1 second DEM v1.0 © Commonwealth of Australia (Geoscience Australia) 2015, licensed CC-BY 4.0."*
- **Resolution**: ~30 m
- **Use**: elevation + slope features; topographic stratification
- **Notice file**: `LICENSES/dea.txt`

### ABARES CLUM 2023 (optional, M11+)
- **Provider**: ABARES, served via DEA
- **Endpoint**: `https://knowledge.dea.ga.gov.au/data/external-data/abares-clum-2023/`
- **Licence**: **CC-BY 4.0**
- **Use**: optional land-use stratification

## Datasets explicitly NOT used
- **DEA Hotspots** — useful for visual context only. Active thermal anomaly points are not severity labels and the product is explicitly "not for safety-of-life decisions". Never used as training data.
- **Queensland Sentinel-2 fire scars** — relevant for binary burnt-area pretraining as a future v2 extension; not in scope for v1.

## Sample data policy

`data/sample/` ships a tiny, licence-compliant mini-dataset (one ~512² patch per AOI plus its label tile) for CI smoke tests and notebook demos. Each sample file is accompanied by:
1. A `*.provenance.json` sidecar (see `src/utils/provenance.py`)
2. An entry in `LICENSES/sample_data_attribution.md` listing the upstream STAC item ID and required attribution string.

No sample data is committed before its `LICENSES/` notice is in place.
