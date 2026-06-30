"""PostGIS round-trip for severity vectors.

Demonstrates the spatial-database half of the geospatial stack: push the
severity polygons and management units into PostGIS, then let the database do
the heavy spatial SQL (``ST_Intersection``, ``ST_Area``, ``ST_Union``). The DSN
is read from ``POSTGIS_DSN`` (the docker-compose postgis service sets a default).
Everything degrades to a clear error if no database is reachable, so the rest of
the pipeline never depends on a running container.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd

DEFAULT_DSN = "postgresql://burn:burn@localhost:5432/burn"


def get_engine(dsn: str | None = None):
    from sqlalchemy import create_engine

    return create_engine(dsn or os.environ.get("POSTGIS_DSN", DEFAULT_DSN))


def write_gdf(gdf: gpd.GeoDataFrame, table: str, engine, if_exists: str = "replace") -> None:
    gdf.to_postgis(table, engine, if_exists=if_exists, index=False)


def run_sql(engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, engine)
