-- PostGIS schema for burn-severity vector analysis.
-- (geopandas.to_postgis creates the tables; this documents the canonical schema
--  and adds the GiST spatial indexes that make the overlay queries fast.)

CREATE EXTENSION IF NOT EXISTS postgis;

-- Severity polygons, dissolved per class, in Kangaroo Island UTM (EPSG:32753).
-- class: 0 Unburnt · 1 Low–Moderate · 2 High · 3 Very High
-- geom is stored projected so ST_Area returns square metres directly.

CREATE INDEX IF NOT EXISTS severity_geom_gix
    ON severity_polygons USING GIST (geom);

CREATE INDEX IF NOT EXISTS units_geom_gix
    ON management_units USING GIST (geom);
