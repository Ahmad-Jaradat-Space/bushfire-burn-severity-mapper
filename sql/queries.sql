-- Worked spatial-SQL examples over the severity + management-unit tables.
-- Run after scripts/compute_vector.py --postgis has loaded them.

-- 1. Total hectares per severity class (ST_Area on the projected geometry).
SELECT class_name,
       ROUND(SUM(ST_Area(geom)) / 1e4)::int AS hectares
FROM   severity_polygons
GROUP  BY class_name
ORDER  BY class_name;

-- 2. High + Very-High severity area falling inside each management unit
--    (ST_Intersection clips the severity polygons to each unit, ST_Area sums it).
SELECT u.unit_id,
       ROUND((SUM(ST_Area(ST_Intersection(s.geom, u.geom))) / 1e4)::numeric, 1) AS severe_ha
FROM   severity_polygons s
JOIN   management_units  u ON ST_Intersects(s.geom, u.geom)
WHERE  s.class_name IN ('High', 'Very High')
GROUP  BY u.unit_id
ORDER  BY severe_ha DESC
LIMIT  20;

-- 3. Dissolve every burnt polygon into a single geometry and report its footprint
--    (ST_Union as an aggregate).
SELECT ROUND(ST_Area(ST_Union(geom)) / 1e4)::int AS total_burnt_ha
FROM   severity_polygons
WHERE  class_name <> 'Unburnt';

-- 4. Share of each management unit that burned at any severity.
SELECT u.unit_id,
       ROUND((100 * SUM(ST_Area(ST_Intersection(s.geom, u.geom)))
                 / ST_Area(u.geom))::numeric, 1) AS pct_burnt
FROM   management_units  u
JOIN   severity_polygons s ON ST_Intersects(s.geom, u.geom)
WHERE  s.class_name <> 'Unburnt'
GROUP  BY u.unit_id, u.geom
ORDER  BY pct_burnt DESC
LIMIT  20;
