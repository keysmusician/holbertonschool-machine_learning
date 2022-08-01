-- Lists all bands with glam rock as their main style, ranked by their
-- longevity
SELECT
    band_name,
    IFNULL(split, 2020) - formed AS lifespan
FROM
    metal_bands
WHERE
    style LIKE '%Glam rock%';
