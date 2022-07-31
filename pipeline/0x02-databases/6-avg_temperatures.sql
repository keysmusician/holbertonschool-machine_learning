-- Displays the average temperature (Fahrenheit) by city ordered by temperature
-- (descending).
SELECT
    city, AVG(`value`)
AS
    `agv_temp`
FROM
    `temperatures`
GROUP BY
    `city`
ORDER BY
    `agv_temp`
DESC;
