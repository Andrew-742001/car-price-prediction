-- name: avg_price_by_brand
SELECT brand,
       COUNT(*) AS cars,
       ROUND(AVG(price), 0) AS avg_price,
       ROUND(AVG(horsepower), 0) AS avg_hp
FROM cars
GROUP BY brand
HAVING COUNT(*) >= 3
ORDER BY avg_price DESC;

-- name: price_by_body_and_drive
SELECT carbody,
       drivewheel,
       COUNT(*) AS cars,
       ROUND(AVG(price), 0) AS avg_price
FROM cars
GROUP BY carbody, drivewheel
ORDER BY avg_price DESC;

-- name: price_tiers
SELECT CASE
         WHEN price < 10000 THEN 'Budget (<$10k)'
         WHEN price < 20000 THEN 'Mid ($10k-20k)'
         ELSE 'Premium ($20k+)'
       END AS tier,
       COUNT(*) AS cars,
       ROUND(AVG(horsepower), 0) AS avg_hp,
       ROUND(AVG(enginesize), 0) AS avg_engine,
       ROUND(AVG(avg_mpg), 1) AS avg_mpg
FROM cars
GROUP BY tier
ORDER BY avg_hp;

-- name: above_brand_average
SELECT c.brand,
       c.carbody,
       c.horsepower,
       c.price,
       ROUND(b.brand_avg, 0) AS brand_avg,
       ROUND(c.price - b.brand_avg, 0) AS diff
FROM cars c
JOIN (SELECT brand, AVG(price) AS brand_avg FROM cars GROUP BY brand) b
  ON c.brand = b.brand
WHERE c.price > b.brand_avg * 1.3
ORDER BY diff DESC
LIMIT 10;

-- name: fuel_and_aspiration
SELECT fueltype,
       aspiration,
       COUNT(*) AS cars,
       ROUND(AVG(price), 0) AS avg_price,
       ROUND(AVG(avg_mpg), 1) AS avg_mpg
FROM cars
GROUP BY fueltype, aspiration
ORDER BY avg_price DESC;
