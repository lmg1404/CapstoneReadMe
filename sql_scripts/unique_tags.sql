-- query pull all unique tags accumulated so far
SELECT DISTINCT jsonb_array_elements(tags) AS "Tags"
FROM papers;