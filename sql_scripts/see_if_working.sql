-- just tinkering here

SELECT * FROM papers
ORDER BY publishdate;

SELECT * FROM summaries
INNER JOIN papers ON summaries.paper_id = papers.paper_id;

SELECT * FROM analytics;

-- deletes any duplicates
DELETE FROM papers
WHERE paper_id IN (
    SELECT paper_id
    FROM (
        SELECT paper_id, 
               ROW_NUMBER() OVER (PARTITION BY title, publishdate ORDER BY paper_id) AS row_num
        FROM papers
    ) AS duplicates
    WHERE row_num > 1
);

--ALTER TABLE summaries
--ADD COLUMN sample_qa JSONB;

--TRUNCATE TABLE summaries;

SELECT * FROM analytics;
SELECT * FROM summaries;
SELECT * FROM papers;

SELECT 
	p.publishdate,
	COUNT(*) AS count
FROM summaries s
LEFT JOIN papers p ON p.paper_id = s.paper_id
GROUP BY p.publishdate
ORDER BY p.publishdate DESC;

SELECT 
	p.publishdate,
	COUNT(*) AS count
FROM papers p
GROUP BY p.publishdate
ORDER BY p.publishdate DESC;

SELECT 
	p.publishdate::date, 
	jsonb_array_elements_text(p. tags) AS tag
FROM papers p
JOIN summaries s ON p.paper_id = s.paper_id
WHERE p.publishdate BETWEEN '2025-01-01' AND '2025-03-31';

select pg_database_size('readme');


