-- db schema start

CREATE TABLE IF NOT EXISTS papers (
    paper_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    publishdate DATE,
    tags JSONB
);

CREATE TABLE IF NOT EXISTS summaries (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    abstract TEXT,
    claude_summary TEXT,
    keywords JSON
);

CREATE TABLE IF NOT EXISTS analytics (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    sentiment REAL,
    named_entities JSON,
    word_count JSON
);

