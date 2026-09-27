"""SQLite database setup and schema definitions for structured research memory."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
-- Entities Table
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    aliases TEXT, -- JSON array of strings
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Sources Table
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    title TEXT,
    publisher TEXT,
    quality_score REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL
);

-- Evidence Table
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    source_id TEXT,
    url TEXT NOT NULL,
    title TEXT,
    publisher TEXT,
    published_date TEXT,
    passage TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);

-- Claims Table (Entity-centric)
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    entity_id TEXT,
    text TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'UNAUDITED', -- VERIFIED, SUPPORTED, CONTRADICTED, UNSUPPORTED
    created_at TEXT NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

-- Claim Evidence Junction Table
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, evidence_id),
    FOREIGN KEY (claim_id) REFERENCES claims(id),
    FOREIGN KEY (evidence_id) REFERENCES evidence(id)
);

-- Audit Results Table
CREATE TABLE IF NOT EXISTS audit_results (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    claim_id TEXT,
    claim_text TEXT NOT NULL,
    status TEXT NOT NULL, -- SUPPORTED, UNSUPPORTED, CONTRADICTED, MISSING_CITATION
    citation_present INTEGER NOT NULL,
    source_url TEXT,
    evidence_passage TEXT,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);

-- Feedback / Learned Policies Table
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    policy_rule TEXT NOT NULL,
    source_audit_id TEXT,
    trigger_reason TEXT,
    created_at TEXT NOT NULL
);

-- Runs Table
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question_id TEXT,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    estimated_cost_inr REAL NOT NULL,
    llm_calls INTEGER NOT NULL,
    search_calls INTEGER NOT NULL,
    fetch_calls INTEGER NOT NULL,
    memory_hits INTEGER NOT NULL,
    memory_misses INTEGER NOT NULL,
    failures TEXT, -- JSON array of error strings
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Indices for rapid lookup
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
CREATE INDEX IF NOT EXISTS idx_evidence_url ON evidence(url);
CREATE INDEX IF NOT EXISTS idx_claims_entity ON claims(entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_results_run ON audit_results(run_id);
"""


class DatabaseManager:
    """Manages SQLite connections and table initialization."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize with database path (defaults to config settings)."""
        self.db_path = db_path or settings.database_path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite connection with foreign keys enabled."""
        if self.db_path == ":memory:":
            conn = sqlite3.connect("file:research_mem?mode=memory&cache=shared", uri=True)
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize all schema tables and indexes."""
        with self.get_connection() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            conn.commit()
        logger.info(f"Initialized SQLite research database at: {self.db_path}")
