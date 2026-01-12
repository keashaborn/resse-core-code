#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS expansions (
  concept_id TEXT PRIMARY KEY,
  domain TEXT NOT NULL,
  first_seen_ts INTEGER NOT NULL,
  last_seen_ts INTEGER NOT NULL,
  n_expansions INTEGER NOT NULL,
  last_field_dir TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_expansions_domain ON expansions(domain);
"""


def open_db(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")
    conn.commit()
    return conn


def get_seen_set(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT concept_id FROM expansions;")
    return {r[0] for r in cur.fetchall()}
def get_expansion_stats(conn: sqlite3.Connection) -> Dict[str, Tuple[int, int]]:
    """
    Returns {concept_id: (n_expansions, last_seen_ts)}
    """
    cur = conn.execute("SELECT concept_id, n_expansions, last_seen_ts FROM expansions;")
    return {str(cid): (int(n), int(ts)) for (cid, n, ts) in cur.fetchall()}





def mark_expanded(
    conn: sqlite3.Connection,
    concept_id: str,
    domain: str,
    field_dir: str,
    ts: Optional[int] = None,
) -> None:
    now = int(ts or time.time())
    sql = """
    INSERT INTO expansions(concept_id, domain, first_seen_ts, last_seen_ts, n_expansions, last_field_dir)
    VALUES(?, ?, ?, ?, 1, ?)
    ON CONFLICT(concept_id) DO UPDATE SET
      domain=excluded.domain,
      last_seen_ts=excluded.last_seen_ts,
      n_expansions=expansions.n_expansions + 1,
      last_field_dir=excluded.last_field_dir
    """

    # Retry briefly if another writer holds the lock.
    for attempt in range(1, 11):
        try:
            conn.execute(sql, (concept_id, domain, now, now, field_dir))
            return
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "locked" in msg and attempt < 10:
                time.sleep(min(2.0, 0.05 * attempt))
                continue
            raise


def counts_by_domain(conn: sqlite3.Connection) -> Dict[str, Tuple[int, int]]:
    """
    Returns {domain: (unique_concepts, total_expansions)}
    """
    cur = conn.execute(
        """
        SELECT domain, COUNT(1) AS uniq, SUM(n_expansions) AS total
        FROM expansions
        GROUP BY domain
        ORDER BY uniq DESC;
        """
    )
    out: Dict[str, Tuple[int, int]] = {}
    for dom, uniq, total in cur.fetchall():
        out[str(dom)] = (int(uniq), int(total or 0))
    return out
