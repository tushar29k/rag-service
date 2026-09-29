"""pgvector store backend: the same 4-method interface as store.VectorStore
(upsert/search/save/load), backed by a real Postgres + pgvector instead of
an in-memory numpy array.

Needs a live server — the repo's docker-compose.yml stands one up locally:
    docker compose up -d pgvector
    export PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag

Without a DSN (or without psycopg installed) the constructor raises a
RuntimeError that says exactly this — numpy stays the default, so an
offline box never crashes. The self-test below skips cleanly with no DSN
instead of failing.
"""
import json
import os

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # caught in __init__ with a helpful error


class PgVectorStore:
    def __init__(self, dim, dsn=None, table="chunks"):
        if psycopg is None:
            raise RuntimeError(
                "store_backend: pgvector needs psycopg — "
                "pip install 'psycopg[binary]'")
        self.dsn = dsn or os.environ.get("PGVECTOR_DSN")
        if not self.dsn:
            raise RuntimeError(
                "no pgvector server configured — run "
                "'docker compose up -d pgvector', then export "
                "PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag")
        self.dim = dim
        self.table = table
        self._conn = psycopg.connect(self.dsn)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "id bigserial PRIMARY KEY, text text NOT NULL, "
                f"embedding vector({dim}) NOT NULL, "
                "meta jsonb NOT NULL DEFAULT '{}')")

    def upsert(self, texts, embs, metas):
        rows = [(t, list(e), json.dumps(m))
                for t, e, m in zip(texts, embs, metas)]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (text, embedding, meta) "
                "VALUES (%s, %s::vector, %s::jsonb)", rows)

    def search(self, query_vec, top_k=5, filters=None):
        # <=> is cosine distance, so 1 - distance lines up with the numpy
        # store's cosine similarity — same score scale, min_score gate
        # in pipeline.py treats both the same
        clauses, params = [], []
        if filters:
            # filters are exact matches on jsonb metadata keys
            clauses = ["meta ->> %s = %s"] * len(filters)
            for k, v in filters.items():
                params += [k, str(v)]
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT text, 1 - (embedding <=> %s::vector), meta "
                f"FROM {self.table} {where} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                [list(query_vec)] + params + [list(query_vec), top_k])
            return [(t, float(s), m) for t, s, m in cur.fetchall()]

    def save(self, path):
        # the data lives in postgres — save records what to reconnect to.
        # the dsn stays in the environment and never lands on disk
        json.dump({"table": self.table, "dim": self.dim},
                  open(path + ".json", "w"))

    @classmethod
    def load(cls, path):
        meta = json.load(open(path + ".json"))
        return cls(dim=meta["dim"], table=meta["table"])

    def __len__(self):
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return cur.fetchone()[0]

    @property
    def metas(self):
        # pipeline.py dedupes new chunks on content hash via retriever.metas
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT meta FROM {self.table} ORDER BY id")
            return [m for (m,) in cur.fetchall()]

    def close(self):
        self._conn.close()


if __name__ == "__main__":
    # mirrors store.py's self-test so "store_backend: pgvector passes
    # store.py self-tests" is one env var away from being literally true
    import numpy as np
    if not os.environ.get("PGVECTOR_DSN"):
        print("pgvector self-test skipped — set PGVECTOR_DSN "
              "(docker compose up -d pgvector)")
        raise SystemExit(0)
    s = PgVectorStore(dim=4, table="selftest_chunks")
    s._conn.execute("TRUNCATE selftest_chunks")
    s.upsert(["refund policy text", "leave policy text"],
             np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
             [{"dept": "sales"}, {"dept": "hr"}])
    res = s.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5,
                   filters={"dept": "sales"})
    assert res[0][0] == "refund policy text" and len(res) == 1
    s.save("/tmp/test_pgstore")
    s2 = PgVectorStore.load("/tmp/test_pgstore")
    assert len(s2) == 2
    s.close()
    print("pgvector store OK")
