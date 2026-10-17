"""Shared backend isolation conformance assertions (RFC 001 isolation contract).

Any backend's test module can import these to prove the isolation guarantees
declared on :class:`mempalace.backends.PalaceRef`:

* per-``PalaceRef.id`` isolation — required of every backend;
* per-``PalaceRef.namespace`` isolation — required of backends advertising the
  ``supports_namespace_isolation`` capability.

This module is intentionally not a ``test_*`` file: it ships assertions, not
test cases, so pytest does not collect it directly.
"""

_PROBE_ID = "conformance-isolation-probe"
_PROBE_DOC = "partition isolation probe document"
_PROBE_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


def assert_partition_isolation(backend, writer, other, *, embedding=None):
    """Assert ``writer`` and ``other`` are isolated partitions of ``backend``.

    A record written to ``writer`` MUST NOT be returned, modified, or deleted
    through ``other`` (query / get / count / delete), and ``writer`` MUST still
    hold it afterwards. ``writer`` and ``other`` are two collections that the
    isolation contract says must not see each other — distinct palace ids (the
    universal guarantee) or distinct namespaces (the namespace guarantee).

    Embeddings are supplied only for backends that require explicit vectors, so
    the same assertion works for text-embedding backends (Chroma) and
    explicit-vector backends (qdrant, sqlite_exact) alike.
    """
    explicit = "requires_explicit_embeddings" in backend.capabilities
    vector = list(embedding if embedding is not None else _PROBE_EMBEDDING)

    baseline_other = other.count()

    add_kwargs = {
        "ids": [_PROBE_ID],
        "documents": [_PROBE_DOC],
        "metadatas": [{"wing": "conformance"}],
    }
    if explicit:
        add_kwargs["embeddings"] = [vector]
    writer.add(**add_kwargs)

    if explicit:
        leaked = other.query(query_embeddings=[vector], n_results=10)
    else:
        leaked = other.query(query_texts=[_PROBE_DOC], n_results=10)
    hit_ids = leaked.ids[0] if leaked.ids else []
    assert _PROBE_ID not in hit_ids, "query() leaked a record across the isolation boundary"

    assert other.get(ids=[_PROBE_ID]).ids == [], (
        "get() leaked a record across the isolation boundary"
    )
    assert other.count() == baseline_other, "count() leaked a record across the isolation boundary"

    # A delete issued against the other partition MUST NOT touch writer's record.
    other.delete(ids=[_PROBE_ID])
    survivor = writer.get(ids=[_PROBE_ID])
    assert survivor.ids == [_PROBE_ID], "delete() crossed the isolation boundary"


def assert_update_skips_missing_ids(backend, col, *, embedding=None):
    """Assert ``update`` skips ids that do not exist and applies the rest.

    An update MUST NOT create a record for a missing id, MUST NOT fail the
    batch because one id is missing, and MUST still change the ids that exist,
    for metadata-only and document updates alike.
    """
    explicit = "requires_explicit_embeddings" in backend.capabilities
    vector = list(embedding if embedding is not None else _PROBE_EMBEDDING)

    add_kwargs = {
        "ids": ["present"],
        "documents": ["the record that exists"],
        "metadatas": [{"wing": "conformance"}],
    }
    if explicit:
        add_kwargs["embeddings"] = [vector]
    col.add(**add_kwargs)
    baseline = col.count()

    col.update(ids=["ghost-a", "present"], metadatas=[{"room": "x"}, {"room": "moved"}])
    doc_kwargs = {"ids": ["present", "ghost-b"], "documents": ["rewritten", "phantom"]}
    if explicit:
        doc_kwargs["embeddings"] = [vector, vector]
    col.update(**doc_kwargs)

    assert col.count() == baseline, "update() created records for missing ids"
    assert col.get(ids=["ghost-a", "ghost-b"]).ids == [], "update() created a missing id"
    got = col.get(ids=["present"], include=["documents", "metadatas"])
    assert got.ids == ["present"]
    assert got.documents == ["rewritten"], "update() skipped an id that exists"
    assert got.metadatas[0].get("room") == "moved"
    assert got.metadatas[0].get("wing") == "conformance", "update() dropped existing metadata"
