"""Chunking: split documents into overlapping word-windows.

Word-based chunking is crude but deterministic, which is exactly what you
want while you're learning the mechanics. Once you're comfortable here,
swap in something sentence-aware (e.g. LangChain's
RecursiveCharacterTextSplitter) that respects sentence boundaries.
"""

def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into overlapping windows of `chunk_size` words.

    The overlap is the whole point: without it, a sentence that straddles
    a boundary gets cut in half, and the retriever can miss the one chunk
    that had your answer.
    """
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap   # step back so the next window re-covers the tail
    return chunks


if __name__ == "__main__":
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    print(f"{len(chunks)} chunks; first chunk ends: ...{chunks[0][-40:]}")
    print(f"second chunk starts: {chunks[1][:40]}...")
    assert chunks[0].split()[-20:] == chunks[1].split()[:20], "overlap broken"
    print("overlap OK")
