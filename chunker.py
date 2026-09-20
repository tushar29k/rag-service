"""Chunking: split documents into overlapping word-windows.

Why word-based? Simple, deterministic, and good enough to learn the
mechanics. Production swap: sentence-aware splitters (LangChain's
RecursiveCharacterTextSplitter) that respect sentence boundaries.
"""

def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into overlapping chunks of `chunk_size` words.

    Overlap exists so a sentence straddling a boundary appears whole in at
    least one chunk — otherwise the retriever can miss it entirely.
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
        start = end - overlap   # step back so the next chunk overlaps
    return chunks


if __name__ == "__main__":
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    print(f"{len(chunks)} chunks; first chunk ends: ...{chunks[0][-40:]}")
    print(f"second chunk starts: {chunks[1][:40]}...")
    assert chunks[0].split()[-20:] == chunks[1].split()[:20], "overlap broken"
    print("overlap OK")
