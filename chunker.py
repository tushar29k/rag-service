"""Chunking: split documents into overlapping word-windows.

Word-based chunking is crude but deterministic, which is exactly what you
want while you're learning the mechanics. recursive_chunk_text below is the
sentence-aware alternative (picked with `chunker: recursive` in config):
it packs whole sentences into chunks so boundaries never slice a sentence
in half — at the cost of slightly uneven chunk sizes.
"""
import re


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


def _split_sentences(text):
    # cheap sentence splitter: cut after . ! ? followed by whitespace.
    # not perfect (abbreviations bleed through) but zero deps and good
    # enough for chunk boundaries — retrieval doesn't need perfect NLP
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def recursive_chunk_text(text, chunk_size=120, overlap=20):
    """Sentence-aware chunker: pack whole sentences up to `chunk_size` words.

    Boundaries only fall between sentences, so a chunk never starts or ends
    mid-thought — the thing word-windows get wrong. Overlap reuses trailing
    sentences covering roughly `overlap` words. If the text has no sentence
    boundaries (or one monster sentence), it degrades to word windows.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return chunk_text(text, chunk_size, overlap)
    chunks, current, cur_words = [], [], 0
    for sent in sentences:
        n_words = len(sent.split())
        if current and cur_words + n_words > chunk_size:
            chunks.append(" ".join(current))
            # overlap: carry the trailing sentences that hold ~overlap words
            tail, tail_words = [], 0
            for s in reversed(current):
                tail.insert(0, s)
                tail_words += len(s.split())
                if tail_words >= overlap:
                    break
            current, cur_words = tail, tail_words
        current.append(sent)
        cur_words += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    print(f"{len(chunks)} chunks; first chunk ends: ...{chunks[0][-40:]}")
    print(f"second chunk starts: {chunks[1][:40]}...")
    assert chunks[0].split()[-20:] == chunks[1].split()[:20], "overlap broken"
    print("overlap OK")

    # recursive: 8 sentences, ~12 words each -> 2 chunks of 4-5 sentences
    text2 = ("The refund window is thirty days from the date of purchase. "
             "You must keep the original receipt for all returns. "
             "Refunds are issued to the original payment method. "
             "Shipping fees are non-refundable in all cases. "
             "Damaged items are replaced free of charge within a week. "
             "Contact support with your order number for help. "
             "The support team answers within two business days. "
             "Holiday returns get an extended forty-five day window.")
    rchunks = recursive_chunk_text(text2, chunk_size=50, overlap=10)
    for c in rchunks:
        assert re.search(r"[.!?]$", c.strip()), f"chunk doesn't end on a sentence: {c!r}"
    joined = " ".join(rchunks)
    for sent in _split_sentences(text2):
        assert sent in joined, f"sentence lost: {sent!r}"
    print(f"recursive: {len(rchunks)} chunks, all boundaries on sentences, no sentence lost")
    print("chunker self-test OK")
