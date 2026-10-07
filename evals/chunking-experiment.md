# Chunking experiment — 4 strategies, same queries

Date: 2026-10-07. Corpus: 4 fixture policy docs (~400 words each, facts buried mid-document), 8 questions, dense retriever + hashing embeddings + mock generator, chunk_size=120 words, overlap=20, top_k=3. Every strategy saw the same corpus and the same questions.

| strategy | chunks | mean words/chunk | recall@3 | evidence |
|---|---|---|---|---|
| word | 13 | 102 | 5/8 | 4/8 |
| recursive | 14 | 103 | 5/8 | 4/8 |
| fixed_char | 15 | 89 | 5/8 | 4/8 |
| sentence_window | 54 | 61 | 5/8 | 4/8 |

Per-question detail (R = right doc retrieved, E = answer keywords in the retrieved chunks):

| question | word | recursive | fixed_char | sentence_window |
|---|---|---|---|---|
| How long do I have to return something I bought? | – | – | – | – |
| My order arrived damaged. What do I get? | R+E | R+E | R+E | R+E |
| How long does standard shipping take? | – | – | – | – |
| Is there a free shipping option? | R+E | R+E | R+E | R+E |
| How long does the standard warranty last? | R | R | R | R |
| Is water damage covered by the warranty? | R+E | R+E | R+E | R+E |
| What discount do new customers receive? | R+E | R+E | R+E | R+E |
| Can I return clearance items? | – | – | – | – |

**Winner: word** — evidence 4/8, recall@3 5/8, 13 chunks (vs recursive: evidence 4/8, recall@3 5/8, 14 chunks).
The top two tied on retrieval quality; word wins on index size (13 vs 14 chunks).
Questions missed by every strategy: "How long do I have to return something I bought?"; "How long does standard shipping take?"; "Can I return clearance items?" — these need better retrieval (query rewriting, hybrid), not better chunking.

Caveats: small fixture corpus; hashing embeddings are bag-of-words, not neural — boundary placement is what this measures, and the same script reruns unchanged against st embeddings. The live default stays `chunker: word`; promoting the winner is a separate decision once confirmed on the neural path.
