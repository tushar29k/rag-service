# SQuAD-format QA slice — results

Slice version `squad-qa-slice-1.0-9b465d64775a`: 20 hand-written questions over 10 passages in SQuAD v1.1 format (evals/squad_qa_slice.json). A slice, not the SQuAD dataset — the full-scale retrieval slice over real SQuAD data is evals/squad_demo.py. Scored by evals/run_squad_qa.py against the full answer path with the extractive mock generator.

| backend | retrieval hit | exact match | mean token F1 | refused |
|---|---|---|---|---|
| dense | 19/20 | 19/20 | 0.203 | 1 |
| bm25 | 20/20 | 20/20 | 0.213 | 0 |
| hybrid | 20/20 | 20/20 | 0.213 | 0 |

Per-question detail (hit / EM / F1):

| id | passage | dense hit · EM · F1 | bm25 hit · EM · F1 | hybrid hit · EM · F1 |
|---|---|---|---|---|
| sq01 | The Nile | ✓ · ✓ · 0.31 | ✓ · ✓ · 0.31 | ✓ · ✓ · 0.31 |
| sq02 | The Nile | ✓ · ✓ · 0.12 | ✓ · ✓ · 0.12 | ✓ · ✓ · 0.12 |
| sq03 | Photosynthesis | ✓ · ✓ · 0.18 | ✓ · ✓ · 0.18 | ✓ · ✓ · 0.18 |
| sq04 | Photosynthesis | ✓ · ✓ · 0.22 | ✓ · ✓ · 0.22 | ✓ · ✓ · 0.22 |
| sq05 | Marie Curie | ✓ · ✓ · 0.14 | ✓ · ✓ · 0.14 | ✓ · ✓ · 0.14 |
| sq06 | Marie Curie | ✓ · ✓ · 0.24 | ✓ · ✓ · 0.24 | ✓ · ✓ · 0.24 |
| sq07 | The printing press | ✓ · ✓ · 0.25 | ✓ · ✓ · 0.25 | ✓ · ✓ · 0.25 |
| sq08 | The printing press | ✓ · ✓ · 0.13 | ✓ · ✓ · 0.13 | ✓ · ✓ · 0.13 |
| sq09 | Honeybees | ✓ · ✓ · 0.31 | ✓ · ✓ · 0.31 | ✓ · ✓ · 0.31 |
| sq10 | Honeybees | ✗ · ✗ · 0.00 | ✓ · ✓ · 0.20 | ✓ · ✓ · 0.20 |
| sq11 | Penicillin | ✓ · ✓ · 0.10 | ✓ · ✓ · 0.10 | ✓ · ✓ · 0.10 |
| sq12 | Penicillin | ✓ · ✓ · 0.19 | ✓ · ✓ · 0.19 | ✓ · ✓ · 0.19 |
| sq13 | The Moon landing | ✓ · ✓ · 0.19 | ✓ · ✓ · 0.19 | ✓ · ✓ · 0.19 |
| sq14 | The Moon landing | ✓ · ✓ · 0.43 | ✓ · ✓ · 0.43 | ✓ · ✓ · 0.43 |
| sq15 | The Great Wall | ✓ · ✓ · 0.24 | ✓ · ✓ · 0.24 | ✓ · ✓ · 0.24 |
| sq16 | The Great Wall | ✓ · ✓ · 0.31 | ✓ · ✓ · 0.31 | ✓ · ✓ · 0.31 |
| sq17 | Coffee | ✓ · ✓ · 0.10 | ✓ · ✓ · 0.10 | ✓ · ✓ · 0.10 |
| sq18 | Coffee | ✓ · ✓ · 0.10 | ✓ · ✓ · 0.10 | ✓ · ✓ · 0.10 |
| sq19 | Chess | ✓ · ✓ · 0.27 | ✓ · ✓ · 0.27 | ✓ · ✓ · 0.27 |
| sq20 | Chess | ✓ · ✓ · 0.25 | ✓ · ✓ · 0.25 | ✓ · ✓ · 0.25 |
