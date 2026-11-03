"""Generator for evals/squad_qa_slice.json — a small SQuAD-format
extractive-QA slice, hand-written for this repo.

    python3 evals/make_squad_qa_slice.py

This is NOT the SQuAD dataset. It is 20 questions over 10 short
passages written for this slice, in the exact SQuAD v1.1 shape
(context + question + answer span with character offset), so the
service's QA path can be scored with the standard exact-match / F1
machinery and the numbers can sit next to recognised benchmark
results. (The full-scale retrieval slice over real SQuAD data is
evals/squad_demo.py — it grades retrieval only; this one grades the
whole answer path.)

The passages below are the single source of truth. The JSON is a
build artefact: it stamps the sha256 of the passage set into its
version field and regenerates byte-identically, so a hand-edit to
the JSON that disagrees with this file is visible in the diff.
"""
import hashlib
import json

# (title, context, [(question, answer span), ...]) — every answer
# span must occur exactly once in its context; the generator asserts
# that instead of trusting the offsets it computes.
PASSAGES = [
    ("The Nile",
     "The Nile is generally regarded as the longest river in the "
     "world, stretching about 6,650 kilometres through northeastern "
     "Africa. It flows northward through eleven countries before "
     "emptying into the Mediterranean Sea. Its two main tributaries "
     "are the White Nile and the Blue Nile, which meet near Khartoum "
     "in Sudan.",
     [("Into which sea does the Nile empty?", "the Mediterranean Sea"),
      ("Near which city do the White Nile and the Blue Nile meet?",
       "Khartoum")]),
    ("Photosynthesis",
     "Photosynthesis is the process by which green plants convert "
     "sunlight, water and carbon dioxide into oxygen and glucose. It "
     "takes place mainly in the leaves, inside organelles called "
     "chloroplasts. The green pigment chlorophyll absorbs the "
     "sunlight that powers the reaction.",
     [("Inside which organelles does photosynthesis take place?",
       "chloroplasts"),
      ("Which pigment absorbs the sunlight that powers photosynthesis?",
       "chlorophyll")]),
    ("Marie Curie",
     "Marie Curie was a Polish-born physicist and chemist who "
     "conducted pioneering research on radioactivity. She was the "
     "first woman to win a Nobel Prize, and the only person to win "
     "Nobel Prizes in two different sciences, physics and chemistry. "
     "She died in 1934 from aplastic anaemia, likely caused by her "
     "long exposure to radiation.",
     [("What did Marie Curie conduct pioneering research on?",
       "radioactivity"),
      ("In which two sciences did Marie Curie win Nobel Prizes?",
       "physics and chemistry")]),
    ("The printing press",
     "Around 1440, Johannes Gutenberg developed a printing press with "
     "movable metal type in Mainz, Germany. The press could produce "
     "hundreds of pages a day, far more than hand copying. Within a "
     "few decades, printing shops had spread across Europe and the "
     "price of books fell sharply.",
     [("Who developed the movable-type printing press?",
       "Johannes Gutenberg"),
      ("In which city did Gutenberg develop his printing press?",
       "Mainz")]),
    ("Honeybees",
     "A honeybee colony consists of a single queen, hundreds of male "
     "drones and tens of thousands of female worker bees. Workers "
     "communicate the location of flowers by performing a waggle "
     "dance on the comb. The angle of the dance relative to the sun "
     "tells the other bees which direction to fly.",
     [("How do worker bees communicate the location of flowers?",
       "waggle dance"),
      ("How many queens does a honeybee colony have?",
       "a single queen")]),
    ("Penicillin",
     "In 1928, Alexander Fleming noticed that a mould called "
     "Penicillium notatum had killed bacteria growing in a dish in "
     "his laboratory. The substance the mould produced, penicillin, "
     "became the first widely used antibiotic. Mass production began "
     "during the Second World War, and the drug saved millions of "
     "lives.",
     [("In which year did Fleming notice the mould's effect?", "1928"),
      ("What was the mould Fleming noticed called?",
       "Penicillium notatum")]),
    ("The Moon landing",
     "On 20 July 1969, Apollo 11 landed the first humans on the "
     "Moon. Astronauts Neil Armstrong and Buzz Aldrin walked on the "
     "lunar surface while Michael Collins remained in orbit aboard "
     "the command module. The landing fulfilled a goal set by "
     "President John F. Kennedy at the start of the decade.",
     [("Who remained in orbit during the Apollo 11 landing?",
       "Michael Collins"),
      ("On what date did Apollo 11 land on the Moon?",
       "20 July 1969")]),
    ("The Great Wall",
     "The Great Wall of China was built over many centuries to "
     "protect northern borders from invaders. The best-known "
     "sections were built during the Ming dynasty, when the wall "
     "was rebuilt in stone and brick. All of its branches together "
     "stretch for more than 21,000 kilometres.",
     [("During which dynasty were the best-known sections of the "
       "Great Wall built?", "the Ming dynasty"),
      ("How long are all the branches of the Great Wall together?",
       "21,000 kilometres")]),
    ("Coffee",
     "Coffee plants first grew wild in the highlands of Ethiopia, "
     "where legend credits a goatherd named Kaldi with noticing "
     "their stimulating effect. By the fifteenth century, coffee "
     "was being cultivated in Yemen and drunk in coffee houses "
     "across the Arabian Peninsula. It reached Europe in the "
     "seventeenth century through trade with Venice.",
     [("In which country did coffee plants first grow wild?",
       "Ethiopia"),
      ("Which goatherd is credited with noticing coffee's effect?",
       "Kaldi")]),
    ("Chess",
     "Chess is a two-player strategy board game played on an "
     "eight-by-eight board of 64 squares. Each player begins with "
     "sixteen pieces, including one king, one queen and eight "
     "pawns. A player wins by checkmating the opponent's king, "
     "leaving it under attack with no legal escape.",
     [("How many squares are on a chess board?", "64 squares"),
      ("How many pieces does each chess player begin with?",
       "sixteen pieces")]),
]


def build():
    data, n = [], 0
    for title, context, qas in PASSAGES:
        para = {"context": context, "qas": []}
        for question, span in qas:
            assert context.count(span) == 1, \
                f"span {span!r} must occur exactly once in {title!r}"
            n += 1
            para["qas"].append({
                "id": f"sq{n:02d}",
                "question": question,
                "answers": [{"text": span,
                             "answer_start": context.index(span)}],
            })
        data.append({"title": title, "paragraphs": [para]})
    stamp = hashlib.sha256(json.dumps(
        PASSAGES, sort_keys=True).encode()).hexdigest()[:12]
    return {"version": f"squad-qa-slice-1.0-{stamp}", "data": data}


if __name__ == "__main__":
    out = build()
    with open("evals/squad_qa_slice.json", "w") as f:
        json.dump(out, f, indent=2)
    n = sum(len(p["qas"]) for d in out["data"] for p in d["paragraphs"])
    print(f"wrote evals/squad_qa_slice.json — {n} questions over "
          f"{len(out['data'])} passages, version {out['version']}")
