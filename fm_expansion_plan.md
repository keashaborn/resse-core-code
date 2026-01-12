# FM Expansion Engine — Phase 1 Specification
# (Fractal Monism Applied Extrapolation Pipeline)

This document defines the first stable version of the FM expansion system,
including domains, evaluator behavior, bucket logic, and the overall pipeline
flow.  
It is intentionally simple, modular, and designed to grow over time.

---

## 1. GOALS

- Expand the FM canon with **applied, practical, domain-specific Q&A**.
- Maintain strict philosophical coherence through:
  - Fractal Monism axioms (A1–A15)
  - paraphrased axioms
  - monistic / relational tone
  - avoidance of cognitive-behavioral drift
- Use a **bucketed evaluation system (A–D)** to ensure quality and safety.
- Eventually allow **auto-growth** (autonomous canon expansion) after enough
  trust is built in the evaluator.

---

## 2. INPUTS

### 2.1 Canon Corpus (NTA format)
Pulled from:~/resse-core/resse_nta_clean/

Especially:
- `primary_core/`
- `secondary_core/`  
- (optionally) `applied/`

Each file is normalized, tagged, and (after current run) axiomed.

### 2.2 Axioms
From Qdrant collection:core_axioms_v2

Contains:
- 15 canonical axioms
- all paraphrases

### 2.3 Models
- **Generator**: GPT-5.1  
- **Evaluator** (“Axiom Gate++”): GPT-5.1  
- **Embedding model**: text-embedding-3-large  
- **Vector search**: Qdrant (local)

---

## 3. TARGET DOMAINS (V1)

Initial 8 domains (balanced between psychological, interpersonal, existential):

therapy
relationships
parenting
training
work_business
emotion
identity
existential

Future domains:
- nutrition
- conflict
- spiritual practice
- trauma
- organization systems
- creative work

These can be enabled later.

---

## 4. PIPELINE (ONE SOURCE Q&A → MANY APPLIED EXAMPLES)

### STEP 1 — Principle Extraction
Input:
- Q&A item (question + answer)
- its axioms
- optional tags

Output:
- `summary`: distilled philosophical principle
- `axiom_links`: which axioms the principle expresses
- `domain_neutral_statement`: principle expressed without domain context
- `tags`: FM-consistent functional categories

### STEP 2 — Multi-Domain Extrapolation
For each domain in `DOMAINS`:

Produce:

{
“domain”: “…”,
“format”: “qa”,
“question”: “…”,
“answer”: “…”,
“notes”: “how this maps from the principle”
}

Constraints:
- monistic
- relational frame
- no dualistic metaphors
- no CBT-style “fix your thoughts”
- no moralizing
- concrete applied examples

### STEP 3 — FM Evaluator (Axiom Gate++)
Input:
- axioms
- principle JSON
- extrapolated Q&A

Output:


{
“alignment_score”: float,
“novelty_score”: float,
“drift_flags”: […],
“axiom_flags”: {…},
“verdict”: “bucket_a|bucket_b|bucket_c|bucket_d”,
“critic_comment”: “…”
}

### STEP 4 — Bucket Sorting
Write each evaluated item as a JSONL line into:

bucket_A.jsonl
bucket_B.jsonl
bucket_C.jsonl
bucket_D.jsonl

Each record includes:
- source item metadata
- principle
- extrapolation
- evaluation metadata
- timestamps
- run_id/iteration

---

## 5. EVALUATION STRICTNESS (DOMAIN-AWARE)

We compute a **baseline** from the existing corpus:

For each domain `d`:
- mean alignment score: `μᵈ`
- standard deviation: `σᵈ`

Bucket rules:

**Bucket A**  


alignment ≥ μᵈ + 0.5σᵈ
AND no drift flags


**Bucket B**  

μᵈ - 0.5σᵈ ≤ alignment < μᵈ + 0.5σᵈ
AND no critical drift flags

**Bucket C**  

μᵈ - 1σᵈ ≤ alignment < μᵈ - 0.5σᵈ
OR mild drift flags

**Bucket D**  

alignment < μᵈ - 1σᵈ
OR any major drift flag
OR explicit dualistic / CBT-dominant logic

This keeps evaluation tied to *how your real corpus performs*, not an abstract number.

---

## 6. AUTO-PROMOTION POLICY (FUTURE)

**Phase 1:**  
- You review A and B items manually.

**Phase 2:**  
- Auto-promote A items with:
  - high alignment  
  - no drift  
  - clean axiom flags

**Phase 3:**  
- Fully automatic canon growth for trusted domains
- You spot-check for drift or novelty issues

Auto-promotion is optional and should only begin once
the evaluator is reliably matching your judgment.

---

## 7. SCHEDULING

**V1:** Manual runs.

**V2:** Scripted “batch runs” triggered manually.

**V3:** Optional cron-style auto-expansion:
- nightly: small domain batch
- weekly: deeper pass
- monthly: review metrics and drift

---

## 8. NEXT STEPS (after axiom filler completes)

1. **Build domain baseline metrics**  
   - run evaluator across entire canon in Qdrant  
   - compute μᵈ, σᵈ per domain

2. **Prototype V1**  
   - run principle → extrapolation → evaluation on 1–2 items  
   - inspect the buckets

3. **Adjust evaluator strictness**  
   - tune drift flag penalties  
   - adjust A/B boundaries if needed

4. **Begin multi-item tests**  
   - run expansion for all items in one domain (e.g., “emotion”)  
   - inspect bucket distributions

5. **Move to V2**  
   - automatic bucket sorting  
   - push everything into JSONL  
   - prepare for auto-promotion testing

---

# END OF PLAN

