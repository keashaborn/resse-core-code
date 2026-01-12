# Fractal Monism Growth Engine — V1 Design

## 0. Goal

Build a system that can **grow** the Fractal Monism corpus in a way that is:

- Consistent with the axioms (A1–A15)
- Respectful of the existing canon
- Able to generate new Q&A, concepts, and applications
- Self-evaluating and self-pruning
- Explicitly separated from non-FM reference content

---

## 1. Data we already have

### 1.1 Axioms

- `core_axioms.yaml`
- Qdrant collection: `core_axioms_v2` (15 axioms + 10 paraphrases)

### 1.2 Canon & tiers (NTA)

From `resse_nta_clean` → `fm_tiers`:

- `fm_canon_v1.jsonl`
  - Pure FM-aligned Q&A (FM_CANON)
- `fm_context_v1.jsonl`
  - FM-adjacent / applied / clarifying material (FM_CONTEXT)
- `external_context_v1.jsonl`
  - Non-FM but useful reference: ABA, AI infra, other philosophies (NON_FM_CONTEXT)

Each item has:

- `id`
- `question`, `answer`
- `tags`
- `axioms_primary`, `axioms_secondary`
- `metadata`, `dataset`

### 1.3 Axiom seeds

From `fm_axiom_seeds/AX_seeds.jsonl`:

- `A1_seeds.jsonl`, …, `A15_seeds.jsonl`
- Each contains canon Q&A that strongly express that axiom.

### 1.4 Axiom concepts

From `fm_axiom_concepts/AX_concepts.jsonl`:

- `A1_concepts.jsonl`, …, `A15_concepts.jsonl`
- Each file contains ~6 conceptual children:
  - `id` (e.g. `"A1.1"`)
  - `axiom_id`
  - `title`
  - `summary`
  - `notes`

### 1.5 Extracted principles (in progress)

- `results/fm_principles.jsonl`
- One distilled FM principle per Q&A (across all tiers)

This will be the “principle cloud” for all FM content.

---

## 2. Target architecture for the Growth Engine

### 2.1 Layers

1. **Axiom Layer**
   - A1–A15 + paraphrases
2. **Concept Layer**
   - A1.* … A15.* conceptual children
3. **Principle Layer**
   - One principle per Q&A (from `fm_principles.jsonl`)
4. **Q&A Layer**
   - Canon + context Q&A
5. **External Layer**
   - Reference and non-FM context (for grounding, not for worldview)

---

## 3. Core loop (for generating new content)

For now, we focus on **Q&A generation** as the main “growth” object.

### Step 1 — Select a seed

Choose one of:

- An **axiom** (A1–A15)
- A **concept** (e.g. A1.4 “Co-Arising of Knower and Known”)
- A **principle** (from `fm_principles.jsonl`, canon-only)

### Step 2 — Build prompt context

Retrieve:

- The axiom statement + paraphrases
- 3–5 **concepts** linked to that axiom
- 3–5 **canon Q&A** closest in vector space (from `fm_canon_v1`)
- Relevant **principles** that cluster nearby

This becomes the “frame”:

> Axiom → concepts → typical Q&A → distilled principle

### Step 3 — Generate candidate Q&A

Ask the model to generate:

- 1–3 new Q&A pairs
- In FM tone
- Explicitly monistic, recursive, non-dual
- Optional domain focus (e.g. therapy, relationships, identity, etc.)

### Step 4 — Evaluate each candidate

Use:

- Axiom alignment (via `core_axioms_v2`)
- Vector similarity to `fm_canon_v1` vs. `external_context_v1`
- Style/tone classifier (later)
- Optionally re-run through a principle extractor to see its distilled idea

Produce:

```json
{
  "alignment_score": ...,
  "canon_similarity": ...,
  "external_similarity": ...,
  "drift_flags": [...],
  "bucket": "A|B|C|D"
}

Step 5 — Bucket the candidate

Proposed rules:
	•	Bucket A (FM_CANON candidate)
	•	High alignment_score
	•	High similarity to fm_canon_v1
	•	Low similarity to external_context_v1
	•	No serious drift flags
	•	Bucket B (FM_CONTEXT candidate)
	•	Good alignment, but:
	•	more applied
	•	more domain-specific
	•	not core metaphysics
	•	Bucket C (BORDERLINE / EXPERIMENTAL)
	•	Some FM flavor, but:
	•	too close to external context
	•	tone drifting toward CBT/self-help
	•	or overly speculative
	•	Bucket D (REJECT / QUARANTINE)
	•	Serious drift flags
	•	Dualistic framing
	•	Conflicts with core axioms

Step 6 — Integrate or discard

Depending on bucket:
	•	A → propose for inclusion in fm_canon_v2
	•	B → propose for fm_context_v2
	•	C → keep as experimental, do not use as seeds
	•	D → store in a fm_quarantine log for analysis (optional)

No automatic insertion at first. Eric reviews A/B items.

⸻

4. Use of the three tier collections

During evaluation & generation:
	•	When FM worldview is needed:
	•	search fm_canon_v1 first
	•	then fm_context_v1 as backup
	•	When practical / technical info is needed:
	•	search external_context_v1 + fm_context_v1
	•	When checking for drift:
	•	compare candidate vector’s neighborhood in:
	•	fm_canon_v1
	•	external_context_v1

⸻

5. Phases of implementation

Phase 1 — Analysis only
	•	Finish evaluation of fm_principles.jsonl
	•	Inspect:
	•	common principles
	•	frequency per axiom
	•	overlap clusters

Phase 2 — Handheld growth
	•	Choose a single axiom (e.g. A1)
	•	Run:
	•	concept → Q&A generation → evaluation → buckets
	•	Manually review A/B candidates with Eric

Phase 3 — Wider growth
	•	Enable generation for:
	•	A1–A5 first
	•	Then A6–A10
	•	Then A11–A15
	•	Keep human-in-the-loop promotion into canon/context

Phase 4 — Auto-growth (cautious)
	•	Allow:
	•	high-confidence A-level items to auto-append to a “candidate canon” file
	•	Still gated by:
	•	periodic manual review
	•	consistency checks against axioms and existing canon

⸻

6. Safety and alignment
	•	Axioms and fm_canon_v1 act as a hard spine.
	•	Q&A that score high but cluster near external_context_v1 are never canonized.
	•	The system is allowed to explore, but not to rewrite FM’s core without:
	•	explicit conflict detection
	•	explicit user approval

⸻

7. What comes next

Once embeddings and principle extraction finish:
	1.	Sample 10–20 principles from fm_principles.jsonl (A1-heavy first)
	2.	Verify principle quality (do they feel like FM?)
	3.	Wire a small, single-axiom growth loop for A1:
	•	pick a concept
	•	generate Q&A
	•	evaluate and bucket
	•	review together

This gives us:
	•	confidence in the method
	•	examples of good vs. bad expansion
	•	a template we can extend

⸻

(End of fm_growth_engine_v1.md)