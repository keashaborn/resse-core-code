import yaml, sys, os
if len(sys.argv)<3:
    print("Usage: python3 write_supporting_principles.py <candidates_yaml> <support_out_yaml>")
    sys.exit(1)
cand_path, out_path = sys.argv[1], sys.argv[2]
cands = yaml.safe_load(open(cand_path)) or []
# load existing supports if present
existing = []
if os.path.exists(out_path):
    existing = yaml.safe_load(open(out_path)) or []
new_cards = []
for c in cands:
    new_cards.append({
      "title": c["title"].replace("Candidate:","Supporting:").strip(),
      "statement": c["statement"],
      "evidence_count": c["support_count"],
      "avg_coherence": c["avg_coherence"],
      "avg_novelty": c["avg_novelty"],
      "status": "supporting"
    })
yaml.dump(existing + new_cards, open(out_path,"w"), allow_unicode=True)
print(f"✅ appended {len(new_cards)} supporting principles → {out_path}")
