import sys, yaml

if len(sys.argv)<3:
    print("Usage: python3 export_contradictions.py <eval.yaml> <out.yaml>")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(inp))

bad = [x for x in d if x.get("stance")=="potential_contradiction"]
yaml.dump(bad, open(outp,"w"), allow_unicode=True)
print(f"✅ wrote {len(bad)} items → {outp}")

