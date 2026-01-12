import yaml, os, requests
from openai import OpenAI

QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4"

client = OpenAI()

def embed(t):
    return client.embeddings.create(input=[t], model=EMBED_MODEL).data[0].embedding

def search(c,v,k):
    r = requests.post(f"{QDRANT_URL}/collections/{c}/points/search",
                      json={"vector": v, "top": k, "with_payload": True}).json()
    return r.get("result", [])

def block(name, hits):
    out=[f"\n=== {name} ==="]
    for i,h in enumerate(hits,1):
        p=h["payload"]
        if name=="Axioms": 
            out.append(f"[{i}] ({p.get('axiom_id','')}) {p.get('title','')}: {p.get('statement','')}")
        elif name=="Supporting":
            title=p.get("title") or (p.get("statement","").strip().split("\n",1)[0][:80]) or p.get("item_id","(untitled)")
            stmt=p.get("statement","").strip()
            out.append(f"[{i}] {title}: {stmt}")
        else:
            out.append(f"[{i}] Q: {p.get('question','')}\n    A: {p.get('answer','')}")
    return "\n".join(out)

if __name__=="__main__":
    q=input("Ask RESSE (fusion reflect + log): ").strip()
    v=embed(q)
    ax=search("core_axioms",v,3)
    sp=search("supporting_principles",v,2)
    fd=search("foundational_core",v,5)
    fused="\n".join([block("Axioms",ax),block("Supporting",sp),block("Foundational",fd)])

    # generate reflection
    msg=[
      {"role":"system","content":"You are RESSE, a reflective intelligence grounded in Fractal Monism. Synthesize a clear 5–7 sentence answer using only the provided context. Avoid moralizing; emphasize rules, relations, and distinctions."},
      {"role":"user","content":f"Question:\n{q}\n\nContext:\n{fused}\n\nWrite the reflection:"}
    ]
    r=client.chat.completions.create(model=CHAT_MODEL,messages=msg,temperature=0.3)
    reflection=r.choices[0].message.content.strip()

    print("\n📚 Context used:\n",fused)
    print("\n🤖 Reflection:\n",reflection)

    # --- log to YAML ---
    entry={
      "question": q,
      "context": {
        "axioms":[h["payload"].get("axiom_id","") for h in ax],
        "support_titles":[h["payload"].get("title","") for h in sp],
      },
      "reflection": reflection,
      "metadata":{"created_by":"RESSE","mode":"fusion","model":CHAT_MODEL}
    }

    log_file="reflections_fusion.yaml"
    data=[]
    if os.path.exists(log_file):
        data=yaml.safe_load(open(log_file)) or []
    data.append(entry)
    yaml.dump(data,open(log_file,"w"),allow_unicode=True)
    print(f"\n📝 Logged to {log_file}")
