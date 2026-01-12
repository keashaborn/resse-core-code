from __future__ import annotations

import argparse, csv, re, statistics as st
from datetime import datetime, timezone
from pathlib import Path

def p90(xs):
    xs = sorted(xs)
    return xs[int(0.9*(len(xs)-1))] if xs else None

def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_glob", default="loop_logs/tier01_*.log")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    logs = sorted(Path(".").glob(args.log_glob), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"NO_LOGS: {args.log_glob}")

    rx_start = re.compile(r"^=== ITER_START (\S+) tag=(\S+) ===$")
    rx_samp  = re.compile(r"RFF_LEDGER sampling picked=\d+ unseen_used=(\d+)")
    rx_end   = re.compile(r"^=== ITER_END (\S+) tag=(\S+) rc=\d+ nodes=(\d+) edges=(\d+) seeds=(\d+) ===$")

    start_by_tag = {}
    unseen_by_tag = {}
    rows = []  # (t_end, tag, nodes, edges, seeds, unseen_used, dur_min)

    cur_tag = None
    for lp in logs:
        for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
            m = rx_start.search(ln)
            if m:
                cur_tag = m.group(2)
                start_by_tag[cur_tag] = parse_ts(m.group(1))
                continue
            m = rx_samp.search(ln)
            if m and cur_tag:
                unseen_by_tag[cur_tag] = int(m.group(1))
                continue
            m = rx_end.search(ln)
            if m:
                t_end = parse_ts(m.group(1))
                tag = m.group(2)
                nodes, edges, seeds = map(int, m.group(3,4,5))
                t_start = start_by_tag.get(tag)
                dur_min = (t_end - t_start).total_seconds()/60.0 if t_start else None
                rows.append((t_end, tag, nodes, edges, seeds, unseen_by_tag.get(tag), dur_min))
                cur_tag = None

    rows.sort(key=lambda r: r[0])
    if len(rows) < 2:
        raise SystemExit(f"NOT_ENOUGH_ITERS: {len(rows)}")

    # deltas
    d_edges = []
    d_nodes = []
    d_seeds = []
    unseen = []
    dur = []
    for i in range(1, len(rows)):
        _, _, n0, e0, s0, _, _ = rows[i-1]
        _, _, n1, e1, s1, u1, d1 = rows[i]
        d_nodes.append(n1-n0)
        d_edges.append(e1-e0)
        d_seeds.append(s1-s0)
        if u1 is not None:
            unseen.append(u1)
        if d1 is not None:
            dur.append(d1)

    print("logs_n =", len(logs))
    print("iters_completed_total =", len(rows))
    print("latest_iter =", rows[-1][1], "t_end_utc =", rows[-1][0].isoformat())

    for win in (30, 60, 90):
        if len(d_edges) < win:
            continue
        w_edges = d_edges[-win:]
        w_nodes = d_nodes[-win:]
        w_seeds = d_seeds[-win:]
        w_unseen = unseen[-win:] if len(unseen) >= win else unseen
        w_dur = dur[-win:] if len(dur) >= win else dur

        print(
            f"last{win}: "
            f"median_Δedges={st.median(w_edges):.1f} p90_Δedges={p90(w_edges)} max_Δedges={max(w_edges)} | "
            f"median_Δnodes={st.median(w_nodes):.1f} median_Δseeds={st.median(w_seeds):.1f} | "
            f"median_unseen_used={st.median(w_unseen):.1f} p90_unseen_used={p90(w_unseen)} max_unseen_used={max(w_unseen)} | "
            f"median_min={st.median(w_dur):.1f} p90_min={p90(w_dur):.1f} max_min={max(w_dur):.1f}"
        )

    out_csv = args.out_csv.strip()
    if not out_csv:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_csv = f"reports/plots/tier01_iters_{ts}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_end_utc","tag","nodes","edges","seeds","unseen_used","dur_min"])
        for (t_end, tag, nodes, edges, seeds, u, d) in rows:
            w.writerow([t_end.isoformat(), tag, nodes, edges, seeds, u if u is not None else "", f"{d:.3f}" if d is not None else ""])
    print("WROTE", out_csv)

if __name__ == "__main__":
    main()
