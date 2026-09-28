"""Stage-1 benchmark artifacts for the report (plan milestone 7).

Runs naive backtracking vs branch-and-bound vs greedy across a range of |C| and writes three
files: a CSV of the raw numbers, a PNG of the runtime and quality curves, and a Markdown table
ready to paste into the report.

    python bench.py                                  # defaults, writes ./bench/
    python bench.py --sizes 8,12,16,20,24,28,32
    python bench.py --demand 80 --seed 3 --limit 30 --out report/figures

`plot_benchmark` is shared with the dashboard so the chart in the app and the chart in the report
are produced by the same code.
"""
import argparse
import csv
import os

import planner as P

SOLVERS = (("naive", "naive backtracking", "^-"),
           ("bb", "exact (B&B)", "o-"),
           ("greedy", "greedy", "s--"))


def plot_benchmark(rows, plt):
    """Two panels: runtime on a log axis, and solution quality. Returns the figure.

    `plt` is passed in rather than imported so the caller controls the matplotlib backend -
    Streamlit and a headless CLI want different ones.
    """
    ns = [r["n_cand"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, label, style in SOLVERS:
        ax[0].semilogy(ns, [r[key]["runtime_s"] + 1e-6 for r in rows], style, label=label)
    # ring the truncated searches: a timed-out run is NOT the optimum and must not read as one
    to = [(r["n_cand"], r[k]["runtime_s"] + 1e-6) for r in rows for k in ("naive", "bb")
          if r[k].get("timed_out")]
    if to:
        ax[0].plot([p[0] for p in to], [p[1] for p in to], "o", ms=14, mfc="none", mec="red",
                   mew=2, label="hit time limit (NOT optimal)")
    ax[0].set_xlabel("|C| candidate sites")
    ax[0].set_ylabel("seconds")
    ax[0].set_title("runtime (log scale)")
    ax[0].legend(fontsize=7)
    ax[1].plot(ns, [r["bb"]["cost"] for r in rows], "o-", label="exact (B&B)")
    ax[1].plot(ns, [r["greedy"]["cost"] for r in rows], "s--", label="greedy")
    ax[1].set_xlabel("|C| candidate sites")
    ax[1].set_ylabel("total build cost (k$)")
    ax[1].set_title("solution quality")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def table_rows(rows):
    """The per-size summary shown in the dashboard and written to the report table."""
    out = []
    for r in rows:
        nv, bb, gr = r["naive"], r["bb"], r["greedy"]
        out.append({
            "|C|": r["n_cand"],
            "|D|": r["n_demand"],
            "naive nodes": f"{nv['nodes']:,}" + (" (timed out)" if nv["timed_out"] else ""),
            "B&B nodes": f"{bb['nodes']:,}",
            "nodes saved": "-" if nv["timed_out"] else f"{nv['nodes'] / max(1, bb['nodes']):.0f}x",
            "exact k$": bb["cost"],
            "greedy k$": gr["cost"],
            "greedy gap": f"{100 * (gr['cost'] - bb['cost']) / max(1e-9, bb['cost']):.1f}%",
        })
    return out


def write_csv(rows, path):
    """Every field of every solver, one row per size - the raw data behind the figures."""
    fields = ["n_cand", "n_demand"]
    for key, _, _ in SOLVERS:
        fields += [f"{key}_{f}" for f in ("cost", "runtime_s", "nodes", "coverage_pct", "timed_out")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            row = {"n_cand": r["n_cand"], "n_demand": r["n_demand"]}
            for key, _, _ in SOLVERS:
                s = r[key]
                row[f"{key}_cost"] = s["cost"]
                row[f"{key}_runtime_s"] = f"{s['runtime_s']:.6f}"
                row[f"{key}_nodes"] = s.get("nodes", "")
                row[f"{key}_coverage_pct"] = s["coverage_pct"]
                row[f"{key}_timed_out"] = s.get("timed_out", "")
            wr.writerow(row)


def write_markdown(rows, path, meta):
    tbl = table_rows(rows)
    cols = list(tbl[0].keys())
    lines = [f"# Stage 1 runtime scaling", "",
             f"`|D|`={meta['demand']}, seed {meta['seed']}, {meta['limit']} s limit, "
             f"{meta['graph']}. Naive backtracking is the *identical recursion* with every bound "
             f"removed, so node counts are directly comparable.", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in tbl:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines += ["", "A row marked *(timed out)* is not an optimum: the search was cut off.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="8,12,16,20,24,28",
                    help="comma-separated |C| values (default: 8,12,16,20,24,28)")
    ap.add_argument("--demand", type=int, default=60, help="demand points per instance")
    ap.add_argument("--seed", type=int, default=0, help="instance seed")
    ap.add_argument("--limit", type=float, default=20.0, help="per-solve time limit, seconds")
    ap.add_argument("--out", default="bench", help="output directory")
    ap.add_argument("--no-naive", action="store_true",
                    help="skip unpruned backtracking (much faster, but loses the 2^n curve)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")           # headless: no display needed
    import matplotlib.pyplot as plt

    sizes = tuple(int(s) for s in args.sizes.split(","))
    os.makedirs(args.out, exist_ok=True)
    print(f"building graph...")
    G, excluded = P.largest_component(P.synthetic_graph())
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({excluded} component(s) excluded)")
    print(f"benchmarking |C| = {', '.join(map(str, sizes))}  "
          f"(naive backtracking {'OFF' if args.no_naive else 'ON'}, {args.limit}s limit)")
    rows = P.benchmark(G, sizes=sizes, n_demand=args.demand, seed=args.seed, limit=args.limit,
                       naive=not args.no_naive,
                       progress=lambda f, n: print(f"  |C|={n:<3} done ({100 * f:.0f}%)"))
    if args.no_naive:   # keep the writers' shape uniform
        for r in rows:
            r["naive"] = {"cost": r["bb"]["cost"], "runtime_s": 0.0, "nodes": 0,
                          "coverage_pct": r["bb"]["coverage_pct"], "timed_out": False}

    csv_path = os.path.join(args.out, "stage1_scaling.csv")
    png_path = os.path.join(args.out, "stage1_scaling.png")
    md_path = os.path.join(args.out, "stage1_scaling.md")
    write_csv(rows, csv_path)
    fig = plot_benchmark(rows, plt)
    fig.savefig(png_path, dpi=150)
    write_markdown(rows, md_path, {"demand": args.demand, "seed": args.seed,
                                   "limit": args.limit, "graph": "synthetic grid"})
    print("\nwrote:")
    for p in (csv_path, png_path, md_path):
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")
    print()
    for row in table_rows(rows):
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
