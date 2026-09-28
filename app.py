import folium, streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import planner as P
import bench

st.set_page_config(page_title="ISP Network Planner", layout="wide")
st.title("ISP Network Planner")

with st.sidebar:
    src = st.radio("Road network", ["Synthetic grid (offline)", "OpenStreetMap place"])
    place = st.text_input("Place", "Kothrud, Pune, India") if src.startswith("Open") else None
    n_d = st.slider("Demand points", 20, 150, 60)
    n_c = st.slider("Candidate towers |C|", 8, 40, 20)
    seed = st.number_input("Seed", 0, 999, 0)
    limit = st.slider("Exact solver time limit (s)", 5, 60, 20)
    extra = st.slider("Redundant backbone links", 0, 5, 2)
    mbps = st.slider("Mbps per customer", 1, 20, 5)
    budget = st.slider("Per-tower equipment budget (k$)", 4, 23, 12)
    run = st.button("Plan network", type="primary")


@st.cache_resource(show_spinner="Loading graph...")
def load(src, place):
    # `src` is unused on purpose: it is here only so switching source invalidates the cache.
    # cache_resource, not cache_data - the latter deep-copies the whole graph on every rerun.
    G = P.osm_graph(place) if place else P.synthetic_graph()
    return P.largest_component(G)


if run:
    G, comps_dropped = load(src, place)
    D, C, dropped = P.make_instance(G, n_d, n_c, seed)
    ex, gr = P.exact_cover(C, D, limit), P.greedy_cover(C, D)
    sites = ex["sites"]
    towers = [C[i]["node"] for i in sites]
    # Stage 2 sizes each tower to its own load, so customer assignment has to run first.
    cust = P.assign_customers(G, C, D, sites)
    dem = [cust[k] * mbps for k in range(len(sites))]
    caps, picks, spends = P.equip_towers(C, sites, dem, budget)
    edges, cable = P.backbone(G, towers, extra)
    xs = [G.nodes[t]["x"] for t in towers]; ys = [G.nodes[t]["y"] for t in towers]
    exch = min(range(len(sites)), key=lambda i: (xs[i] - sum(xs) / len(xs)) ** 2 + (ys[i] - sum(ys) / len(ys)) ** 2)
    net = P.build_flow_network(caps, edges, dem, exch)
    flow, cut = P.edmonds_karp(net, "EX", "SINK")
    cut_rows, cut_links, cut_nodes = P.describe_cut(cut)
    st.session_state["r"] = dict(G=G, D=D, C=C, ex=ex, gr=gr, sites=sites, towers=towers, edges=edges,
                                 cable=cable, flow=flow, cut_rows=cut_rows, cut_links=cut_links,
                                 cut_nodes=cut_nodes, dem=dem, exch=exch, caps=caps, picks=picks,
                                 spends=spends, dropped=dropped, comps_dropped=comps_dropped)

r = st.session_state.get("r")
if not r:
    st.info("Set parameters and click **Plan network**.")
    st.stop()

G, D, C = r["G"], r["D"], r["C"]
build, equip = r["ex"]["cost"], sum(r["spends"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Towers built (exact)", len(r["sites"]))
c2.metric("Project cost", f"{build + equip:.1f} k$", f"{build:.1f} build + {equip:.1f} equipment",
          delta_color="off")
c3.metric("Backbone cable", f"{r['cable'] / 1000:.1f} km")
c4.metric("Max flow / demand", f"{r['flow']} / {sum(r['dem'])} Mbps",
          f"{100 * r['flow'] / max(1, sum(r['dem'])):.0f}% served", delta_color="off")
c5.metric("Customers covered", f"{P.coverage_of_requested(r['ex'], D, r['dropped'])}%",
          f"{r['dropped']['count']} of {r['dropped']['requested']} unreachable", delta_color="off")

if r["dropped"]["count"]:
    st.warning(f"{r['dropped']['count']} of {r['dropped']['requested']} demand points "
               f"({r['dropped']['weight']} customers) lie outside **every** candidate site's radius, so they "
               f"are excluded from the set-cover instance. The solvers' own 'coverage' figures below are "
               f"100% of the {len(D)} reachable points; the tile above is the honest citywide number.")
if r["comps_dropped"]:
    st.caption(f"Stage 0: {r['comps_dropped']} disconnected road component(s) excluded by the BFS check.")

# ---- stage 1 ----
st.subheader("Stage 1: exact vs greedy (the headline)")
gap = 100 * (r["gr"]["cost"] - r["ex"]["cost"]) / max(1e-9, r["ex"]["cost"])
st.table([{k: (f"{v:.5f}" if k in ("runtime_s", "warmstart_s") else v)
           for k, v in x.items() if k != "sites"} for x in (r["ex"], r["gr"])])
st.caption(f"Greedy costs **{gap:.1f}% more** than the optimum here ({len(r['gr']['sites'])} towers vs "
           f"{len(r['ex']['sites'])}). Guarantee: greedy is an (ln n + 1)-approximation; the B&B runtime "
           f"excludes its greedy warm-start, which is reported separately as `warmstart_s`.")

# ---- stage 2 ----
st.subheader("Stage 2: equipment per tower (0/1 knapsack DP)")
st.table([{"tower": f"T{k}", "tier": C[s]["tier"], "demand Mbps": r["dem"][k], "capacity Mbps": r["caps"][k],
           "spend k$": r["spends"][k], "bought (cost, Mbps)": str(r["picks"][k]),
           "met": "yes" if r["caps"][k] >= r["dem"][k] else "NO"}
          for k, s in enumerate(r["sites"])])
st.caption(f"Each tower runs its own DP: allowance = slider + {P.TIER_BONUS['high']} k$ for high-tier sites, "
           f"and the purchase is trimmed to the cheapest level meeting {P.HEADROOM}x its own load "
           f"(the multiple leaves capacity to relay neighbours' traffic).")

# ---- map ----
def ll(n): return (G.nodes[n]["lat"], G.nodes[n]["lon"])

m = folium.Map(location=ll(r["towers"][0]), zoom_start=14)
for d in D:
    folium.CircleMarker(ll(d["node"]), radius=2 + d["w"] / 8, color="gray", fill=True, weight=1).add_to(m)
for i, j, dist in r["edges"]:
    bott = frozenset((i, j)) in r["cut_links"]
    folium.PolyLine([ll(r["towers"][i]), ll(r["towers"][j])], color="red" if bott else "blue",
                    weight=5 if bott else 2,
                    tooltip=f"{dist:.0f} m" + (" (saturated: min-cut)" if bott else "")).add_to(m)
for k, s in enumerate(r["sites"]):
    c = C[s]
    folium.Circle(ll(c["node"]), radius=P.RADIUS[c["tier"]], color="green" if c["tier"] == "mid" else "purple",
                  fill=True, fill_opacity=0.08, weight=1).add_to(m)
    folium.Marker(ll(c["node"]), tooltip=f"Tower {k} ({c['tier']}, build {c['cost']}k$, "
                  f"{r['caps'][k]} Mbps for {r['dem'][k]} Mbps demand"
                  f"{', EXCHANGE' if k == r['exch'] else ''}"
                  f"{', OWN KIT SATURATED' if k in r['cut_nodes'] else ''})").add_to(m)

st.subheader("Stage 5: signal routing")
tk = st.selectbox("From tower", range(len(r["sites"])))
dj = st.selectbox("To customer", range(len(D)))
dist, path = P.dijkstra(G, r["towers"][tk], D[dj]["node"])
st.write(f"Shortest road path: **{dist:.0f} m**, {len(path)} intersections")
if path:
    folium.PolyLine([ll(n) for n in path], color="orange", weight=5).add_to(m)
st_folium(m, height=550, width=None, returned_objects=[])

# ---- stage 4 ----
st.subheader("Stage 4: bottleneck (min-cut)")
if r["cut_rows"]:
    st.table([{"from": u, "to": v, "capacity Mbps": c} for u, v, c in r["cut_rows"]])
    bits = []
    if r["cut_links"]:
        bits.append("saturated backbone links: " + ", ".join(f"T{min(l)}-T{max(l)}" for l in r["cut_links"]))
    if r["cut_nodes"]:
        bits.append("towers limited by their own equipment: " + ", ".join(f"T{k}" for k in sorted(r["cut_nodes"])))
    st.caption(". ".join(bits) + f". Towers are split into `Tk.in -> Tk.out` so a tower cannot relay more "
               f"than its own hardware switches; the upstream fibre enters at T{r['exch']}.out.")
else:
    st.write("No bottleneck: demand fully served.")

# ---- stage 1 scaling ----
st.subheader("Stage 1: runtime scaling (what branch-and-bound buys)")
st.caption("Runs naive backtracking (every bound removed), branch-and-bound, and greedy on the same "
           "instance at each |C|. Naive hits the time limit around |C|=24 - that is the 2^n wall the "
           "plan predicts. This can take a few minutes.")
if st.button("Run benchmark"):
    bar = st.progress(0.0, text="benchmarking stage 1...")
    rows = P.benchmark(G, n_demand=n_d, seed=seed, limit=limit,
                       progress=lambda f, n: bar.progress(f, text=f"|C| = {n} done"))
    bar.empty()
    # same plotting code the CLI uses, so the app chart and the report figure cannot drift apart
    st.pyplot(bench.plot_benchmark(rows, plt))
    st.table(bench.table_rows(rows))
    st.caption("Naive cost is omitted from the quality panel: once it times out its incumbent is "
               "not an optimum. Node counts are directly comparable - both modes run the identical "
               "recursion. `python bench.py` writes these same figures plus a CSV for the report.")
