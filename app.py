import folium, streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import planner as P
import bench

st.set_page_config(page_title="ISP Network Planner", layout="wide")
st.title("ISP Network Planner")

with st.sidebar:
    src = st.radio("Road network", ["Synthetic grid (offline)", "OpenStreetMap place"])
    place = st.text_input("Place", "Pune, India") if src.startswith("Open") else None
    osm_dist = st.slider("OSM radius (m)", 1000, 8000, 3000,
                         help="Used when the place name geocodes to a point rather than a "
                              "boundary polygon. Also bounds the download so a broad query "
                              "cannot hang the demo.") if src.startswith("Open") else 3000
    n_d = st.slider("Demand points", 20, 150, 60)
    n_c = st.slider("Candidate towers |C|", 8, 40, 20)
    seed = st.number_input("Seed", 0, 999, 0)
    limit = st.slider("Exact solver time limit (s)", 5, 60, 20)
    extra = st.slider("Redundant backbone links", 0, 5, 2)
    exch_mode = st.radio("Exchange placement", ["median", "center", "centroid"], horizontal=True,
                         help="median: least total road distance to the towers, weighted by "
                              "customer load. center: least worst-case distance. centroid: the "
                              "original rule, nearest the geometric middle, ignores the roads.")
    redundancy = st.radio("Redundancy placement", ["bridges", "cheapest"], horizontal=True,
                          help="bridges: spend each link where it removes a single point of "
                               "failure. cheapest: the globally cheapest links, which may leave "
                               "a bridge standing.")
    mbps = st.slider("Mbps per customer", 1, 20, 5)
    budget = st.slider("Per-tower equipment budget (k$)", 4, 23, 12)
    run = st.button("Plan network", type="primary")


@st.cache_resource(show_spinner="Loading graph (OpenStreetMap can take a minute)...")
def load(src, place, osm_dist):
    # `src` is unused on purpose: it is here only so switching source invalidates the cache.
    # cache_resource, not cache_data - the latter deep-copies the whole graph on every rerun.
    G = P.osm_graph(place, dist=osm_dist) if place else P.synthetic_graph()
    return P.largest_component(G)


if run:
    try:
        G, comps_dropped = load(src, place, osm_dist)
    except P.OsmFetchError as e:
        # OSM has two separate services behind it and either can fail; show which, and let the
        # user fall back to the offline grid rather than reading a traceback.
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Could not load the road network: {type(e).__name__}: {e}")
        st.stop()
    D, C, dropped = P.make_instance(G, n_d, n_c, seed)
    ex, gr = P.exact_cover(C, D, limit), P.greedy_cover(C, D)
    sites = ex["sites"]
    towers = [C[i]["node"] for i in sites]
    # Capacity and assignment are circular: stage 2 sizes equipment from the load, but a
    # capacity-aware assignment needs the capacities. Break it in two phases - the greedy
    # nearest-tower pass gives a provisional load to buy equipment against, then max-flow
    # reassigns optimally against what was actually bought.
    cust = P.assign_customers(G, C, D, sites)
    dem_greedy = [cust[k] * mbps for k in range(len(sites))]
    caps, picks, spends = P.equip_towers(C, sites, dem_greedy, budget)
    dem, unserved = P.assign_customers_flow(G, C, D, sites, caps, mbps)
    total_demand = sum(d["w"] for d in D) * mbps
    edges, cable, dm = P.backbone(G, towers, extra, strategy=redundancy)
    xy = [(G.nodes[t]["x"], G.nodes[t]["y"]) for t in towers]
    exch = P.choose_exchange(dm, exch_mode, load=dem, xy=xy)
    exch_compare = {m: P.choose_exchange(dm, m, load=dem, xy=xy)
                    for m in ("median", "center", "centroid")}
    spof = P.bridges(len(sites), [(i, j) for i, j, _ in edges])
    net = P.build_flow_network(caps, edges, dem, exch)
    flow, cut = P.edmonds_karp(net, "EX", "SINK")
    cut_rows, cut_links, cut_nodes = P.describe_cut(cut)
    st.session_state["r"] = dict(G=G, D=D, C=C, ex=ex, gr=gr, sites=sites, towers=towers, edges=edges,
                                 cable=cable, flow=flow, cut_rows=cut_rows, cut_links=cut_links,
                                 cut_nodes=cut_nodes, dem=dem, exch=exch, caps=caps, picks=picks,
                                 spends=spends, dropped=dropped, comps_dropped=comps_dropped,
                                 dem_greedy=dem_greedy, unserved=unserved,
                                 total_demand=total_demand, spof=spof, extra=extra,
                                 redundancy=redundancy, dm=dm, exch_compare=exch_compare,
                                 exch_mode=exch_mode)

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
st.table([{"tower": f"T{k}", "tier": C[s]["tier"], "sized for Mbps": r["dem_greedy"][k],
           "capacity Mbps": r["caps"][k], "spend k$": r["spends"][k],
           "bought (cost, Mbps)": str(r["picks"][k]),
           "covers its load": "yes" if r["caps"][k] >= r["dem_greedy"][k] else "NO - capped"}
          for k, s in enumerate(r["sites"])])
st.caption(f"Each tower runs its own DP: allowance = slider + {P.TIER_BONUS['high']} k$ for high-tier sites, "
           f"and the purchase is trimmed to the cheapest level meeting {P.HEADROOM}x its own load "
           f"(the multiple leaves capacity to relay neighbours' traffic).")

# ---- stage 2b ----
st.subheader("Stage 2b: customer assignment — nearest tower vs max-flow")
st.table([{"tower": f"T{k}", "capacity Mbps": r["caps"][k],
           "nearest-tower Mbps": r["dem_greedy"][k],
           "nearest util": f"{100 * r['dem_greedy'][k] / max(1, r['caps'][k]):.0f}%",
           "max-flow Mbps": r["dem"][k],
           "flow util": f"{100 * r['dem'][k] / max(1, r['caps'][k]):.0f}%",
           "over capacity?": "YES" if r["dem_greedy"][k] > r["caps"][k] else ""}
          for k in range(len(r["sites"]))])
g_over = sum(1 for k in range(len(r["sites"])) if r["dem_greedy"][k] > r["caps"][k])
total_demand = r["total_demand"]
if g_over:
    st.caption(
        f"Assigning every customer to their **nearest** tower puts {g_over} tower(s) over the "
        f"equipment bought for them — it claims {sum(r['dem_greedy'])} Mbps served when the kit "
        f"can only carry {sum(r['dem'])}. The **max-flow** assignment respects every capacity and "
        f"reports the shortfall honestly: **{r['unserved']} of {total_demand} Mbps cannot be "
        f"served**. Where a tower is the only one covering its customers, no amount of "
        f"reassignment helps — that is a signal to build another tower there, not to buy a bigger "
        f"radio.")
else:
    st.caption(f"No tower is over-subscribed at these settings, so both assignments agree. "
               f"Unserved: {r['unserved']} of {total_demand} Mbps.")
st.caption("Max-flow maximises *total* customers served and is indifferent between tied optima, so "
           "a customer may be routed to a farther tower than necessary. Preferring nearer towers "
           "among equal-value solutions would need a min-cost flow, which is out of scope.")

# ---- map ----
def ll(n): return (G.nodes[n]["lat"], G.nodes[n]["lon"])

m = folium.Map(location=ll(r["towers"][0]), zoom_start=14)
for d in D:
    folium.CircleMarker(ll(d["node"]), radius=2 + d["w"] / 8, color="gray", fill=True, weight=1).add_to(m)
for i, j, dist in r["edges"]:
    bott = frozenset((i, j)) in r["cut_links"]
    brid = frozenset((i, j)) in r["spof"]
    colour = "red" if bott else ("darkorange" if brid else "blue")
    note = (" (saturated: min-cut)" if bott else "") + (" (single point of failure)" if brid else "")
    folium.PolyLine([ll(r["towers"][i]), ll(r["towers"][j])], color=colour,
                    weight=5 if (bott or brid) else 2,
                    tooltip=f"{dist:.0f} m" + note).add_to(m)
for k, s in enumerate(r["sites"]):
    c = C[s]
    folium.Circle(ll(c["node"]), radius=P.RADIUS[c["tier"]], color="green" if c["tier"] == "mid" else "purple",
                  fill=True, fill_opacity=0.08, weight=1).add_to(m)
    folium.Marker(ll(c["node"]), tooltip=f"Tower {k} ({c['tier']}, build {c['cost']}k$, "
                  f"{r['caps'][k]} Mbps for {r['dem'][k]} Mbps demand"
                  f"{', EXCHANGE' if k == r['exch'] else ''}"
                  f"{', OWN KIT SATURATED' if k in r['cut_nodes'] else ''})").add_to(m)

st.subheader("Stage 5: signal routing — Dijkstra vs A*")
tk = st.selectbox("From tower", range(len(r["sites"])))
dj = st.selectbox("To customer", range(len(D)))
s_d, s_a = {}, {}
dist, path = P.dijkstra(G, r["towers"][tk], D[dj]["node"], stats=s_d)
dist_a, path_a = P.astar(G, r["towers"][tk], D[dj]["node"], stats=s_a)
st.write(f"Shortest road path: **{dist:.0f} m**, {len(path)} intersections")
saved = 1 - s_a["popped"] / max(1, s_d["popped"])
st.table([{"algorithm": "Dijkstra", "distance m": round(dist), "nodes expanded": s_d["popped"],
           "guarantee": "optimal"},
          {"algorithm": "A* (straight-line h)", "distance m": round(dist_a),
           "nodes expanded": s_a["popped"], "guarantee": "optimal (h is admissible)"}])
st.caption(f"A* expanded **{saved:.0%} fewer nodes** for the same answer. The heuristic is a "
           f"straight-line distance in projected metres, which never over-estimates a road "
           f"distance - so A* is exact, not approximate. It prunes least when the target is far "
           f"across the map, since then almost every node lies on a plausible route.")
if abs(dist - dist_a) > 1e-6:
    st.error(f"A* and Dijkstra disagree ({dist_a:.1f} vs {dist:.1f}) - the heuristic is not admissible.")
if path:
    folium.PolyLine([ll(n) for n in path], color="orange", weight=5).add_to(m)
st_folium(m, height=550, width=None, returned_objects=[])

# ---- stage 3 resilience ----
st.subheader("Stage 3: backbone resilience (single points of failure)")
spof = sorted(tuple(sorted(b)) for b in r["spof"])
if spof:
    st.warning(f"**{len(spof)} single point(s) of failure**: "
               + ", ".join(f"T{i}-T{j}" for i, j in spof)
               + ". Cutting any one of these splits the backbone in two.")
else:
    st.success(f"No single points of failure: every tower has at least two independent paths "
               f"to the rest of the backbone.")
st.caption(f"{len(r['edges'])} links = {len(r['sites']) - 1} spanning-tree links + {r['extra']} "
           f"redundant, placed by the **{r['redundancy']}** rule. A spanning tree alone is all "
           f"bridges by definition, so redundancy is the only thing that removes them. Found with "
           f"Tarjan's DFS low-link algorithm, O(V+E).")

st.subheader("Stage 3: where to put the exchange")
_dm = r["dm"]
st.table([{"rule": mode + (" (in use)" if mode == r["exch_mode"] else ""),
           "picks": f"T{i}",
           "total road distance": f"{sum(_dm[i]) / 1000:.2f} km",
           "worst single hop": f"{max(_dm[i]) / 1000:.2f} km"}
          for mode, i in r["exch_compare"].items()])
st.caption("The exchange must sit at a tower, so 1-median and 1-center are solvable exactly by "
           "enumerating the k candidates - unlike general k-median, which is NP-hard. The original "
           "`centroid` rule ignores the road network, which is why it can lose on both measures.")

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
