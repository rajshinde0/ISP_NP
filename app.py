import folium, streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import planner as P

st.set_page_config(page_title="ISP Network Planner", layout="wide")
st.title("ISP Network Planner")

with st.sidebar:
    src = st.radio("Road network", ["Synthetic grid (offline)", "OpenStreetMap place"])
    place = st.text_input("Place", "Kothrud, Pune, India") if src.startswith("Open") else None
    n_d = st.slider("Demand points", 20, 150, 60)
    n_c = st.slider("Candidate towers |C|", 8, 22, 16)
    seed = st.number_input("Seed", 0, 999, 0)
    limit = st.slider("Exact solver time limit (s)", 5, 60, 20)
    extra = st.slider("Redundant backbone links", 0, 5, 2)
    mbps = st.slider("Mbps per customer", 1, 20, 5)
    budget = st.slider("Per-tower equipment budget (k$)", 4, 23, 12)
    run = st.button("Plan network", type="primary")


@st.cache_data(show_spinner="Loading graph...")
def load(src, place):
    G = P.osm_graph(place) if place else P.synthetic_graph()
    return P.largest_component(G)


if run:
    G, dropped = load(src, place)
    D, C = P.make_instance(G, n_d, n_c, seed)
    ex, gr = P.exact_cover(C, D, limit), P.greedy_cover(C, D)
    sites = ex["sites"]
    towers = [C[i]["node"] for i in sites]
    cap_one, picked = P.knapsack(budget=budget)
    caps = [cap_one] * len(sites)
    edges, cable = P.backbone(G, towers, extra)
    cust = P.assign_customers(G, C, D, sites)
    dem = [cust[i] * mbps for i in range(len(sites))]
    xs = [G.nodes[t]["x"] for t in towers]; ys = [G.nodes[t]["y"] for t in towers]
    exch = min(range(len(sites)), key=lambda i: (xs[i] - sum(xs) / len(xs)) ** 2 + (ys[i] - sum(ys) / len(ys)) ** 2)
    net = P.build_flow_network(caps, edges, dem, exch)
    net.setdefault("SINK", {}); net["EX"] = {exch: 10 ** 9}
    flow, cut = P.edmonds_karp(net, "EX", "SINK")
    st.session_state["r"] = dict(G=G, D=D, C=C, ex=ex, gr=gr, sites=sites, towers=towers, edges=edges, cable=cable,
                                 flow=flow, cut=cut, dem=dem, exch=exch, picked=picked, dropped=dropped, caps=caps)

r = st.session_state.get("r")
if not r:
    st.info("Set parameters and click **Plan network**.")
    st.stop()

G, D, C = r["G"], r["D"], r["C"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Towers built (exact)", len(r["sites"]), f"greedy cost {r['gr']['cost']} vs exact {r['ex']['cost']}")
c2.metric("Backbone cable", f"{r['cable'] / 1000:.1f} km")
c3.metric("Max flow / demand", f"{r['flow']} / {sum(r['dem'])} Mbps")
c4.metric("Excluded components", r["dropped"])
st.subheader("Stage 1: exact vs greedy")
st.table([{k: (f"{v:.5f}" if k == "runtime_s" else v) for k, v in x.items() if k != "sites"} for x in (r["ex"], r["gr"])])
st.caption(f"Stage 2 equipment per tower: {r['picked']}. Greedy guarantee: (ln n + 1)-approximation of optimal cost.")

# ---- map ----
def ll(n): return (G.nodes[n]["lat"], G.nodes[n]["lon"])

m = folium.Map(location=ll(r["towers"][0]), zoom_start=15)
for d in D:
    folium.CircleMarker(ll(d["node"]), radius=2 + d["w"] / 8, color="gray", fill=True, weight=1).add_to(m)
cutset = {frozenset((u, v)) for u, v, _ in r["cut"]}
for i, j, dist in r["edges"]:
    bott = frozenset((i, j)) in cutset
    folium.PolyLine([ll(r["towers"][i]), ll(r["towers"][j])], color="red" if bott else "blue",
                    weight=5 if bott else 2, tooltip=f"{dist:.0f} m" + (" (min-cut)" if bott else "")).add_to(m)
for k, s in enumerate(r["sites"]):
    c = C[s]
    folium.Circle(ll(c["node"]), radius=P.RADIUS[c["tier"]], color="green" if c["tier"] == "mid" else "purple",
                  fill=True, fill_opacity=0.08, weight=1).add_to(m)
    folium.Marker(ll(c["node"]), tooltip=f"Tower {k} ({c['tier']}, {c['cost']}k$, {r['caps'][k]} Mbps"
                  f"{', EXCHANGE' if k == r['exch'] else ''})").add_to(m)

st.subheader("Stage 5: signal routing")
tk = st.selectbox("From tower", range(len(r["sites"])))
dj = st.selectbox("To customer", range(len(D)))
dist, path = P.dijkstra(G, r["towers"][tk], D[dj]["node"])
st.write(f"Shortest road path: **{dist:.0f} m**, {len(path)} intersections")
if path:
    folium.PolyLine([ll(n) for n in path], color="orange", weight=5).add_to(m)
st_folium(m, height=550, width=None, returned_objects=[])

st.subheader("Stage 4: bottleneck (min-cut)")
st.write([(f"T{u}" if u != "EX" else "EX", f"T{v}" if v != "SINK" else "SINK", c) for u, v, c in r["cut"]] or "No bottleneck: demand fully served.")

st.subheader("Stage 1: runtime scaling")
if st.button("Run benchmark"):
    rows = P.benchmark(G)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ns = [x[0] for x in rows]
    ax[0].semilogy(ns, [x[1]["runtime_s"] + 1e-6 for x in rows], "o-", label="exact"); ax[0].semilogy(ns, [x[2]["runtime_s"] + 1e-6 for x in rows], "s-", label="greedy")
    ax[0].set_xlabel("|C|"); ax[0].set_ylabel("seconds"); ax[0].legend()
    ax[1].plot(ns, [x[1]["cost"] for x in rows], "o-", label="exact"); ax[1].plot(ns, [x[2]["cost"] for x in rows], "s--", label="greedy")
    ax[1].set_xlabel("|C|"); ax[1].set_ylabel("total cost (k$)"); ax[1].legend()
    st.pyplot(fig)
