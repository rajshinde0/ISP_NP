"""Correctness tests for the hand-written algorithms.

The whole premise of this project is that stages 0-5 are implemented from scratch rather than
called out of a library, so "it runs" is not evidence. Each algorithm here is checked against an
independent oracle: a NetworkX built-in where one exists, brute force where the instance is small
enough, and the algorithm's own theoretical guarantee otherwise.

Run:  pytest -q          (from the project root, with the venv active)
"""
import itertools
import math
import random

import networkx as nx
import pytest

import planner as P
from interface import HEADROOM, OPTIONS, RADIUS


# ---------------- fixtures ----------------
@pytest.fixture(scope="module")
def graph():
    """One shared synthetic city; largest_component makes it connected."""
    G, _ = P.largest_component(P.synthetic_graph())
    return G


def instance(graph, n_demand=40, n_cand=12, seed=0):
    return P.make_instance(graph, n_demand, n_cand, seed)


# ---------------- Stage 0: BFS connectivity ----------------
def test_components_matches_networkx():
    """components() is a hand-rolled BFS; compare the partition itself, not just the count."""
    for seed in range(4):
        G = P.synthetic_graph(n=12, seed=seed)
        mine = {frozenset(c) for c in P.components(G)}
        ref = {frozenset(c) for c in nx.connected_components(G)}
        assert mine == ref


def test_components_on_a_deliberately_split_graph():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (10, 11), (20, 21), (21, 22), (22, 20)])
    G.add_node(99)  # isolated
    assert {frozenset(c) for c in P.components(G)} == {
        frozenset({1, 2, 3}), frozenset({10, 11}), frozenset({20, 21, 22}), frozenset({99})}


def test_largest_component_is_connected_and_maximal():
    G = P.synthetic_graph(n=14, seed=3)
    H, excluded = P.largest_component(G)
    assert nx.is_connected(H)
    assert len(H) == max(len(c) for c in nx.connected_components(G))
    assert excluded == nx.number_connected_components(G) - 1


# ---------------- Stage 5: Dijkstra ----------------
def test_dijkstra_distance_matches_networkx(graph):
    rng = random.Random(0)
    nodes = list(graph.nodes)
    for _ in range(25):
        s, t = rng.sample(nodes, 2)
        mine, path = P.dijkstra(graph, s, t)
        ref = nx.dijkstra_path_length(graph, s, t, weight="length")
        assert mine == pytest.approx(ref, rel=1e-9)
        # the returned path must be a real walk whose lengths sum to the distance
        assert path[0] == s and path[-1] == t
        assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
        assert sum(graph[u][v]["length"] for u, v in zip(path, path[1:])) == pytest.approx(mine, rel=1e-9)


def test_dijkstra_single_source_matches_networkx(graph):
    src = next(iter(graph.nodes))
    dist, _ = P.dijkstra(graph, src)
    ref = nx.single_source_dijkstra_path_length(graph, src, weight="length")
    assert set(dist) == set(ref)
    for n in ref:
        assert dist[n] == pytest.approx(ref[n], rel=1e-9)


def test_dijkstra_same_node_and_unreachable():
    G = nx.Graph()
    G.add_edge(1, 2, length=5.0)
    G.add_edge(8, 9, length=1.0)          # separate component
    assert P.dijkstra(G, 1, 1) == (0.0, [1])
    d, path = P.dijkstra(G, 1, 9)
    assert d == math.inf and path == []


# ---------------- Stage 5: A* ----------------
def test_astar_finds_the_same_distances_as_dijkstra_and_networkx(graph):
    """A* is only worth anything if it is still exact. Admissibility is the whole argument."""
    rng = random.Random(11)
    nodes = list(graph.nodes)
    for _ in range(40):
        s, t = rng.sample(nodes, 2)
        da, path = P.astar(graph, s, t)
        dd, _ = P.dijkstra(graph, s, t)
        ref = nx.dijkstra_path_length(graph, s, t, weight="length")
        assert da == pytest.approx(dd, rel=1e-9)
        assert da == pytest.approx(ref, rel=1e-9)
        assert path[0] == s and path[-1] == t
        assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
        assert sum(graph[u][v]["length"] for u, v in zip(path, path[1:])) == pytest.approx(da, rel=1e-9)


def test_astar_explores_fewer_nodes_than_dijkstra(graph):
    """The point of the heuristic. Measured over many pairs so one lucky pair cannot carry it."""
    rng = random.Random(12)
    nodes = list(graph.nodes)
    tot_a = tot_d = 0
    for _ in range(25):
        s, t = rng.sample(nodes, 2)
        sa, sd = {}, {}
        P.astar(graph, s, t, stats=sa)
        P.dijkstra(graph, s, t, stats=sd)
        assert sa["popped"] <= sd["popped"], "A* expanded more nodes than Dijkstra on some pair"
        tot_a += sa["popped"]; tot_d += sd["popped"]
    assert tot_a < tot_d * 0.5, f"expected a big saving, got {tot_a} vs {tot_d}"


def test_astar_same_node_and_unreachable_match_dijkstra():
    G = nx.Graph()
    G.add_node(1, x=0.0, y=0.0); G.add_node(2, x=3.0, y=4.0)
    G.add_node(8, x=99.0, y=0.0); G.add_node(9, x=99.0, y=1.0)
    G.add_edge(1, 2, length=5.0)
    G.add_edge(8, 9, length=1.0)                      # separate component
    assert P.astar(G, 1, 1) == (0.0, [1])
    assert P.dijkstra(G, 1, 1) == (0.0, [1])
    d, path = P.astar(G, 1, 9)
    assert d == math.inf and path == []


def test_astar_stats_are_recorded_and_the_return_shape_is_unchanged(graph):
    """dijkstra's stats dict must be purely additive: every existing caller unpacks two values."""
    src = next(iter(graph.nodes))
    dst = list(graph.nodes)[-1]
    st = {}
    d, path = P.dijkstra(graph, src, dst, stats=st)
    assert st["popped"] > 0
    dist_map, prev = P.dijkstra(graph, src, stats=st)   # dst=None branch still returns two values
    assert st["popped"] > 0 and isinstance(dist_map, dict) and isinstance(prev, dict)
    st2 = {}
    P.astar(graph, src, dst, stats=st2)
    assert st2["popped"] > 0


def test_astar_saving_on_the_tower_to_customer_queries_the_app_actually_runs(graph):
    """Pins the figure the README quotes, in the regime the dashboard uses: a built tower to a
    demand point. On synthetic_graph each edge length IS the straight-line distance between its
    endpoints, so h is not just admissible but exact - hence the large saving."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 20)["sites"]]
    tot_a = tot_d = 0
    for t in towers:
        for d in D[:6]:
            sa, sd = {}, {}
            P.astar(graph, t, d["node"], stats=sa)
            P.dijkstra(graph, t, d["node"], stats=sd)
            tot_a += sa["popped"]; tot_d += sd["popped"]
    saving = 1 - tot_a / tot_d
    assert saving > 0.6, f"heuristic saving collapsed to {saving:.0%}"


def test_astar_never_expands_a_node_twice(graph):
    """With a *consistent* heuristic (the triangle inequality holds for straight lines) a settled
    node's g is final, so A* should expand each node exactly once. Asserting expansions == settled
    pins the `done` guard: dropping it still returns the right distance, so only this invariant
    catches it."""
    rng = random.Random(13)
    nodes = list(graph.nodes)
    for _ in range(15):
        s, t = rng.sample(nodes, 2)
        st = {}
        P.astar(graph, s, t, stats=st)
        assert st["expansions"] == st["settled"],             f"expanded {st['expansions']} times but settled only {st['settled']} nodes"
        assert st["settled"] <= graph.number_of_nodes()
        assert st["popped"] >= st["expansions"], "every expansion comes from a pop"


def test_astar_saves_least_on_the_longest_paths(graph):
    """A* wins by being *directed*. When the goal is on the far side of the map almost every node
    lies on a plausible route, so there is little left to prune and the saving nearly vanishes -
    measured at ~11% between opposite corners versus ~85% for typical queries. Documenting this
    matters: it is why the k Dijkstras in backbone() were left alone (no single target at all) and
    it stops the report over-claiming what the heuristic buys."""
    nodes = sorted(graph.nodes, key=lambda n: graph.nodes[n]["x"] + graph.nodes[n]["y"])
    corner_a, corner_b = nodes[0], nodes[-1]
    sa, sd = {}, {}
    da, _ = P.astar(graph, corner_a, corner_b, stats=sa)
    dd, _ = P.dijkstra(graph, corner_a, corner_b, stats=sd)
    assert da == pytest.approx(dd, rel=1e-9), "still exact, however little it prunes"
    assert sa["popped"] <= sd["popped"]
    assert 1 - sa["popped"] / sd["popped"] < 0.5,         "if antipodal queries ever prune well, re-measure the README's figures"


# ---------------- Stage 3: Prim MST ----------------
def test_backbone_matches_networkx_mst(graph):
    """Prim runs on the towers' road-distance metric closure, so the oracle must run on that
    same closure - build it from our own Dijkstra distances and hand it to NetworkX."""
    for seed in range(3):
        D, C, _ = instance(graph, seed=seed)
        sites = P.greedy_cover(C, D)["sites"]
        towers = [C[i]["node"] for i in sites]
        if len(towers) < 3:
            continue
        edges, total = P.backbone(graph, towers, extra_links=0)
        K = nx.Graph()
        for i, t in enumerate(towers):
            dist, _ = P.dijkstra(graph, t)
            for j, u in enumerate(towers):
                if i < j:
                    K.add_edge(i, j, weight=dist[u])
        ref = nx.minimum_spanning_tree(K, weight="weight")
        assert len(edges) == len(towers) - 1                      # it is a spanning tree
        assert total == pytest.approx(ref.size(weight="weight"), rel=1e-9)


def test_backbone_tree_is_spanning_and_acyclic(graph):
    D, C, _ = instance(graph, seed=1)
    sites = P.greedy_cover(C, D)["sites"]
    towers = [C[i]["node"] for i in sites]
    edges, _ = P.backbone(graph, towers, extra_links=0)
    T = nx.Graph([(i, j) for i, j, _ in edges])
    assert T.number_of_nodes() == len(towers)
    assert nx.is_connected(T) and nx.is_tree(T)


def test_backbone_extra_links_add_cycles(graph):
    """Redundant links are what give stage 4 a non-trivial min-cut, so they must really add
    edges beyond the tree and never duplicate one."""
    D, C, _ = instance(graph, seed=1)
    sites = P.greedy_cover(C, D)["sites"]
    towers = [C[i]["node"] for i in sites]
    tree, tree_len = P.backbone(graph, towers, extra_links=0)
    plus, plus_len = P.backbone(graph, towers, extra_links=2)
    assert len(plus) == len(tree) + 2
    assert plus_len >= tree_len
    pairs = [frozenset((i, j)) for i, j, _ in plus]
    assert len(pairs) == len(set(pairs))                          # no duplicated link


# ---------------- Stage 4: Edmonds-Karp ----------------
def _to_digraph(cap):
    G = nx.DiGraph()
    for u, vs in cap.items():
        for v, c in vs.items():
            G.add_edge(u, v, capacity=c)
    return G


def test_edmonds_karp_matches_networkx_on_random_networks():
    rng = random.Random(7)
    for trial in range(300):
        n = rng.randint(4, 9)
        cap = {}
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.45:
                    cap.setdefault(u, {})[v] = rng.randint(1, 30)
        cap.setdefault(0, {}); cap.setdefault(n - 1, {})
        ref_graph = _to_digraph(cap)
        if 0 not in ref_graph or n - 1 not in ref_graph:
            # a wholly isolated source or sink: networkx refuses it, and our answer is trivially 0
            assert P.edmonds_karp(cap, 0, n - 1) == (0, [])
            continue
        flow, cut = P.edmonds_karp(cap, 0, n - 1)
        ref = nx.maximum_flow_value(ref_graph, 0, n - 1, capacity="capacity")
        assert flow == ref, f"trial {trial}: got {flow}, networkx says {ref}"
        # max-flow min-cut theorem: the reported cut must cost exactly the flow
        assert sum(c for _, _, c in cut) == pytest.approx(flow)


def test_edmonds_karp_textbook_example():
    """CLRS figure 26.1 network; max flow is 23."""
    cap = {"s": {"v1": 16, "v2": 13}, "v1": {"v3": 12}, "v2": {"v1": 4, "v4": 14},
           "v3": {"v2": 9, "t": 20}, "v4": {"v3": 7, "t": 4}, "t": {}}
    flow, cut = P.edmonds_karp(cap, "s", "t")
    assert flow == 23
    assert sum(c for _, _, c in cut) == 23


def test_edmonds_karp_cancels_flow_through_reverse_edges():
    """Regression for the residual reverse edge (`res[v][u] += f`). Found by random search: on
    this network an augmenting path has to cancel earlier flow, so dropping the reverse-edge
    update silently returns 7 instead of the true maximum of 9. Antiparallel arcs are what make
    it bite, which is exactly what the undirected backbone produces."""
    cap = {0: {2: 1, 3: 6, 4: 2}, 1: {0: 6, 5: 3}, 2: {1: 1, 3: 4, 4: 6, 5: 6},
           3: {0: 3, 1: 5, 2: 7, 4: 5}, 4: {0: 1, 1: 7}, 5: {2: 8, 3: 3}}
    flow, cut = P.edmonds_karp(cap, 0, 5)
    assert flow == 9
    assert flow == nx.maximum_flow_value(_to_digraph(cap), 0, 5, capacity="capacity")
    assert sum(c for _, _, c in cut) == pytest.approx(flow)


def test_edmonds_karp_disconnected_sink_is_zero():
    cap = {"s": {"a": 5}, "a": {}, "t": {}}
    flow, cut = P.edmonds_karp(cap, "s", "t")
    assert flow == 0 and cut == []


# ---------------- Stage 4: the node-split flow network ----------------
def test_node_splitting_caps_a_tower_at_its_own_capacity():
    """A star backbone: the hub has 3 links but only 100 Mbps of kit. Without node splitting the
    hub could relay 300. The exchange is a leaf so the hub really is the constriction."""
    caps = [100, 100, 100, 100]
    edges = [(1, 0, 1.0), (1, 2, 1.0), (1, 3, 1.0)]        # tower 1 is the hub
    dem = [0, 0, 500, 500]                                  # demand sits behind the hub
    net = P.build_flow_network(caps, edges, dem, exchange=0)
    flow, _ = P.edmonds_karp(net, "EX", "SINK")
    assert flow <= 100, "traffic through the hub must not exceed the hub's own capacity"


def test_node_capacity_binds_when_two_links_feed_one_tower():
    """A diamond: tower 3 is fed by two 100 Mbps links (200 Mbps of inbound pipe) but owns only
    100 Mbps of kit. Only the in/out split can hold it to 100 - in a tree each tower has a single
    parent link, so this redundant topology is the case that actually exercises the split."""
    caps = [1000, 1000, 1000, 100]
    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
    dem = [0, 0, 0, 500]
    net = P.build_flow_network(caps, edges, dem, exchange=0)
    # sanity: the two links into tower 3 really do carry 200 Mbps between them
    assert net[("out", 1)][("in", 3)] + net[("out", 2)][("in", 3)] == 200
    flow, _ = P.edmonds_karp(net, "EX", "SINK")
    assert flow == 100, "tower 3 must be held to its own 100 Mbps, not the 200 Mbps feeding it"


def test_flow_never_exceeds_total_demand_or_total_capacity(graph):
    for seed in range(3):
        D, C, _ = instance(graph, seed=seed)
        ex = P.exact_cover(C, D, 10)
        sites = ex["sites"]
        cust = P.assign_customers(graph, C, D, sites)
        dem = [cust[k] * 5 for k in range(len(sites))]
        caps, _, _ = P.equip_towers(C, sites, dem, 12)
        edges, _ = P.backbone(graph, [C[i]["node"] for i in sites], 2)
        net = P.build_flow_network(caps, edges, dem, 0)
        flow, cut = P.edmonds_karp(net, "EX", "SINK")
        assert 0 <= flow <= sum(dem)
        assert sum(c for _, _, c in cut) == pytest.approx(flow)


def test_exchange_feed_bypasses_its_own_access_radio():
    """The upstream fibre lands on the exchange's backbone switch. A lone exchange with a big
    pipe to one neighbour must be able to push more than its own access capacity."""
    caps = [100, 1000]
    edges = [(0, 1, 1.0)]
    dem = [0, 900]
    net = P.build_flow_network(caps, edges, dem, exchange=0)
    flow, _ = P.edmonds_karp(net, "EX", "SINK")
    assert flow == 100          # link capacity is min(100, 1000); the radio does not also bite


# ---------------- Stage 2: knapsack ----------------
def _brute_knapsack(options, budget):
    best = 0
    for r in range(len(options) + 1):
        for combo in itertools.combinations(options, r):
            if sum(c for c, _ in combo) <= budget:
                best = max(best, sum(cap for _, cap in combo))
    return best


def test_knapsack_matches_brute_force():
    rng = random.Random(3)
    for _ in range(40):
        opts = [(rng.randint(1, 9), rng.randint(5, 200)) for _ in range(rng.randint(1, 7))]
        budget = rng.randint(1, 25)
        cap, picked, spend = P.knapsack(opts, budget)
        assert cap == _brute_knapsack(opts, budget)
        assert spend <= budget
        assert sum(c for c, _ in picked) == spend
        assert sum(x for _, x in picked) == cap          # reconstruction agrees with the DP value


def test_knapsack_reconstruction_uses_each_option_at_most_once():
    cap, picked, _ = P.knapsack(OPTIONS, 23)
    assert len(picked) == len(set(picked)) or len(picked) == len(OPTIONS)
    for item in picked:
        assert item in OPTIONS


def test_knapsack_target_buys_the_cheapest_sufficient_package():
    """A tower needing 100 Mbps must not be sold the full-budget 900 Mbps package."""
    full_cap, _, full_spend = P.knapsack(OPTIONS, 12)
    small_cap, _, small_spend = P.knapsack(OPTIONS, 12, target=100)
    assert small_cap >= 100
    assert small_spend < full_spend
    assert small_cap < full_cap


def test_knapsack_zero_demand_still_buys_one_radio():
    """A relay-only tower with no customers must not end up with zero capacity, or it would
    sever the backbone it exists to carry."""
    cap, picked, spend = P.knapsack(OPTIONS, 12, target=0)
    assert cap > 0 and spend > 0 and picked


def test_knapsack_unreachable_target_falls_back_to_full_budget():
    cap, _, _ = P.knapsack(OPTIONS, 12, target=10 ** 9)
    assert cap == P.knapsack(OPTIONS, 12)[0]


# ---------------- Stage 2: per-tower equipment ----------------
def test_equip_towers_varies_by_tower_and_respects_budget(graph):
    D, C, _ = instance(graph, n_cand=16, seed=0)
    ex = P.exact_cover(C, D, 10)
    sites = ex["sites"]
    cust = P.assign_customers(graph, C, D, sites)
    dem = [cust[k] * 5 for k in range(len(sites))]
    caps, picks, spends = P.equip_towers(C, sites, dem, 12)
    assert len(caps) == len(picks) == len(spends) == len(sites)
    for k, s in enumerate(sites):
        allowance = 12 + (6 if C[s]["tier"] == "high" else 0)
        assert spends[k] <= allowance
        assert caps[k] == sum(cap for _, cap in picks[k])
        # either the headroom target is met, or the allowance simply could not reach it
        if caps[k] < dem[k] * HEADROOM:
            assert caps[k] == P.knapsack(OPTIONS, allowance)[0]
    assert len(set(caps)) > 1, "stage 2 must produce per-tower variation, not one broadcast value"


# ---------------- Stage 1: set cover ----------------
def _brute_cover(C, D):
    """Optimal weighted set cover by exhaustive subset search. Only for tiny |C|."""
    full = (1 << len(D)) - 1
    best = math.inf
    for r in range(len(C) + 1):
        for combo in itertools.combinations(range(len(C)), r):
            cov = 0
            for i in combo:
                cov |= C[i]["mask"]
            if cov == full:
                best = min(best, round(sum(C[i]["cost"] for i in combo), 2))
    return best


def test_exact_cover_matches_brute_force(graph):
    for seed in range(5):
        D, C, _ = instance(graph, n_demand=30, n_cand=9, seed=seed)
        got = P.exact_cover(C, D, 30)
        assert not got["timed_out"]
        assert got["cost"] == pytest.approx(_brute_cover(C, D), abs=0.011)


def test_pruning_is_lossless(graph):
    """The headline claim: branch and bound removes only branches that cannot improve the
    incumbent, so it must return the same optimum as the unpruned search."""
    for seed in range(4):
        for n_cand in (10, 14):
            D, C, _ = instance(graph, n_cand=n_cand, seed=seed)
            bb = P.exact_cover(C, D, 30)
            naive = P.exact_cover(C, D, 30, prune=False)
            assert not naive["timed_out"], "raise the limit; this test needs a completed search"
            assert bb["cost"] == pytest.approx(naive["cost"], abs=1e-6)
            assert bb["nodes"] < naive["nodes"], "pruning must actually prune"


def test_solvers_cover_everything_they_claim(graph):
    for seed in range(4):
        D, C, _ = instance(graph, seed=seed)
        for r in (P.exact_cover(C, D, 10), P.greedy_cover(C, D)):
            cov = 0
            for i in r["sites"]:
                cov |= C[i]["mask"]
            assert cov == (1 << len(D)) - 1, f"{r['solver']} left demand uncovered"
            assert r["coverage_pct"] == pytest.approx(100.0)
            assert r["cost"] == pytest.approx(round(sum(C[i]["cost"] for i in r["sites"]), 2))


def test_greedy_is_never_better_than_exact_and_honours_its_bound(graph):
    for seed in range(8):
        D, C, _ = instance(graph, n_cand=14, seed=seed)
        ex, gr = P.exact_cover(C, D, 20), P.greedy_cover(C, D)
        assert gr["cost"] >= ex["cost"] - 1e-6, "greedy beat the optimum: one of them is wrong"
        # Chvatal's bound is H(d) * OPT where d is the largest SET size. Our demand points carry
        # integer weights and greedy ranks on weight-per-cost, which is the same algorithm on an
        # instance where an element of weight w is w copies - so d is the largest set *weight*,
        # not len(D). Using H(len(D)) here would be the unit-weight bound and is not guaranteed.
        w = [d["w"] for d in D]
        d_max = max(P.wmask(c["mask"], w) for c in C)
        H = sum(1.0 / k for k in range(1, d_max + 1))
        assert gr["cost"] <= H * ex["cost"] + 1e-6,             f"greedy exceeded its H(d)={H:.2f} approximation guarantee"


def test_greedy_maximises_coverage_per_cost_not_raw_coverage():
    """Hand-built instance that separates the two rules. Site B covers the most demand but is
    wildly overpriced; cost-weighted greedy must take A then C for 2.0, while a greedy that
    ranked on coverage alone would take B and pay 10.0."""
    D = [{"node": j, "w": 1} for j in range(4)]
    C = [{"node": 100, "tier": "mid", "cost": 1.0, "mask": 0b0111},    # A: 3 demands, cheap
         {"node": 101, "tier": "mid", "cost": 10.0, "mask": 0b1111},   # B: all 4, overpriced
         {"node": 102, "tier": "mid", "cost": 1.0, "mask": 0b1000}]    # C: the leftover demand
    gr = P.greedy_cover(C, D)
    assert gr["sites"] == [0, 2]
    assert gr["cost"] == pytest.approx(2.0)
    assert gr["coverage_pct"] == pytest.approx(100.0)
    assert P.exact_cover(C, D, 5)["cost"] == pytest.approx(2.0)        # and 2.0 is optimal


def test_exact_cover_reports_a_timeout_rather_than_hanging(graph):
    D, C, _ = instance(graph, n_demand=60, n_cand=30, seed=0)
    r = P.exact_cover(C, D, 0.05, prune=False)
    assert r["timed_out"]
    assert r["runtime_s"] < 5.0, "the time limit must actually bound the search"


# ---------------- instance generation ----------------
def test_make_instance_dropped_accounting_adds_up(graph):
    for seed in range(5):
        D, C, dropped = instance(graph, n_demand=50, n_cand=10, seed=seed)
        assert dropped["requested"] == 50
        assert dropped["count"] == 50 - len(D)
        assert dropped["weight"] == dropped["requested_weight"] - sum(d["w"] for d in D)
        assert dropped["count"] >= 0 and dropped["weight"] >= 0


def test_every_surviving_demand_is_reachable_by_some_candidate(graph):
    """The invariant both stage-1 solvers rely on: the universe is coverable."""
    for seed in range(5):
        D, C, _ = instance(graph, n_demand=50, n_cand=10, seed=seed)
        union = 0
        for c in C:
            union |= c["mask"]
        assert union == (1 << len(D)) - 1


def test_masks_agree_with_the_radius_after_reindexing(graph):
    """Dropping demand re-indexes every bitmask; verify the masks still match the geometry."""
    D, C, _ = instance(graph, n_demand=50, n_cand=10, seed=2)
    for c in C:
        cn = graph.nodes[c["node"]]
        for j, d in enumerate(D):
            dn = graph.nodes[d["node"]]
            near = math.hypot(cn["x"] - dn["x"], cn["y"] - dn["y"]) <= RADIUS[c["tier"]]
            assert bool(c["mask"] >> j & 1) == near


def test_make_instance_is_deterministic_for_a_seed(graph):
    a = instance(graph, seed=5)
    b = instance(graph, seed=5)
    assert [d["node"] for d in a[0]] == [d["node"] for d in b[0]]
    assert [c["mask"] for c in a[1]] == [c["mask"] for c in b[1]]


def test_coverage_of_requested_never_overstates(graph):
    for seed in range(5):
        D, C, dropped = instance(graph, n_demand=50, n_cand=10, seed=seed)
        ex = P.exact_cover(C, D, 10)
        honest = P.coverage_of_requested(ex, D, dropped)
        assert honest <= ex["coverage_pct"] + 1e-9
        if dropped["count"]:
            assert honest < 100.0, "dropped demand must pull the citywide figure below 100%"


# ---------------- customer assignment ----------------
def test_assign_customers_conserves_weight_and_respects_radius(graph):
    D, C, _ = instance(graph, seed=0)
    sites = P.exact_cover(C, D, 10)["sites"]
    cust = P.assign_customers(graph, C, D, sites)
    assert sum(cust.values()) == sum(d["w"] for d in D), "every covered customer lands somewhere"
    for k, s in enumerate(sites):
        assert k in cust
    # each demand point must go to a tower that can actually reach it
    for d in D:
        dn = graph.nodes[d["node"]]
        assert any(math.hypot(graph.nodes[C[s]["node"]]["x"] - dn["x"],
                              graph.nodes[C[s]["node"]]["y"] - dn["y"]) <= RADIUS[C[s]["tier"]]
                   for s in sites)


# ---------------- benchmark plumbing ----------------
def test_benchmark_rows_are_well_formed(graph):
    rows = P.benchmark(graph, sizes=(8, 10), n_demand=30, seed=0, limit=5)
    assert [r["n_cand"] for r in rows] == [8, 10]
    for r in rows:
        assert r["naive"]["cost"] == pytest.approx(r["bb"]["cost"], abs=1e-6)
        assert r["greedy"]["cost"] >= r["bb"]["cost"] - 1e-6
        assert r["naive"]["nodes"] > r["bb"]["nodes"]


def test_benchmark_progress_callback_fires(graph):
    seen = []
    P.benchmark(graph, sizes=(8, 9), n_demand=20, seed=0, limit=5,
                progress=lambda f, n: seen.append((round(f, 2), n)))
    assert seen == [(0.5, 8), (1.0, 9)]


# ---------------- Stage 0: the OpenStreetMap path ----------------
# The network fetch itself cannot be tested offline, but everything around it can: the lat/lon
# swap, the parallel-edge collapse, and the fallback chain are all ours and all deterministic.
def _fake_osm_pair():
    """An (unprojected, projected) MultiDiGraph pair shaped like what OSMnx hands back."""
    G0 = nx.MultiDiGraph()                       # unprojected: x is LONGITUDE, y is LATITUDE
    G0.add_node(1, x=73.85, y=18.52)
    G0.add_node(2, x=73.86, y=18.53)
    G0.add_node(3, x=73.87, y=18.51)
    Gp = nx.MultiDiGraph()                       # projected: x/y in metres
    Gp.add_node(1, x=1000.0, y=2000.0)
    Gp.add_node(2, x=1500.0, y=2400.0)
    Gp.add_node(3, x=2000.0, y=1800.0)
    Gp.add_edge(1, 2, length=700.0)
    Gp.add_edge(2, 1, length=640.0)              # opposite direction, shorter -> should win
    Gp.add_edge(1, 2, length=900.0)              # parallel edge, longer -> should lose
    Gp.add_edge(2, 3, length=800.0)
    Gp.add_edge(3, 3, length=50.0)               # self-loop -> should be dropped
    return G0, Gp


def test_flatten_projected_keeps_lat_and_lon_the_right_way_round():
    """OSMnx stores x=lon, y=lat. Swapping them is silent and puts every tower in the sea."""
    G0, Gp = _fake_osm_pair()
    H = P._flatten_projected(Gp, G0)
    assert H.nodes[1]["lat"] == pytest.approx(18.52)
    assert H.nodes[1]["lon"] == pytest.approx(73.85)
    assert 18.0 < H.nodes[2]["lat"] < 19.0 and 73.0 < H.nodes[2]["lon"] < 74.0


def test_flatten_projected_uses_metric_coordinates_for_x_y():
    """Coverage radii are in metres, so x/y must come from the projected graph, not degrees."""
    G0, Gp = _fake_osm_pair()
    H = P._flatten_projected(Gp, G0)
    assert H.nodes[1]["x"] == pytest.approx(1000.0)
    assert H.nodes[1]["y"] == pytest.approx(2000.0)


def test_flatten_projected_collapses_parallel_edges_to_the_shortest():
    G0, Gp = _fake_osm_pair()
    H = P._flatten_projected(Gp, G0)
    assert not H.is_directed()
    assert H[1][2]["length"] == pytest.approx(640.0), "must keep the cheapest of the three arcs"
    assert H[2][3]["length"] == pytest.approx(800.0)


def test_flatten_projected_drops_self_loops():
    """A self-loop adds no connectivity and hands Prim a zero-cost edge to chew on."""
    G0, Gp = _fake_osm_pair()
    H = P._flatten_projected(Gp, G0)
    assert not H.has_edge(3, 3)
    assert H.number_of_nodes() == 3 and H.number_of_edges() == 2


def test_flatten_projected_output_feeds_the_rest_of_the_pipeline():
    """Whatever stage 0 returns must carry exactly the attributes stages 1-5 read."""
    G0, Gp = _fake_osm_pair()
    H = P._flatten_projected(Gp, G0)
    for _, d in H.nodes(data=True):
        assert {"x", "y", "lat", "lon"} <= set(d)
    for _, _, d in H.edges(data=True):
        assert "length" in d
    d, path = P.dijkstra(H, 1, 3)
    assert d == pytest.approx(1440.0) and path == [1, 2, 3]


def test_osm_graph_falls_back_to_address_when_place_has_no_polygon(monkeypatch):
    """The real-world failure: Nominatim returns a bare point for most neighbourhood names and
    graph_from_place rejects it. The address+radius strategy must pick up the slack."""
    import osmnx as ox
    G0, Gp = _fake_osm_pair()
    calls = []

    def no_polygon(*a, **k):
        calls.append("place")
        raise TypeError("Nominatim did not geocode query to a (Multi)Polygon")

    def by_address(*a, **k):
        calls.append("address")
        return G0

    monkeypatch.setattr(ox, "graph_from_place", no_polygon)
    monkeypatch.setattr(ox, "graph_from_address", by_address)
    monkeypatch.setattr(ox, "project_graph", lambda g: Gp)
    H = P.osm_graph("Karve Nagar, Pune, India")
    assert calls == ["place", "address"], "both strategies must be tried, in order"
    assert H.number_of_nodes() == 3


def test_osm_graph_skips_a_strategy_that_returns_an_empty_graph(monkeypatch):
    """'Found no graph nodes within the requested polygon' comes back as an empty graph on some
    osmnx paths rather than an exception; that must still fall through to the next strategy."""
    import osmnx as ox
    G0, Gp = _fake_osm_pair()
    monkeypatch.setattr(ox, "graph_from_place", lambda *a, **k: nx.MultiDiGraph())
    monkeypatch.setattr(ox, "graph_from_address", lambda *a, **k: G0)
    monkeypatch.setattr(ox, "project_graph", lambda g: Gp)
    assert P.osm_graph("Kothrud, Pune, India").number_of_nodes() == 3


def test_osm_graph_error_names_every_strategy_it_tried(monkeypatch):
    """When everything fails the user must get something actionable, not a bare traceback."""
    import osmnx as ox
    monkeypatch.setattr(ox, "graph_from_place", lambda *a, **k: (_ for _ in ()).throw(
        TypeError("did not geocode query to a (Multi)Polygon")))
    monkeypatch.setattr(ox, "graph_from_address", lambda *a, **k: (_ for _ in ()).throw(
        ConnectionError("overpass-api.de timed out")))
    with pytest.raises(P.OsmFetchError) as excinfo:
        P.osm_graph("Nowhere, Atlantis")
    msg = str(excinfo.value)
    assert "Nowhere, Atlantis" in msg
    assert "graph_from_place" in msg and "graph_from_address" in msg
    assert "(Multi)Polygon" in msg and "overpass" in msg
    assert "synthetic grid" in msg, "the offline escape hatch must be suggested"
