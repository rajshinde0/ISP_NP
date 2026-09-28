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
        edges, total, _ = P.backbone(graph, towers, extra_links=0)
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
    edges, _, _ = P.backbone(graph, towers, extra_links=0)
    T = nx.Graph([(i, j) for i, j, _ in edges])
    assert T.number_of_nodes() == len(towers)
    assert nx.is_connected(T) and nx.is_tree(T)


def test_backbone_extra_links_add_cycles(graph):
    """Redundant links are what give stage 4 a non-trivial min-cut, so they must really add
    edges beyond the tree and never duplicate one."""
    D, C, _ = instance(graph, seed=1)
    sites = P.greedy_cover(C, D)["sites"]
    towers = [C[i]["node"] for i in sites]
    tree, tree_len, _ = P.backbone(graph, towers, extra_links=0)
    plus, plus_len, _ = P.backbone(graph, towers, extra_links=2)
    assert len(plus) == len(tree) + 2
    assert plus_len >= tree_len
    pairs = [frozenset((i, j)) for i, j, _ in plus]
    assert len(pairs) == len(set(pairs))                          # no duplicated link


# ---------------- Stage 3: union-find and Kruskal ----------------
def test_union_find_matches_a_naive_reference():
    """Path compression and union by rank must not change *which* elements are connected, only
    how fast the answer comes back."""
    rng = random.Random(0)
    for _ in range(400):
        n = rng.randint(2, 12)
        uf = P.UnionFind(n)
        groups = {i: {i} for i in range(n)}
        for _ in range(rng.randint(0, 15)):
            a, b = rng.randrange(n), rng.randrange(n)
            ga = next(g for g in groups.values() if a in g)
            gb = next(g for g in groups.values() if b in g)
            expected_merge = ga is not gb
            assert uf.union(a, b) == expected_merge, "union must report whether it merged"
            if expected_merge:
                ga |= gb
                for x in gb:
                    groups[x] = ga
        for i in range(n):
            for j in range(n):
                naive = next(g for g in groups.values() if i in g) is                         next(g for g in groups.values() if j in g)
                assert (uf.find(i) == uf.find(j)) == naive
        distinct = len({id(next(g for g in groups.values() if i in g)) for i in range(n)})
        assert uf.components == distinct


def test_union_find_compresses_paths():
    """Build a deliberate chain, then one find must flatten it. Without compression the parent
    pointers stay a chain and repeated finds stay linear."""
    uf = P.UnionFind(6)
    for i in range(5):
        uf.parent[i] = i + 1            # hand-built chain 0 -> 1 -> ... -> 5
    assert uf.find(0) == 5
    assert all(uf.parent[i] == 5 for i in range(5)), "find() must point every node at the root"


def test_union_find_is_idempotent_on_an_existing_pair():
    uf = P.UnionFind(3)
    assert uf.union(0, 1) is True
    assert uf.union(1, 0) is False, "already joined"
    assert uf.components == 2


def test_kruskal_total_equals_prim_and_networkx(graph):
    """The MST weight is unique even where the tree is not, so assert on weight, not edge sets."""
    for seed in range(5):
        D, C, _ = P.make_instance(graph, 60, 20, seed)
        towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
        dm = P.road_distance_matrix(graph, towers)
        k = len(towers)
        _, prim_total = P.prim(dm)
        kr_edges, kr_total = P.kruskal(dm)
        K = nx.Graph()
        for i in range(k):
            for j in range(i + 1, k):
                K.add_edge(i, j, weight=dm[i][j])
        ref = nx.minimum_spanning_tree(K, weight="weight").size(weight="weight")
        assert kr_total == pytest.approx(prim_total, rel=1e-9)
        assert kr_total == pytest.approx(ref, rel=1e-9)
        assert len(kr_edges) == k - 1


def test_kruskal_output_is_a_spanning_tree(graph):
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    edges, _ = P.kruskal(P.road_distance_matrix(graph, towers))
    T = nx.Graph([(i, j) for i, j, _ in edges])
    assert T.number_of_nodes() == len(towers)
    assert nx.is_connected(T) and nx.is_tree(T)


def test_kruskal_takes_edges_in_nondecreasing_order(graph):
    """The defining property of the algorithm, distinct from Prim's grow-one-tree strategy."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    edges, _ = P.kruskal(P.road_distance_matrix(graph, towers))
    lengths = [d for _, _, d in edges]
    assert lengths == sorted(lengths)


def test_kruskal_on_a_hand_built_matrix_picks_the_known_tree():
    dm = [[0, 1, 5, 9], [1, 0, 2, 8], [5, 2, 0, 3], [9, 8, 3, 0]]
    edges, total = P.kruskal(dm)
    assert total == pytest.approx(1 + 2 + 3)
    assert {frozenset(e[:2]) for e in edges} == {
        frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_backbone_gives_the_same_length_with_either_mst(graph):
    """Which algorithm runs is a demonstration, not a decision - the cable bill is identical."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    dm = P.road_distance_matrix(graph, towers)
    for extra in (0, 2):
        _, a, _ = P.backbone(graph, towers, extra, dm=dm, mst="prim")
        _, b, _ = P.backbone(graph, towers, extra, dm=dm, mst="kruskal")
        assert a == pytest.approx(b, rel=1e-9)
    with pytest.raises(ValueError, match="unknown mst algorithm"):
        P.backbone(graph, towers, 0, dm=dm, mst="nonsense")


# ---------------- Stage 3: bridges (Tarjan / DFS) ----------------
def test_bridges_matches_networkx_on_random_graphs():
    rng = random.Random(0)
    for _ in range(300):
        k = rng.randint(2, 9)
        links = {(i, j) for i in range(k) for j in range(i + 1, k) if rng.random() < 0.35}
        H = nx.Graph()
        H.add_nodes_from(range(k))
        H.add_edges_from(links)
        ref = {frozenset(b) for b in nx.bridges(H)} if H.number_of_edges() else set()
        assert P.bridges(k, links) == ref


def test_bridges_on_the_textbook_shapes():
    assert P.bridges(4, [(0, 1), (1, 2), (2, 3)]) == {
        frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}      # a path: all bridges
    assert P.bridges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) == set()    # a cycle: none
    # barbell: two triangles joined by one link - only the bar is a bridge
    barbell = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
    assert P.bridges(6, barbell) == {frozenset({2, 3})}
    assert P.bridges(3, []) == set()                                  # no edges at all


def test_bridges_ignores_self_loops_and_duplicate_links():
    """Two towers joined twice have no bridge between them. The usual skip-the-parent trick gets
    this wrong, which is why _covering dedupes into a set first."""
    assert P.bridges(2, [(0, 1), (0, 1)]) == set()
    assert P.bridges(2, [(0, 0), (0, 1)]) == {frozenset({0, 1})}


def test_bridges_handles_a_long_chain_without_blowing_the_stack():
    """Iterative DFS, not recursive: a 4000-tower chain must not hit the recursion limit."""
    n = 4000
    assert len(P.bridges(n, [(i, i + 1) for i in range(n - 1)])) == n - 1


def test_a_spanning_tree_is_entirely_bridges(graph):
    """Which is why redundancy is the only thing that can remove a single point of failure."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 20)["sites"]]
    edges, _, _ = P.backbone(graph, towers, extra_links=0)
    assert len(P.bridges(len(towers), [(i, j) for i, j, _ in edges])) == len(towers) - 1


def test_bridge_strategy_never_leaves_more_failure_points_than_cheapest(graph):
    """The real invariant. The two agree on most instances - the cheapest link often happens to
    cover a bridge anyway - so this asserts 'never worse', not 'always better'."""
    worse = 0
    for seed in (0, 1, 2, 3):
        D, C, _ = P.make_instance(graph, 60, 20, seed)
        sites = P.exact_cover(graph and C, D, 10)["sites"]
        towers = [C[i]["node"] for i in sites]
        dm = P.road_distance_matrix(graph, towers)
        for extra in (1, 2, 3):
            counts = {}
            for strat in ("cheapest", "bridges"):
                e, _, _ = P.backbone(graph, towers, extra, strategy=strat, dm=dm)
                assert len(e) == len(towers) - 1 + extra, "both must spend the same link budget"
                counts[strat] = len(P.bridges(len(towers), [(i, j) for i, j, _ in e]))
            if counts["bridges"] > counts["cheapest"]:
                worse += 1
    assert worse == 0, "the bridge-targeted rule left more single points of failure"


def test_bridge_strategy_wins_on_an_instance_where_cheapest_does_not(graph):
    """Pins a case where the rule actually pays, found by sweeping seeds: it buys a little more
    cable to clear the last single point of failure."""
    D, C, _ = P.make_instance(graph, 60, 20, 3)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    dm = P.road_distance_matrix(graph, towers)
    res = {}
    for strat in ("cheapest", "bridges"):
        e, total, _ = P.backbone(graph, towers, 2, strategy=strat, dm=dm)
        res[strat] = (len(P.bridges(len(towers), [(i, j) for i, j, _ in e])), total)
    assert res["cheapest"][0] > res["bridges"][0] == 0, f"expected an improvement, got {res}"
    assert res["bridges"][1] >= res["cheapest"][1], "resilience here costs a little extra cable"


def test_the_redundancy_strategy_does_not_change_the_spanning_tree(graph):
    """Only the extra links differ; the MST underneath must be identical."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    dm = P.road_distance_matrix(graph, towers)
    tree, _, _ = P.backbone(graph, towers, 0, dm=dm)
    tree_set = {frozenset(e[:2]) for e in tree}
    for strat in ("cheapest", "bridges"):
        e, _, _ = P.backbone(graph, towers, 3, strategy=strat, dm=dm)
        assert tree_set <= {frozenset(x[:2]) for x in e}


def test_unknown_redundancy_strategy_is_rejected(graph):
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    with pytest.raises(ValueError, match="unknown redundancy strategy"):
        P.backbone(graph, towers, 2, strategy="nonsense")


def test_backbone_returns_a_reusable_distance_matrix(graph):
    """Returned so the exchange choice and a second MST need not pay for k more Dijkstras."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    edges, total, dm = P.backbone(graph, towers, 2)
    k = len(towers)
    assert len(dm) == k and all(len(row) == k for row in dm)
    for i in range(k):
        assert dm[i][i] == pytest.approx(0.0)
        for j in range(k):
            assert dm[i][j] == pytest.approx(dm[j][i]), "road distances are symmetric here"
    # passing it back in must reproduce the identical backbone
    again, total2, _ = P.backbone(graph, towers, 2, dm=dm)
    assert again == edges and total2 == pytest.approx(total)


# ---------------- Stage 3: exchange placement ----------------
# T0 hugs a tight cluster but is stranded far from T5; T1 is middling from everyone. So the
# minimum-TOTAL tower and the minimum-WORST-HOP tower are different, which is what makes this
# matrix able to tell `sum` and `max` apart. Row sums: 93, 158, 97, 97, 97, 278 -> median picks T0.
# Row maxima:     60,  35,  61,  61,  61,  61 -> center picks T1.
_DM_SPLIT = [
    [0, 30, 1, 1, 1, 60],
    [30, 0, 31, 31, 31, 35],
    [1, 31, 0, 2, 2, 61],
    [1, 31, 2, 0, 2, 61],
    [1, 31, 2, 2, 0, 61],
    [60, 35, 61, 61, 61, 0],
]


def test_choose_exchange_median_minimises_total_not_worst_distance():
    """Brute-forced against the definition, on a matrix where sum and max disagree - otherwise
    a median that secretly computed max would pass."""
    assert P.choose_exchange(_DM_SPLIT, "median") == 0
    assert P.choose_exchange(_DM_SPLIT, "median") == min(
        range(6), key=lambda i: sum(_DM_SPLIT[i]))
    assert P.choose_exchange(_DM_SPLIT, "median") != min(
        range(6), key=lambda i: max(_DM_SPLIT[i])), "this matrix must separate the two rules"


def test_choose_exchange_center_minimises_the_worst_hop_not_the_total():
    assert P.choose_exchange(_DM_SPLIT, "center") == 1
    assert P.choose_exchange(_DM_SPLIT, "center") == min(
        range(6), key=lambda i: max(_DM_SPLIT[i]))
    assert P.choose_exchange(_DM_SPLIT, "center") != min(
        range(6), key=lambda i: sum(_DM_SPLIT[i]))


def test_choose_exchange_median_and_center_really_disagree():
    """Otherwise offering both modes would be pointless. The previous version of this test used a
    symmetric matrix where both rules picked the same tower, making every assertion vacuous."""
    med = P.choose_exchange(_DM_SPLIT, "median")
    cen = P.choose_exchange(_DM_SPLIT, "center")
    assert med != cen
    assert sum(_DM_SPLIT[med]) < sum(_DM_SPLIT[cen]), "median must win on total"
    assert max(_DM_SPLIT[cen]) < max(_DM_SPLIT[med]), "center must win on worst hop"


def test_choose_exchange_load_weighting_moves_the_pick():
    """Unweighted the median sits with the cluster; weight the demand onto the far tower and it
    must follow the load there."""
    unweighted = P.choose_exchange(_DM_SPLIT, "median")
    load = [1, 1, 1, 1, 1, 50]                      # nearly all customers at the far tower
    weighted = P.choose_exchange(_DM_SPLIT, "median", load=load)
    brute = min(range(6), key=lambda i: sum(_DM_SPLIT[i][j] * load[j] for j in range(6)))
    assert weighted == brute
    assert weighted != unweighted, "the load weighting had no effect"


def test_choose_exchange_centroid_reproduces_the_rule_it_replaced(graph):
    """Regression against the inline expression that used to live in app.py, so swapping in
    choose_exchange cannot silently change the comparison baseline."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    towers = [C[i]["node"] for i in P.exact_cover(C, D, 10)["sites"]]
    xy = [(graph.nodes[t]["x"], graph.nodes[t]["y"]) for t in towers]
    xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
    old = min(range(len(towers)),
              key=lambda i: (xs[i] - sum(xs) / len(xs)) ** 2 + (ys[i] - sum(ys) / len(ys)) ** 2)
    dm = P.road_distance_matrix(graph, towers)
    assert P.choose_exchange(dm, "centroid", xy=xy) == old


def test_choose_exchange_beats_the_centroid_rule_on_deliverable_bandwidth(graph):
    """The result that justifies the feature. The centroid rule here picks a *shorter* total
    distance yet delivers far less bandwidth, because what matters is sitting near the demand,
    not near the geometric middle of the towers."""
    D, C, _ = P.make_instance(graph, 60, 20, 0)
    sites = P.exact_cover(C, D, 20)["sites"]
    towers = [C[i]["node"] for i in sites]
    cust = P.assign_customers(graph, C, D, sites)
    caps, _, _ = P.equip_towers(C, sites, [cust[k] * 5 for k in range(len(sites))], 12)
    dem, _ = P.assign_customers_flow(graph, C, D, sites, caps, 5)
    edges, _, dm = P.backbone(graph, towers, 2)
    xy = [(graph.nodes[t]["x"], graph.nodes[t]["y"]) for t in towers]
    flows = {}
    for mode in ("median", "centroid"):
        e = P.choose_exchange(dm, mode, load=dem, xy=xy)
        net = P.build_flow_network(caps, edges, dem, e)
        flows[mode], _ = P.edmonds_karp(net, "EX", "SINK")
    assert flows["median"] > flows["centroid"], f"expected the median to win, got {flows}"


def test_choose_exchange_rejects_bad_input():
    dm = [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="unknown exchange mode"):
        P.choose_exchange(dm, "nonsense")
    with pytest.raises(ValueError, match="needs xy"):
        P.choose_exchange(dm, "centroid")
    with pytest.raises(ValueError, match="no towers"):
        P.choose_exchange([], "median")


def test_choose_exchange_handles_a_single_tower():
    assert P.choose_exchange([[0.0]], "median") == 0
    assert P.choose_exchange([[0.0]], "center") == 0
    assert P.choose_exchange([[0.0]], "centroid", xy=[(5.0, 5.0)]) == 0


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
        edges, _, _ = P.backbone(graph, [C[i]["node"] for i in sites], 2)
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


# ---------------- Stage 2b: capacity-aware assignment ----------------
def _bipartite_ref(graph, C, D, sites, caps, mbps):
    """The same assignment network, handed to networkx as an independent oracle."""
    cov = P._covering(graph, C, D, sites)
    H = nx.DiGraph()
    for j, d in enumerate(D):
        if not cov[j]:
            continue
        need = d["w"] * mbps
        H.add_edge("SRC", ("d", j), capacity=need)
        for _, k in cov[j]:
            H.add_edge(("d", j), ("t", k), capacity=need)
    for k, c in enumerate(caps):
        H.add_edge(("t", k), "SINK", capacity=c)
    return nx.maximum_flow_value(H, "SRC", "SINK", capacity="capacity")


def _planned(graph, seed=0, n_cand=20, mbps=5, budget=12):
    D, C, _ = P.make_instance(graph, 60, n_cand, seed)
    sites = P.exact_cover(C, D, 20)["sites"]
    cust = P.assign_customers(graph, C, D, sites)
    greedy_load = [cust[k] * mbps for k in range(len(sites))]
    caps, _, _ = P.equip_towers(C, sites, greedy_load, budget)
    return D, C, sites, caps, greedy_load, mbps


def test_covering_respects_the_radius_and_is_sorted_by_distance(graph):
    D, C, sites, _, _, _ = _planned(graph)
    cov = P._covering(graph, C, D, sites)
    assert len(cov) == len(D)
    for j, hits in enumerate(cov):
        dn = graph.nodes[D[j]["node"]]
        assert hits == sorted(hits), "nearest tower must come first"
        for dd, k in hits:
            cn = graph.nodes[C[sites[k]]["node"]]
            real = math.hypot(cn["x"] - dn["x"], cn["y"] - dn["y"])
            assert dd == pytest.approx(real)
            assert real <= P.RADIUS[C[sites[k]]["tier"]] + 1e-9
        # and nothing in range was left out
        in_range = {k for k, sidx in enumerate(sites)
                    if math.hypot(graph.nodes[C[sidx]["node"]]["x"] - dn["x"],
                                  graph.nodes[C[sidx]["node"]]["y"] - dn["y"])
                    <= P.RADIUS[C[sidx]["tier"]]}
        assert {k for _, k in hits} == in_range


def test_flow_assignment_never_exceeds_a_tower_capacity(graph):
    """The defect this feature exists to fix: the greedy version loads one tower to 146%."""
    for seed in range(4):
        D, C, sites, caps, greedy_load, mbps = _planned(graph, seed=seed)
        load, unserved = P.assign_customers_flow(graph, C, D, sites, caps, mbps)
        for k in range(len(sites)):
            assert load[k] <= caps[k], f"seed {seed}: tower {k} over capacity"
        assert min(unserved, 0) == 0 and unserved >= 0


def test_greedy_assignment_really_does_overload_on_the_default_instance(graph):
    """Pins the motivating measurement, so nobody 'simplifies' the flow version away later."""
    D, C, sites, caps, greedy_load, mbps = _planned(graph, seed=0)
    over = [k for k in range(len(sites)) if greedy_load[k] > caps[k]]
    assert over, "greedy no longer overloads - re-check whether feature 2 is still needed"
    worst = max(greedy_load[k] / caps[k] for k in range(len(sites)))
    assert worst > 1.2, f"worst greedy utilisation only {worst:.0%}"


def test_flow_assignment_conserves_demand(graph):
    for seed in range(4):
        D, C, sites, caps, _, mbps = _planned(graph, seed=seed)
        load, unserved = P.assign_customers_flow(graph, C, D, sites, caps, mbps)
        assert sum(load) + unserved == sum(d["w"] for d in D) * mbps


def test_flow_assignment_matches_networkx_max_flow(graph):
    """It must actually be the maximum, not merely feasible - a capacity-respecting assignment
    that serves nobody would pass every test above."""
    for seed in range(4):
        D, C, sites, caps, _, mbps = _planned(graph, seed=seed)
        load, _ = P.assign_customers_flow(graph, C, D, sites, caps, mbps)
        assert sum(load) == _bipartite_ref(graph, C, D, sites, caps, mbps)


def test_flow_assignment_serves_everything_when_capacity_is_unlimited(graph):
    """Isolates capacity from coverage: with infinite kit the only unserved demand would be
    demand no built tower reaches, and stage 1 guarantees there is none."""
    D, C, sites, caps, _, mbps = _planned(graph, seed=0)
    load, unserved = P.assign_customers_flow(graph, C, D, sites, [10 ** 9] * len(sites), mbps)
    assert unserved == 0
    assert sum(load) == sum(d["w"] for d in D) * mbps


def test_flow_assignment_beats_greedy_on_a_hand_built_overload(graph):
    """Two towers both cover the same crowd; only one is near it. Greedy sends everyone to the
    near one and busts its capacity; flow splits the crowd and serves all of them."""
    Gm = nx.Graph()
    Gm.add_node(0, x=0.0, y=0.0)          # tower A site
    Gm.add_node(1, x=100.0, y=0.0)        # tower B site, a little further from the crowd
    for j in range(4):                    # four demand clusters near A, all within B's reach too
        Gm.add_node(10 + j, x=10.0 + j, y=0.0)
    D = [{"node": 10 + j, "w": 10} for j in range(4)]
    C = [{"node": 0, "tier": "mid", "cost": 1.0, "mask": 0b1111},
         {"node": 1, "tier": "mid", "cost": 1.0, "mask": 0b1111}]
    sites, mbps, caps = [0, 1], 1, [20, 20]     # 40 Mbps demand, 20 + 20 capacity
    cust = P.assign_customers(Gm, C, D, sites)
    assert cust[0] == 40 and cust[1] == 0, "greedy should pile everyone onto the nearer tower"
    load, unserved = P.assign_customers_flow(Gm, C, D, sites, caps, mbps)
    assert unserved == 0, "flow should split the crowd and serve all 40 Mbps"
    assert load == [20, 20] or sorted(load) == [20, 20]
    assert all(load[k] <= caps[k] for k in range(2))


def test_edmonds_karp_flows_dict_is_exact_on_a_dag():
    """assign_customers_flow reads per-arc flows out of this; verify conservation at each node."""
    cap = {"s": {"a": 5, "b": 3}, "a": {"t": 4}, "b": {"t": 6}, "t": {}}
    flows = {}
    total, _ = P.edmonds_karp(cap, "s", "t", flows=flows)
    assert total == 7
    assert flows[("s", "a")] + flows[("s", "b")] == total
    assert flows[("a", "t")] + flows[("b", "t")] == total
    assert flows[("s", "a")] == flows[("a", "t")]          # conservation at a
    assert flows[("s", "b")] == flows[("b", "t")]          # conservation at b
    for (u, v), f in flows.items():
        assert 0 <= f <= cap[u][v]


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
