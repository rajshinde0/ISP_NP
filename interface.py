"""Locked data interface between Person A (stages 0-2) and Person B (stages 3-5).

Plan of action, Section 3: "Before either of you writes stage code, agree on the shared data
format - in writing, today."  This module is that agreement, plus the tunable constants both
halves depend on.  Change it at a sync, never mid-stage.

Divergence from the plan's Section 3 wording, recorded deliberately so the report matches the
code:

    plan said                    code uses    why
    -------------------------------------------------------------------------------------
    eid                          (none)       towers and demand always travel as ordered
                                              lists, so the list index *is* the id
    position (lat/lon or x/y)    node         a graph node id; read coordinates back with
                                              G.nodes[node]["x" / "y" / "lat" / "lon"]
    build_cost                   cost         same meaning (k$), shorter key
    weight                       w            same meaning, shorter key
    -                            mask         stage-1 only: coverage bitmask over D.
                                              Stages 3-5 must ignore it.

The graph object itself is the plan's third contract item and is honoured exactly: stages 3-5
read the same NetworkX graph stage 0 built, never a reconstruction of it.  Every node carries
x / y in **metres** (projected) for distance maths and lat / lon in degrees for display; every
edge carries "length" in metres.
"""
from typing import TypedDict


class Demand(TypedDict):
    """One cluster of customers sitting on a road-network node."""
    node: int      # graph node id
    w: int         # number of customers (demand weight)


class Candidate(TypedDict):
    """One candidate tower site.  `mask` is added by make_instance, not by the caller."""
    node: int      # graph node id
    tier: str      # "mid" | "high"
    cost: float    # build cost, k$
    mask: int      # bitmask of the demand indices this site covers (stage 1 only)


# ---------------- tunables ----------------
# Coverage radius by tower tier, metres.  Doubled from the first cut (400/900): at the old
# radii a candidate reached so little of the map that make_instance had to discard 30-60% of
# the demand as unreachable, and ~11 of 16 candidates were *forced*, leaving stage 1 with no
# real combinatorial choice to make (exact and greedy returned identical costs on 8/10 seeds).
RADIUS = {"mid": 800.0, "high": 1800.0}

# Tower build cost by tier, k$ (before equipment).
BASE_COST = {"mid": 10.0, "high": 28.0}

# Stage 2 equipment catalogue: (cost k$, capacity Mbps), one 0/1 knapsack item each.
OPTIONS = [(2, 100), (4, 250), (7, 500), (10, 800)]

# Extra stage-2 budget a high-tier tower gets: it covers 5x the area, so it carries more load.
TIER_BONUS = {"mid": 0, "high": 6}

# Backhaul headroom: provision each tower for this multiple of its own customer load, because a
# tower on the backbone also has to relay its neighbours' traffic. Sizing every tower to exactly
# its own demand (headroom 1.0) leaves zero transit capacity, which starves the whole tree - the
# first cut of stage 4 served only 19% of demand for exactly this reason.
HEADROOM = 2.0

# Anchor for turning the synthetic grid's metric x/y into plottable lat/lon (central Pune).
# Display only - the OSM path carries its own real coordinates.
LAT0, LON0 = 18.5204, 73.8567
