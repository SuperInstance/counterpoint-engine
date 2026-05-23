#!/usr/bin/env python3
"""
laman_rigidity.py — Demonstrate Laman rigidity checks on constraint graphs.

Demonstrates:
- Building Laman graphs via Henneberg type-I construction
- Verifying the Laman condition (2n-3 edges + subset condition)
- Showing how constraint edges relate to voice independence
- Comparing rigid vs non-rigid constraint graphs

Run:  python3 examples/laman_rigidity.py
"""

from counterpoint_engine import (
    CounterpointGraph,
    henneberg_construct,
    verify_rigidity,
)

# --- Henneberg construction for various N ---
print("HENNEBERG TYPE-I CONSTRUCTION")
print("=" * 50)

for n in range(2, 8):
    edges = henneberg_construct(n)
    is_rigid = verify_rigidity(n, edges)
    expected = 2 * n - 3
    print(f"N={n} voices: {len(edges)} edges (expected {expected}), rigid={is_rigid}")
    print(f"  Edges: {edges}")
    print()

# --- CounterpointGraph convenience ---
print("COUNTERPOINT GRAPH (auto-constructed)")
print("=" * 50)

for n in [2, 3, 4, 5]:
    g = CounterpointGraph(n)
    print(f"N={n}: {g}")
    print(f"  Edges: {g.edges}")
    print(f"  Voice pairs with constraints: {g.voice_pairs()}")
    print()

# --- Show that removing edges breaks rigidity ---
print("RIGIDITY VIOLATION (too few edges)")
print("=" * 50)

# 4 voices: need 2*4-3 = 5 edges for rigidity
full_edges = henneberg_construct(4)
print(f"Full graph (4v, {len(full_edges)} edges): rigid={verify_rigidity(4, full_edges)}")

# Remove one edge — should break rigidity
reduced = full_edges[:-1]
print(f"Reduced graph (4v, {len(reduced)} edges): rigid={verify_rigidity(4, reduced)}")

# 3 edges for 4 vertices — clearly not rigid
sparse = [(0, 1), (1, 2), (2, 3)]
print(f"Sparse graph (4v, {len(sparse)} edges): rigid={verify_rigidity(4, sparse)}")

# --- Custom constraint assignment ---
print()
print("CUSTOM CONSTRAINT GRAPH")
print("=" * 50)

g = CounterpointGraph(3)
print(f"Default constraints: {g.constraints}")

# Add a custom constraint
g.add_constraint(0, 2, "voice_crossing_check")
print(f"After adding custom constraint: {g.constraints}")
print(f"Edge count: {g.edge_count()}")
print(f"Is minimally rigid: {g.is_minimally_rigid()}")

# --- The theoretical claim ---
print()
print("THEORETICAL CLAIM")
print("=" * 50)
print("Counterpoint = Constraint Satisfaction = Laman Rigidity")
print()
print("Each voice is a vertex in a Laman graph.")
print("Each contrapuntal constraint is an edge.")
print(f"For N voices, minimally rigid = exactly 2N-3 edges.")
print(f"This ensures every voice is independently constrained,")
print(f"with no redundant constraints and no unconstrained degrees of freedom.")
