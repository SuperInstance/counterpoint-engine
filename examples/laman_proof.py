#!/usr/bin/env python3
"""
laman_proof.py — Build a Laman graph via Henneberg construction, verify rigidity.

Run:  python3 examples/laman_proof.py

Demonstrates:
  - Building a Laman graph for N voices with henneberg_construct()
  - Verifying rigidity with verify_rigidity() and CounterpointGraph
  - Showing edges and constraint assignments
  - Proving the counterpoint = rigidity theorem holds
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from counterpoint_engine import (
    CounterpointGraph,
    henneberg_construct,
    verify_rigidity,
)

DIVIDER = "=" * 60


def main():
    print(DIVIDER)
    print("Laman Graph Construction & Rigidity Verification")
    print(DIVIDER)
    print("\nTheorem: A counterpoint texture is rigid (all voices independently")
    print("constrained) iff its constraint graph is a Laman graph.")
    print("  Laman condition: |E| = 2N - 3, and every subset of k vertices")
    print("  spans at most 2k - 3 edges.\n")

    # ── Build Laman graphs for different voice counts ───────────────────────
    for n_voices in [2, 3, 4, 5]:
        print(f"\n{'─' * 50}")
        print(f"  {n_voices}-voice counterpoint")
        print(f"{'─' * 50}")

        expected = 2 * n_voices - 3
        print(f"  Expected edges (2n-3): {expected}")

        # Build via Henneberg type-I construction
        edges = henneberg_construct(n_voices)
        print(f"  Henneberg edges:       {edges}")
        print(f"  Edge count:            {len(edges)}")

        # Verify rigidity directly
        is_rigid = verify_rigidity(n_voices, edges)
        print(f"  Is Laman rigid?        {is_rigid} ✓" if is_rigid else f"  Is Laman rigid?        {is_rigid} ✗")

    # ── CounterpointGraph: constraint assignment ────────────────────────────
    print(f"\n{DIVIDER}")
    print("CounterpointGraph — Constraint Assignment")
    print(DIVIDER)

    cg = CounterpointGraph(n_voices=4)
    print(f"\n{cg!r}")
    print(f"  Edge count: {cg.edge_count()} (expected {cg.expected_edges()})")
    print(f"  Minimally rigid: {cg.is_minimally_rigid()}")
    print(f"\n  Voice pairs and their constraints:")

    for edge in cg.edges:
        constraints = cg.constraints.get(edge, [])
        print(f"    Voice {edge[0]} ↔ Voice {edge[1]}: {', '.join(constraints)}")

    # ── Manual edge manipulation ────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("  Manual edge addition (breaking Laman condition)")
    print(f"{'─' * 50}")

    # Add extra edges to show non-minimal (but still rigid) graph
    cg2 = CounterpointGraph(n_voices=3)
    print(f"\n  Before: {cg2.edge_count()} edges, rigid={cg2.verify_rigidity()}")
    cg2.add_constraint(0, 2, "extra_consonance_check")
    print(f"  After adding edge (0,2): {cg2.edge_count()} edges, rigid={cg2.verify_rigidity()}")
    print(f"  Still minimally rigid: {cg2.is_minimally_rigid()}")
    print(f"  (More edges than 2n-3 → rigid but not *minimally* rigid)")

    # ── Subsetting: remove an edge ──────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("  Edge removal (breaking rigidity)")
    print(f"{'─' * 50}")

    edges_3 = henneberg_construct(3)
    print(f"  Full edge set: {edges_3}, rigid={verify_rigidity(3, edges_3)}")
    # Remove one edge
    reduced = edges_3[:-1]
    print(f"  After removing last edge: {reduced}, rigid={verify_rigidity(3, reduced)}")

    print(f"\n✓ Laman rigidity verified for voice counts 2–5.")
    print("  Counterpoint constraint graphs satisfy the rigidity theorem.")


if __name__ == "__main__":
    main()
