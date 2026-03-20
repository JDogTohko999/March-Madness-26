"""
v2/main.py — Orchestrator for the 2026 March Madness bracket optimizer.

─── TUNABLE SETTINGS ────────────────────────────────────────────────────────
  POOL_SIZES          : which pool sizes to simulate
  N_SIMS              : tournament simulations per (champion, pool_size) pair
                        5000 recommended; 1000 for a fast test run
  TOP_N_CANDIDATES    : how many champion candidates to evaluate (by KenPom rank)
  VIRGINIA_MULTIPLIER : float ≥ 1.0
                        1.0  = national pool (no adjustment)
                        3.0  = moderately UVA-heavy local pool
                        5.0  = very UVA-heavy pool (small local bracket group)
  RUN_LOCAL_SIM       : True = also run a second simulation with VIRGINIA_MULTIPLIER
                        applied (produces the Virginia comparison chart)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Tunable settings ─────────────────────────────────────────────────────────
POOL_SIZES          = [10, 20, 50, 250]
N_SIMS              = 5000
TOP_N_CANDIDATES    = 15
VIRGINIA_MULTIPLIER = 1.0   # 1.0 = national pool; >1 = more Virginia picks
RUN_LOCAL_SIM       = True  # set False to skip the UVA comparison sim
LOCAL_VA_MULT       = 4.0   # multiplier used for the "local" comparison run
RANDOM_SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

# Add v2/ to path so imports work when running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from data import load_all
from pool_sim import (
    run_all_pools, build_value_table, build_round_value_table,
    print_summary_tables,
)
from viz import render_all


def print_header():
    print("=" * 66)
    print("  2026 NCAA MARCH MADNESS BRACKET OPTIMIZER  v2")
    print("=" * 66)
    print(f"  Pool sizes:       {POOL_SIZES}")
    print(f"  Simulations:      {N_SIMS:,} per (champion × pool size)")
    print(f"  Candidates:       top {TOP_N_CANDIDATES} by KenPom")
    print(f"  Virginia mult:    {VIRGINIA_MULTIPLIER:.1f}x  "
          f"({'national' if VIRGINIA_MULTIPLIER == 1 else 'local-adjusted'})")
    print(f"  Local sim:        {'yes (mult=' + str(LOCAL_VA_MULT) + 'x)' if RUN_LOCAL_SIM else 'no'}")
    print()


def print_round_recommendations(round_value_df: pd.DataFrame):
    """Print value-pick recommendations for Elite 8 onward."""
    print("\n" + "="*66)
    print("  LATE-ROUND VALUE PICKS  (seeds 1-4, KenPom vs Yahoo)")
    print("  Assumes bracket-coherent picks throughout")
    print("="*66)

    ROUND_LABELS = {3: "Elite 8 (80 pts)", 4: "Final Four (160 pts)",
                    5: "Champ Game (160 pts)", 6: "Champion (320 pts)"}

    for rnum in [3, 4, 5, 6]:
        sub = round_value_df[
            (round_value_df["round"] == rnum) &
            round_value_df["value_ratio"].notna() &
            (round_value_df["kenpom_prob"] > 0.01)
        ].sort_values("value_ratio", ascending=False)

        print(f"\n  {ROUND_LABELS[rnum]}")
        print(f"  {'Team':22s} {'Seed':>4}  {'KenPom':>8} {'Yahoo':>8} {'Value':>7}  Status")
        print("  " + "-"*60)

        for _, row in sub.head(8).iterrows():
            vr = row["value_ratio"]
            status = ("GREAT VALUE" if vr >= 1.5 else
                      "undervalued" if vr >= 1.1 else
                      "OVERVALUED"  if vr < 0.75 else "~fair")
            print(f"  {row['team']:22s} {int(row['seed']):>4}  "
                  f"{row['kenpom_prob']:>8.1%} {row['yahoo_pct']:>8.1%} "
                  f"{vr:>7.2f}x  {status}")


def print_virginia_note(pool_results_nat: dict, pool_results_loc: dict,
                         pool_sizes: list, value_df: pd.DataFrame):
    """Print the UVA local pool strategy note."""
    print("\n" + "="*66)
    print("  CHARLOTTESVILLE / UVA LOCAL POOL ADJUSTMENT")
    print(f"  Virginia multiplier applied: {LOCAL_VA_MULT:.1f}x")
    print("="*66)

    va_row = value_df[value_df["team"] == "Virginia"]
    if not va_row.empty:
        va = va_row.iloc[0]
        print(f"\n  Virginia:  KenPom={va['kenpom_champ']:.1%}  "
              f"National Yahoo={va['public_pct']:.1%}  "
              f"Value={va['value_ratio']:.2f}x")

    print(f"\n  In a UVA-heavy pool, Virginia's effective pick% is ~"
          f"{LOCAL_VA_MULT:.0f}x higher than national average.")
    print("  This makes Virginia a WORSE pick locally because opponent correlation")
    print("  rises: if Virginia wins, many people in your local pool also have them.")

    print(f"\n  {'Pool':>6}  {'National best':^22}  {'Local best':^22}  Diff")
    print("  " + "-"*60)
    for ps in pool_sizes:
        nat_best = max(pool_results_nat.get(ps, {}).items(),
                       key=lambda x: x[1], default=("?", 0))
        loc_best = max(pool_results_loc.get(ps, {}).items(),
                       key=lambda x: x[1], default=("?", 0))
        changed = "← changes!" if nat_best[0] != loc_best[0] else ""
        print(f"  {ps:>6}  {nat_best[0]:<22}  {loc_best[0]:<22}  {changed}")


def main():
    t0 = time.time()
    print_header()

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("Step 1: Loading data")
    raw = load_all(rng, verbose=True)
    kenpom_df    = raw["kenpom_df"]
    yahoo_rounds = raw["yahoo_rounds"]
    betting_probs = raw["betting_probs"]
    print()

    # ── Step 2: Value metrics ─────────────────────────────────────────────────
    print("Step 2: Computing value metrics")
    value_df       = build_value_table(kenpom_df, yahoo_rounds, betting_probs)
    round_value_df = build_round_value_table(kenpom_df, yahoo_rounds)

    print(f"  {len(value_df)} teams in value table")
    print(f"  Top 5 by actual prob: "
          f"{', '.join(value_df.head(5)['team'].tolist())}")

    # ── Step 3: National pool simulation ─────────────────────────────────────
    print(f"\nStep 3: National pool simulation  (virginia_mult={VIRGINIA_MULTIPLIER:.1f}x)")
    pool_results_nat = run_all_pools(
        kenpom_df=kenpom_df,
        yahoo_rounds=yahoo_rounds,
        pool_sizes=POOL_SIZES,
        n_sims=N_SIMS,
        top_n=TOP_N_CANDIDATES,
        virginia_mult=VIRGINIA_MULTIPLIER,
        seed=RANDOM_SEED,
    )

    # ── Step 4: Local (UVA) pool simulation ───────────────────────────────────
    pool_results_loc = None
    if RUN_LOCAL_SIM:
        print(f"\nStep 4: Local UVA pool simulation  (virginia_mult={LOCAL_VA_MULT:.1f}x)")
        pool_results_loc = run_all_pools(
            kenpom_df=kenpom_df,
            yahoo_rounds=yahoo_rounds,
            pool_sizes=POOL_SIZES,
            n_sims=N_SIMS,
            top_n=TOP_N_CANDIDATES,
            virginia_mult=LOCAL_VA_MULT,
            seed=RANDOM_SEED + 9999,
        )
    else:
        print("\nStep 4: Skipping local sim (RUN_LOCAL_SIM=False)")

    # ── Step 5: Print reports ─────────────────────────────────────────────────
    print_summary_tables(pool_results_nat, value_df, POOL_SIZES)
    print_round_recommendations(round_value_df)
    if pool_results_loc:
        print_virginia_note(pool_results_nat, pool_results_loc, POOL_SIZES, value_df)

    # ── Step 6: Save CSVs ─────────────────────────────────────────────────────
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    value_df.to_csv(out / "value_metrics.csv", index=False)
    round_value_df.to_csv(out / "round_value_metrics.csv", index=False)

    all_rows = []
    for ps, res in pool_results_nat.items():
        for team, win_p in res.items():
            all_rows.append({"pool_size": ps, "team": team, "pool_win_prob": win_p})
    pd.DataFrame(all_rows).to_csv(out / "pool_results_national.csv", index=False)

    if pool_results_loc:
        loc_rows = []
        for ps, res in pool_results_loc.items():
            for team, win_p in res.items():
                loc_rows.append({"pool_size": ps, "team": team, "pool_win_prob": win_p})
        pd.DataFrame(loc_rows).to_csv(out / "pool_results_local.csv", index=False)

    print(f"\n  CSVs saved to {out}/")

    # ── Step 7: Visualizations ────────────────────────────────────────────────
    render_all(
        pool_results=pool_results_nat,
        value_df=value_df,
        round_value_df=round_value_df,
        pool_sizes=POOL_SIZES,
        pool_results_local=pool_results_loc,
    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min. "
          f"All outputs in v2/outputs/")
    print("=" * 66)


if __name__ == "__main__":
    main()
