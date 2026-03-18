"""
Main entry point for 2026 March Madness Bracket Optimization.
Run this script to generate all outputs.
"""

import time
import warnings
warnings.filterwarnings("ignore")

from data_collection import collect_all_data
from simulation import (
    build_team_win_probs,
    simulate_champion_probs,
    run_pool_analysis,
)
from analysis import (
    compute_value_metrics,
    build_summary_table,
    print_recommendations,
    plot_value_scatter,
    plot_pool_win_probs,
    plot_value_ratio_table,
    save_csv_outputs,
    OUTPUT_DIR,
)

POOL_SIZES = [10, 20, 50, 100, 500, 1000]
N_SIMS = 3000  # reduce for speed; increase to 5000+ for more stable results


def main():
    t0 = time.time()
    print("=" * 60)
    print("  2026 NCAA MARCH MADNESS BRACKET OPTIMIZER")
    print("=" * 60)

    # ── Step 1: Collect Data ───────────────────────────────────────
    data = collect_all_data()
    teams = data["teams"]
    betting_probs = data["betting_probs"]
    public_pick_pct = data["public_pick_pct"]

    # ── Step 2: Simulation-derived championship probabilities ──────
    print("\nStep 2: Simulating tournament (5000 runs)...")
    strengths = build_team_win_probs(teams, betting_probs)
    sim_probs = simulate_champion_probs(teams, strengths, n_sims=5000, seed=42)

    # ── Step 3: Compute value metrics ─────────────────────────────
    print("\nStep 3: Computing value metrics...")
    value_df = compute_value_metrics(teams, betting_probs, public_pick_pct, sim_probs)

    # Print quick value table
    print("\nTop 20 teams by actual championship probability:")
    print(f"{'Team':20s} {'Seed':>4} {'Region':8s} {'Actual':>8} {'Public':>8} "
          f"{'Value':>7} {'Betting':>8} {'Sim':>8}")
    print("-" * 80)
    for _, row in value_df.head(20).iterrows():
        print(f"{row['team']:20s} {int(row['seed']):>4} {row['region']:8s} "
              f"{row['actual_prob']:>8.2%} {row['public_pct']:>8.2%} "
              f"{row['value_ratio']:>7.2f} {row['betting_prob']:>8.2%} "
              f"{row['sim_prob']:>8.2%}")

    # ── Step 4: Pool simulations ───────────────────────────────────
    print(f"\nStep 4: Running pool win probability simulations ({N_SIMS} sims)...")
    pool_results = run_pool_analysis(
        teams=teams,
        betting_probs=betting_probs,
        public_pick_pct=public_pick_pct,
        pool_sizes=POOL_SIZES,
        top_n_teams=20,
        n_sims=N_SIMS,
    )

    # ── Step 5: Build summary tables ──────────────────────────────
    print("\nStep 5: Building summary tables...")
    summary_tables = build_summary_table(value_df, pool_results, POOL_SIZES)

    print("\n── SUMMARY TABLES ────────────────────────────────────────────────────")
    for ps in POOL_SIZES:
        from analysis import POOL_SIZE_LABELS
        print(f"\n  Pool Size: {ps} ({POOL_SIZE_LABELS[ps]})")
        print(summary_tables[ps].to_string())

    # ── Step 6: Recommendations ───────────────────────────────────
    print_recommendations(value_df, pool_results)

    # ── Step 7: Visualizations ────────────────────────────────────
    print(f"\nStep 6: Generating visualizations → {OUTPUT_DIR}/")
    plot_value_scatter(value_df)
    plot_pool_win_probs(pool_results, value_df)
    plot_value_ratio_table(value_df)

    # ── Step 8: Save CSVs ─────────────────────────────────────────
    print("\nStep 7: Saving CSV outputs...")
    save_csv_outputs(value_df, summary_tables)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. All outputs saved to: {OUTPUT_DIR}/")
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
