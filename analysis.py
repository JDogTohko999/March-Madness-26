"""
Value metric computation and output generation for 2026 March Madness optimization.
Produces tables, charts, and recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

POOL_SIZE_LABELS = {
    10:   "Small (~10)",
    20:   "Small-Med (~20)",
    50:   "Medium (~50)",
    100:  "Large (~100)",
    500:  "Very Large (~500)",
    1000: "Huge (~1000)",
}

# Historical seed performance: P(seed reaches Final Four) from 1985-2024
HISTORICAL_FF_PCT = {
    1: 0.417, 2: 0.216, 3: 0.121, 4: 0.085,
    5: 0.045, 6: 0.039, 7: 0.028, 8: 0.021,
    9: 0.013, 10: 0.018, 11: 0.036,
    12: 0.018, 13: 0.003, 14: 0.001, 15: 0.001, 16: 0.000,
}

# Upset picks with strong value ratios (hand-computed)
UPSET_ALERTS = [
    {
        "round": "Round of 64",
        "matchup": "12 vs 5",
        "lower_seed": "Grand Canyon",
        "higher_seed": "Vanderbilt",
        "region": "West",
        "lower_seed_win_pct": 0.38,
        "public_pick_lower": 0.25,
        "value_ratio": 1.52,
        "rationale": "12-seeds win ~35% historically; Vanderbilt struggled down stretch"
    },
    {
        "round": "Round of 64",
        "matchup": "11 vs 6",
        "lower_seed": "San Diego State",
        "higher_seed": "Ole Miss",
        "region": "West",
        "lower_seed_win_pct": 0.42,
        "public_pick_lower": 0.28,
        "value_ratio": 1.50,
        "rationale": "SDSU historically dominant in tournament; Mountain West toughness"
    },
    {
        "round": "Round of 64",
        "matchup": "10 vs 7",
        "lower_seed": "Colorado State",
        "higher_seed": "Georgia",
        "region": "East",
        "lower_seed_win_pct": 0.44,
        "public_pick_lower": 0.32,
        "value_ratio": 1.38,
        "rationale": "Colorado State MWC runner-up with strong NET; Georgia inconsistent"
    },
    {
        "round": "Round of 64",
        "matchup": "9 vs 8",
        "lower_seed": "Utah State",
        "higher_seed": "Boise State",
        "region": "West",
        "lower_seed_win_pct": 0.48,
        "public_pick_lower": 0.38,
        "value_ratio": 1.26,
        "rationale": "Utah State 27-7 MWC champ; near coin-flip but public undervalues"
    },
    {
        "round": "Round of 32",
        "matchup": "Iowa State deep run",
        "lower_seed": "Iowa State",
        "higher_seed": "Michigan (potential R2)",
        "region": "Midwest",
        "lower_seed_win_pct": 0.41,
        "public_pick_lower": 0.26,
        "value_ratio": 1.58,
        "rationale": "Michigan LJ Cason injury; Iowa State battle-tested in Big 12; "
                     "public heavily on Michigan as 1-seed"
    },
]


def compute_value_metrics(
    teams: dict,
    betting_probs: dict,
    public_pick_pct: dict,
    sim_probs: dict,
) -> pd.DataFrame:
    """
    Build the master value metrics DataFrame.
    Columns: team, seed, region, actual_prob, public_pct, value_ratio, odds
    """
    rows = []
    for team, info in teams.items():
        # Actual prob: blend of betting odds (50%) and simulation (50%)
        bet_p = betting_probs.get(team, 0)
        sim_p = sim_probs.get(team, 0)
        actual_p = 0.5 * bet_p + 0.5 * sim_p if (bet_p + sim_p) > 0 else sim_p

        pub_p = public_pick_pct.get(team, 0.001)  # floor at 0.1%
        value_ratio = actual_p / pub_p if pub_p > 0 else 0

        rows.append({
            "team": team,
            "seed": info["seed"],
            "region": info["region"],
            "record": info["record"],
            "actual_prob": actual_p,
            "betting_prob": bet_p,
            "sim_prob": sim_p,
            "public_pct": pub_p,
            "value_ratio": value_ratio,
        })

    df = pd.DataFrame(rows).sort_values("actual_prob", ascending=False).reset_index(drop=True)
    return df


def build_summary_table(
    value_df: pd.DataFrame,
    pool_results: dict,
    pool_sizes: list,
) -> dict:
    """
    For each pool size, build a ranked table of top 5 champion picks.
    Returns {pool_size: pd.DataFrame}
    """
    tables = {}
    for ps in pool_sizes:
        ps_results = pool_results.get(ps, {})
        rows = []
        for _, row in value_df.iterrows():
            team = row["team"]
            win_p = ps_results.get(team, np.nan)
            baseline = 1 / ps
            rows.append({
                "Team": team,
                "Seed": int(row["seed"]),
                "Region": row["region"],
                "Actual Win%": f"{row['actual_prob']:.1%}",
                "Public Pick%": f"{row['public_pct']:.1%}",
                "Value Ratio": f"{row['value_ratio']:.2f}",
                "Pool Win Prob": f"{win_p:.3%}" if not np.isnan(win_p) else "N/A",
                "Baseline (1/n)": f"{baseline:.3%}",
                "vs Baseline": f"{win_p/baseline:.2f}x" if not np.isnan(win_p) else "N/A",
                "_pool_win": win_p if not np.isnan(win_p) else -1,
            })
        df = (pd.DataFrame(rows)
              .sort_values("_pool_win", ascending=False)
              .drop(columns=["_pool_win"])
              .head(8)
              .reset_index(drop=True))
        df.index += 1
        tables[ps] = df
    return tables


def print_recommendations(value_df: pd.DataFrame, pool_results: dict):
    """Print human-readable recommendations to console."""
    print("\n" + "="*70)
    print("  2026 NCAA MARCH MADNESS BRACKET OPTIMIZATION REPORT")
    print("="*70)

    print("\n── CHAMPION PICKS BY POOL SIZE ──────────────────────────────────────")
    pool_sizes = sorted(pool_results.keys())
    for ps in pool_sizes:
        ps_results = pool_results[ps]
        ranked = sorted(ps_results.items(), key=lambda x: -x[1])
        if not ranked:
            continue
        best_team, best_prob = ranked[0]
        baseline = 1 / ps
        row = value_df[value_df["team"] == best_team].iloc[0]
        print(f"\n  Pool size ~{ps:>4d}:  Pick {best_team} (seed {int(row['seed'])}, {row['region']})")
        print(f"    Actual win%={row['actual_prob']:.1%}  Public pick%={row['public_pct']:.1%}  "
              f"Value={row['value_ratio']:.2f}x")
        print(f"    Expected pool win: {best_prob:.2%}  (baseline: {baseline:.2%}, "
              f"{best_prob/baseline:.1f}x better)")
        # Show next 2 alternatives
        alts = [f"{t} ({ps_results[t]:.2%})" for t, _ in ranked[1:3]]
        print(f"    Alternatives: {', '.join(alts)}")

    print("\n── FINAL FOUR RECOMMENDATIONS ───────────────────────────────────────")
    ff_recs = generate_ff_recommendations(value_df)
    for pool_tier, rec in ff_recs.items():
        print(f"\n  {pool_tier}:")
        for region, team in rec.items():
            row = value_df[value_df["team"] == team]
            if len(row):
                row = row.iloc[0]
                print(f"    {region:8s}: {team} (seed {int(row['seed'])}, "
                      f"val={row['value_ratio']:.2f}x)")

    print("\n── UPSET ALERTS (High-Value Early Round Picks) ──────────────────────")
    for alert in UPSET_ALERTS:
        print(f"\n  [{alert['round']}] {alert['matchup']} — {alert['region']}")
        print(f"  Pick: {alert['lower_seed']} over {alert['higher_seed']}")
        print(f"  Win%: {alert['lower_seed_win_pct']:.0%} | Public: {alert['public_pick_lower']:.0%} | "
              f"Value: {alert['value_ratio']:.2f}x")
        print(f"  Why: {alert['rationale']}")

    print("\n── CHARLOTTESVILLE / UVA LOCAL POOL ADJUSTMENT ─────────────────────")
    va_row = value_df[value_df["team"] == "Virginia"]
    if len(va_row):
        va = va_row.iloc[0]
        print(f"\n  Virginia is a 3-seed (Midwest) with:")
        print(f"    Actual championship prob: {va['actual_prob']:.1%}")
        print(f"    National public pick%: {va['public_pct']:.1%}")
        print(f"    National value ratio: {va['value_ratio']:.2f}x")
        print()
        print("  In UVA-heavy pools, expect Virginia pick% to be 3-5x higher than")
        print("  national average (maybe 6-10% vs ~2% nationally).")
        print()
        print("  STRATEGY:")
        print("    Local small pool (~10):  AVOID Virginia as champion — overvalued locally.")
        print("    Local medium pool (~50): Especially avoid Virginia. Pick Michigan or")
        print("                             Iowa State to differentiate.")
        print("    National pool:           Virginia has decent value ratio nationally —")
        print("                             fine as a sleeper pick in large pools.")
        print()
        print("  Also: Michigan LJ Cason injury is well-known — Michigan may be")
        print("        more overvalued than odds suggest. Adjust if injury worsens.")


def generate_ff_recommendations(value_df: pd.DataFrame) -> dict:
    """
    Generate Final Four recommendations for each pool tier.
    Strategy: include at least one high-value-ratio team per pool tier.
    """
    # Sort by value ratio within each region
    region_best = {}
    for region in ["East", "West", "Midwest", "South"]:
        reg = value_df[value_df["region"] == region].sort_values("value_ratio", ascending=False)
        top = reg[reg["seed"] <= 6].head(3)["team"].tolist()
        region_best[region] = top

    # Small pool: pick favorites (best actual prob per region)
    small_ff = {}
    for region in ["East", "West", "Midwest", "South"]:
        reg = value_df[value_df["region"] == region].sort_values("actual_prob", ascending=False)
        small_ff[region] = reg.iloc[0]["team"] if len(reg) else "?"

    # Medium pool: mix favorites and best-value picks
    medium_ff = dict(small_ff)
    # Replace one region with highest value non-favorite
    # Find region where top value pick differs most from top actual prob pick
    for region in ["Midwest", "South"]:
        reg = value_df[value_df["region"] == region].sort_values("value_ratio", ascending=False)
        top_val = reg[reg["seed"] <= 4].iloc[0]["team"] if len(reg) else small_ff[region]
        medium_ff[region] = top_val

    # Large pool: push toward 2-4 seeds with high value
    large_ff = {}
    for region in ["East", "West", "Midwest", "South"]:
        reg = value_df[value_df["region"] == region]
        # Best value among seeds 2-5
        candidates = reg[reg["seed"].between(2, 5)].sort_values("value_ratio", ascending=False)
        large_ff[region] = candidates.iloc[0]["team"] if len(candidates) else small_ff[region]

    return {
        "Small pools (10-20)":   small_ff,
        "Medium pools (50)":     medium_ff,
        "Large pools (100+)":    large_ff,
    }


def plot_value_scatter(value_df: pd.DataFrame):
    """Scatter plot: Actual Win% vs Public Pick% with 45-degree fair-value line."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by seed tier
    seed_colors = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#1f77b4", 99: "#9467bd"}

    plot_df = value_df[value_df["actual_prob"] > 0.001].copy()

    for _, row in plot_df.iterrows():
        seed = int(row["seed"])
        color_key = seed if seed <= 4 else 99
        color = seed_colors.get(color_key, "#7f7f7f")
        ax.scatter(row["public_pct"], row["actual_prob"], color=color,
                   s=80, alpha=0.8, zorder=3)

        # Label top teams
        if row["actual_prob"] > 0.03 or abs(row["value_ratio"] - 1) > 0.4:
            ax.annotate(
                f"{row['team']}\n({int(row['seed'])})",
                xy=(row["public_pct"], row["actual_prob"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7.5, alpha=0.9
            )

    # 45-degree fair-value line
    max_val = max(plot_df["public_pct"].max(), plot_df["actual_prob"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1.5,
            label="Fair value (ratio = 1.0)")

    # Shade undervalued region
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                    alpha=0.04, color="green")
    ax.fill_between([0, max_val], [0, 0], [0, max_val],
                    alpha=0.04, color="red")
    ax.text(0.02, 0.95, "← UNDERVALUED (good picks)", transform=ax.transAxes,
            color="green", fontsize=9, va="top")
    ax.text(0.55, 0.05, "OVERVALUED (bad picks in pools) →", transform=ax.transAxes,
            color="red", fontsize=9)

    # Legend for seed colors
    patches = [
        mpatches.Patch(color="#d62728", label="1-seeds"),
        mpatches.Patch(color="#ff7f0e", label="2-seeds"),
        mpatches.Patch(color="#2ca02c", label="3-seeds"),
        mpatches.Patch(color="#1f77b4", label="4-seeds"),
        mpatches.Patch(color="#9467bd", label="5+ seeds"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    ax.set_xlabel("Public Pick % (ESPN Tournament Challenge)", fontsize=11)
    ax.set_ylabel("Actual Championship Probability (betting + sim blend)", fontsize=11)
    ax.set_title("2026 NCAA Tournament: Actual vs. Perceived Championship Probability\n"
                 "Above the line = undervalued; Below the line = overvalued", fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = OUTPUT_DIR / "01_value_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pool_win_probs(pool_results: dict, value_df: pd.DataFrame):
    """Bar chart of expected pool win probability by champion pick, faceted by pool size."""
    pool_sizes = sorted(pool_results.keys())
    n = len(pool_sizes)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, ps in enumerate(pool_sizes):
        ax = axes[idx]
        ps_data = pool_results[ps]
        baseline = 1 / ps

        # Top 10 teams by pool win prob
        ranked = sorted(ps_data.items(), key=lambda x: -x[1])[:10]
        teams_plot = [t for t, _ in ranked]
        probs_plot = [p for _, p in ranked]

        # Color bars by value ratio
        colors = []
        for t in teams_plot:
            row = value_df[value_df["team"] == t]
            vr = row["value_ratio"].values[0] if len(row) else 1.0
            if vr >= 1.3:
                colors.append("#2ca02c")  # green = great value
            elif vr >= 1.0:
                colors.append("#ff7f0e")  # orange = slight value
            else:
                colors.append("#d62728")  # red = overvalued

        bars = ax.barh(teams_plot[::-1], probs_plot[::-1], color=colors[::-1], alpha=0.85)

        # Baseline line
        ax.axvline(baseline, color="black", linestyle="--", alpha=0.6, linewidth=1.2,
                   label=f"Baseline 1/n = {baseline:.1%}")

        ax.set_title(f"Pool size {ps}\n({POOL_SIZE_LABELS[ps]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Expected Win Probability", fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

        # Label bars
        for bar, prob in zip(bars, probs_plot[::-1]):
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.2%}", va="center", fontsize=7)

    # Color legend
    green_patch = mpatches.Patch(color="#2ca02c", label="Value ratio ≥ 1.3x (great)")
    orange_patch = mpatches.Patch(color="#ff7f0e", label="Value ratio 1.0-1.3x (ok)")
    red_patch = mpatches.Patch(color="#d62728", label="Value ratio < 1.0x (avoid)")
    fig.legend(handles=[green_patch, orange_patch, red_patch],
               loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0))

    fig.suptitle("2026 March Madness: Expected Pool Win Probability by Champion Pick",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = OUTPUT_DIR / "02_pool_win_probs.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_value_ratio_table(value_df: pd.DataFrame):
    """Horizontal bar chart of value ratios for top 20 teams."""
    top20 = value_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    teams = top20["team"].tolist()[::-1]
    ratios = top20["value_ratio"].tolist()[::-1]
    seeds = top20["seed"].tolist()[::-1]

    colors = ["#2ca02c" if r >= 1.0 else "#d62728" for r in ratios]
    bars = ax.barh(teams, ratios, color=colors, alpha=0.85)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Fair value = 1.0")

    for bar, ratio, seed in zip(bars, ratios, seeds):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.2f}x  (seed {int(seed)})",
                va="center", fontsize=8.5)

    ax.set_xlabel("Value Ratio (Actual Probability / Public Pick %)", fontsize=11)
    ax.set_title("2026 Tournament Value Ratios — Top 20 Championship Contenders\n"
                 "Green = undervalued (good picks); Red = overvalued (bad in competitive pools)",
                 fontsize=11)
    ax.set_xlim(0, max(ratios) * 1.25)
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = OUTPUT_DIR / "03_value_ratios.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_csv_outputs(value_df: pd.DataFrame, summary_tables: dict):
    """Save CSV files for all key tables."""
    path = OUTPUT_DIR / "value_metrics.csv"
    value_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    for ps, df in summary_tables.items():
        path = OUTPUT_DIR / f"summary_pool_{ps}.csv"
        df.to_csv(path)
        print(f"  Saved: {path}")
