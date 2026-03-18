"""
v2/viz.py — All visualizations. Reads exclusively from live simulation output.
No hardcoded data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 4: "#1f77b4", 99: "#9467bd"}
POOL_LABELS = {10: "Small (10)", 20: "Small-Med (20)",
               50: "Medium (50)", 250: "Large (250)"}

def _seed_color(seed: int) -> str:
    return SEED_COLORS.get(seed if seed <= 4 else 99, "#7f7f7f")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Actual vs Public scatter (champion picks)
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_public(value_df: pd.DataFrame, pool_results: dict,
                           pool_sizes: list):
    """
    Scatter: actual championship prob (y) vs public pick % (x).
    Teams above the 45-degree line are undervalued.
    Star markers indicate optimal pick per pool size.
    """
    best_by_pool = {ps: max(res, key=res.get) for ps, res in pool_results.items()
                    if res}

    fig, ax = plt.subplots(figsize=(11, 8))
    plot_df = value_df[value_df["actual_prob"] > 0.003].copy()

    for _, row in plot_df.iterrows():
        c = _seed_color(int(row["seed"]))
        ax.scatter(row["public_pct"], row["actual_prob"],
                   color=c, s=75, alpha=0.8, zorder=3)

        # Label if notable
        optimal_pools = [ps for ps, best in best_by_pool.items() if best == row["team"]]
        label = row["team"]
        if optimal_pools:
            label += f"\n★ ps={optimal_pools[0]}"
        if row["actual_prob"] > 0.04 or abs(row["value_ratio"] - 1) > 0.5 or optimal_pools:
            ax.annotate(label,
                        xy=(row["public_pct"], row["actual_prob"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5,
                        fontweight="bold" if optimal_pools else "normal")

    # Fair-value line
    mx = max(plot_df["public_pct"].max(), plot_df["actual_prob"].max()) * 1.15
    ax.plot([0, mx], [0, mx], "k--", alpha=0.35, lw=1.5, label="Fair value (ratio=1)")
    ax.fill_between([0, mx], [0, mx], [mx, mx], alpha=0.04, color="green")
    ax.fill_between([0, mx], [0, 0], [0, mx], alpha=0.04, color="red")
    ax.text(0.02, 0.97, "UNDERVALUED", transform=ax.transAxes,
            color="darkgreen", fontsize=9, va="top", fontweight="bold")
    ax.text(0.60, 0.03, "OVERVALUED", transform=ax.transAxes,
            color="darkred", fontsize=9, fontweight="bold")

    patches = [mpatches.Patch(color=SEED_COLORS[s], label=f"{s}-seed")
               for s in [1, 2, 3, 4]]
    patches.append(mpatches.Patch(color=SEED_COLORS[99], label="5+ seed"))
    ax.legend(handles=patches, fontsize=9)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Yahoo Public Pick % (Champion)", fontsize=11)
    ax.set_ylabel("Actual Championship Probability (60% KenPom + 40% Betting)", fontsize=11)
    ax.set_title("2026 NCAA Tournament — Actual vs Perceived Championship Probability\n"
                 "★ = optimal champion pick for that pool size  |  "
                 "Above line = undervalued", fontsize=11)
    plt.tight_layout()

    p = OUTPUT_DIR / "01_actual_vs_public.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pool win probability bar chart (faceted by pool size)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pool_win_bars(pool_results: dict, value_df: pd.DataFrame, pool_sizes: list):
    """
    Horizontal bar chart of expected pool win probability, one panel per pool size.
    Bars colored green/orange/red by value ratio tier.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, ps in zip(axes, pool_sizes):
        baseline = 1 / ps
        ps_res   = pool_results.get(ps, {})
        ranked   = sorted(ps_res.items(), key=lambda x: -x[1])[:10]
        teams_p  = [t for t, _ in ranked]
        probs_p  = [p for _, p in ranked]

        colors = []
        for t in teams_p:
            row = value_df[value_df["team"] == t]
            vr = row["value_ratio"].values[0] if not row.empty else 1.0
            colors.append("#2ca02c" if vr >= 1.3 else
                          "#ff7f0e" if vr >= 1.0 else "#d62728")

        bars = ax.barh(teams_p[::-1], probs_p[::-1],
                       color=colors[::-1], alpha=0.85)
        ax.axvline(baseline, color="black", linestyle="--", lw=1.3, alpha=0.7,
                   label=f"Random 1/n = {baseline:.1%}")

        # Multiplier annotations
        for bar, p in zip(bars, probs_p[::-1]):
            ax.text(bar.get_width() + 0.0008,
                    bar.get_y() + bar.get_height() / 2,
                    f"{p/baseline:.1f}x", va="center", fontsize=8)

        ax.set_title(f"Pool size {ps}  ({POOL_LABELS[ps]})",
                     fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
        ax.tick_params(labelsize=8.5)
        ax.legend(fontsize=7.5)

    green  = mpatches.Patch(color="#2ca02c", label="Value ratio ≥ 1.3x")
    orange = mpatches.Patch(color="#ff7f0e", label="Value ratio 1.0-1.3x")
    red    = mpatches.Patch(color="#d62728", label="Value ratio < 1.0x (overvalued)")
    fig.legend(handles=[green, orange, red], loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Expected Pool Win Probability by Champion Pick — All Pool Sizes\n"
                 "Numbers show improvement over random chance (1/n baseline)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    p = OUTPUT_DIR / "02_pool_win_bars.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Team trajectory — win prob vs pool size
# ─────────────────────────────────────────────────────────────────────────────

def plot_team_trajectories(pool_results: dict, value_df: pd.DataFrame, pool_sizes: list):
    """
    Line chart: each team's pool win probability as pool size grows.
    Teams that stay high across pool sizes are the most versatile picks.
    """
    # Gather all teams appearing in any pool's top 8
    featured = set()
    for ps in pool_sizes:
        res = pool_results.get(ps, {})
        top8 = sorted(res, key=res.get, reverse=True)[:8]
        featured.update(top8)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = matplotlib.colormaps.get_cmap("tab20")

    for i, team in enumerate(sorted(featured)):
        xs, ys = [], []
        for ps in pool_sizes:
            v = pool_results.get(ps, {}).get(team)
            if v is not None:
                xs.append(ps); ys.append(v)
        if len(xs) < 2:
            continue

        row = value_df[value_df["team"] == team]
        seed = int(row["seed"].values[0]) if not row.empty else 99
        c = _seed_color(seed)
        ax.plot(xs, ys, marker="o", lw=2, color=c, alpha=0.85, label=team)
        ax.annotate(f"{team} ({seed})",
                    xy=(xs[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=c, va="center")

    # Baseline curve
    ax.plot(pool_sizes, [1/ps for ps in pool_sizes], "k--",
            lw=2, alpha=0.45, label="Random (1/n)")

    ax.set_xscale("log")
    ax.set_xticks(pool_sizes)
    ax.set_xticklabels([str(ps) for ps in pool_sizes], fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
    ax.set_xlabel("Pool Size (log scale)", fontsize=11)
    ax.set_ylabel("Expected Pool Win Probability", fontsize=11)
    ax.set_title("Champion Pick Value Across Pool Sizes\n"
                 "Steeper drop = worse in large pools; "
                 "flat = robust pick regardless of pool size",
                 fontsize=11)
    plt.tight_layout()

    p = OUTPUT_DIR / "03_team_trajectories.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Value ratio heatmap (rounds 3-6, seeds 1-4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_round_value_heatmap(round_value_df: pd.DataFrame):
    """
    Heatmap of value ratios across Elite 8, Final Four, Champ Game, Champion
    for seeds 1-4. Green = undervalued, Red = overvalued.
    """
    ROUND_ORDER = [3, 4, 5, 6]
    ROUND_LABELS = {3: "Elite 8", 4: "Final Four", 5: "Champ Game", 6: "Champion"}

    pivot = round_value_df.pivot_table(
        index="team", columns="round", values="value_ratio"
    ).reindex(columns=ROUND_ORDER)

    # Sort by average value ratio descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # Add seed labels
    seed_map = round_value_df.drop_duplicates("team").set_index("team")["seed"].to_dict()
    ylabels = [f"{t}  ({seed_map.get(t,'?')})" for t in pivot.index]

    fig, ax = plt.subplots(figsize=(9, max(6, len(pivot) * 0.45)))
    mat = pivot.fillna(1.0).values

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=2.2, aspect="auto")

    ax.set_xticks(range(len(ROUND_ORDER)))
    ax.set_xticklabels([ROUND_LABELS[r] for r in ROUND_ORDER], fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(ylabels, fontsize=9)

    for i in range(len(pivot)):
        for j in range(len(ROUND_ORDER)):
            v = mat[i, j]
            color = "black" if 0.55 < v < 1.7 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    plt.colorbar(im, ax=ax, label="Value Ratio (KenPom / Yahoo)  —  Green = Undervalued")
    ax.set_title("Value Ratio Heatmap — Seeds 1-4 (Late Rounds)\n"
                 "KenPom actual probability vs Yahoo public pick %",
                 fontsize=11)
    plt.tight_layout()

    p = OUTPUT_DIR / "04_round_value_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Value ratio bar (champion level, all seeds)
# ─────────────────────────────────────────────────────────────────────────────

def plot_value_ratio_bars(value_df: pd.DataFrame):
    """Horizontal bar chart of champion-level value ratios, top 20 teams."""
    top = value_df.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 8))

    teams  = top["team"].tolist()[::-1]
    ratios = top["value_ratio"].tolist()[::-1]
    seeds  = top["seed"].tolist()[::-1]
    colors = [("#2ca02c" if r >= 1.0 else "#d62728") for r in ratios]

    bars = ax.barh(teams, ratios, color=colors, alpha=0.85)
    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="Fair value = 1.0")

    for bar, ratio, seed in zip(bars, ratios, seeds):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{ratio:.2f}x  (seed {int(seed)})",
                va="center", fontsize=8.5)

    ax.set_xlabel("Value Ratio (Actual Prob / Public Pick %)", fontsize=11)
    ax.set_title("Champion Value Ratios — Top 20 Teams\n"
                 "Green = undervalued  |  Red = overvalued in competitive pools",
                 fontsize=11)
    ax.set_xlim(0, max(ratios) * 1.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    p = OUTPUT_DIR / "05_value_ratio_bars.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Virginia local pool impact chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_virginia_impact(pool_results_national: dict, pool_results_local: dict,
                          pool_sizes: list, value_df: pd.DataFrame):
    """
    Side-by-side comparison of optimal picks in national vs UVA-local pools.
    Shows how Virginia's overrepresentation shifts the rankings.
    """
    fig, axes = plt.subplots(1, len(pool_sizes), figsize=(16, 6), sharey=False)

    for ax, ps in zip(axes, pool_sizes):
        nat = pool_results_national.get(ps, {})
        loc = pool_results_local.get(ps, {})

        # Top 6 teams from national results
        top_teams = [t for t, _ in sorted(nat.items(), key=lambda x: -x[1])[:6]]

        x = np.arange(len(top_teams))
        nat_vals = [nat.get(t, 0) for t in top_teams]
        loc_vals = [loc.get(t, 0) for t in top_teams]

        ax.bar(x - 0.2, nat_vals, 0.38, label="National pool", color="#1f77b4", alpha=0.8)
        ax.bar(x + 0.2, loc_vals, 0.38, label="Local (UVA) pool", color="#d62728", alpha=0.8)
        ax.axhline(1/ps, color="black", linestyle="--", lw=1, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(top_teams, rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
        ax.set_title(f"Pool size {ps}", fontsize=10, fontweight="bold")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("National vs UVA Local Pool — Champion Pick Strategy\n"
                 "Virginia overrepresentation shifts optimal picks",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    p = OUTPUT_DIR / "06_virginia_local_impact.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Master render
# ─────────────────────────────────────────────────────────────────────────────

def render_all(pool_results: dict, value_df: pd.DataFrame,
               round_value_df: pd.DataFrame, pool_sizes: list,
               pool_results_local: dict = None):
    print(f"\nGenerating visualizations → {OUTPUT_DIR}/")
    plot_actual_vs_public(value_df, pool_results, pool_sizes)
    plot_pool_win_bars(pool_results, value_df, pool_sizes)
    plot_team_trajectories(pool_results, value_df, pool_sizes)
    plot_round_value_heatmap(round_value_df)
    plot_value_ratio_bars(value_df)
    if pool_results_local:
        plot_virginia_impact(pool_results, pool_results_local, pool_sizes, value_df)
    print(f"  All charts saved.")
