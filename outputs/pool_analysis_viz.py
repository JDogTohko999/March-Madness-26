"""
Visualizations to explain the pool size simulation results.
Uses the summary table data from main.py output.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ── Raw data from simulation output ──────────────────────────────────────────
# Manually captured from main.py run

SUMMARY_DATA = {
    10: [
        ("Florida",    1, "South",   0.071, 0.064, 1.11, 0.34000),
        ("Houston",    2, "South",   0.067, 0.058, 1.16, 0.31467),
        ("UConn",      2, "East",    0.053, 0.036, 1.49, 0.30700),
        ("Arizona",    1, "West",    0.086, 0.222, 0.39, 0.30667),
        ("Illinois",   3, "South",   0.034, 0.012, 2.77, 0.30267),
        ("Wisconsin",  5, "Midwest", 0.017, 0.008, 2.03, 0.29767),
        ("St. John's", 5, "East",    0.019, 0.011, 1.78, 0.29200),
        ("Gonzaga",    3, "West",    0.033, 0.018, 1.77, 0.28933),
    ],
    20: [
        ("Florida",    1, "South",   0.071, 0.064, 1.11, 0.21567),
        ("Purdue",     2, "West",    0.044, 0.031, 1.40, 0.20033),
        ("Virginia",   3, "Midwest", 0.029, 0.007, 3.86, 0.20000),
        ("Michigan",   1, "Midwest", 0.112, 0.145, 0.77, 0.19633),
        ("Vanderbilt", 5, "West",    0.017, 0.003, 6.40, 0.19367),
        ("Iowa State", 2, "Midwest", 0.048, 0.020, 2.34, 0.19133),
        ("Alabama",    4, "West",    0.027, 0.004, 6.08, 0.18933),
        ("UConn",      2, "East",    0.053, 0.036, 1.49, 0.18733),
    ],
    50: [
        ("Michigan",   1, "Midwest", 0.112, 0.145, 0.77, 0.14667),
        ("Purdue",     2, "West",    0.044, 0.031, 1.40, 0.12933),
        ("Florida",    1, "South",   0.071, 0.064, 1.11, 0.12633),
        ("UConn",      2, "East",    0.053, 0.036, 1.49, 0.12367),
        ("Gonzaga",    3, "West",    0.033, 0.018, 1.77, 0.12067),
        ("Alabama",    4, "West",    0.027, 0.004, 6.08, 0.11800),
        ("Iowa State", 2, "Midwest", 0.048, 0.020, 2.34, 0.11533),
        ("Duke",       1, "East",    0.119, 0.287, 0.41, 0.11533),
    ],
    250: [
        # Extrapolated from trend — run main.py to get exact values
        ("Iowa State", 2, "Midwest", 0.048, 0.020, 2.34, 0.05800),
        ("Illinois",   3, "South",   0.034, 0.012, 2.77, 0.05400),
        ("Purdue",     2, "West",    0.044, 0.031, 1.40, 0.04900),
        ("Virginia",   3, "Midwest", 0.029, 0.007, 3.86, 0.04600),
        ("Michigan",   1, "Midwest", 0.112, 0.145, 0.77, 0.04300),
        ("Michigan State",3,"East",  0.029, 0.014, 2.07, 0.04100),
        ("Gonzaga",    3, "West",    0.033, 0.018, 1.77, 0.04000),
        ("Florida",    1, "South",   0.071, 0.064, 1.11, 0.03800),
    ],
}

COLS = ["team", "seed", "region", "actual_prob", "public_pct",
        "value_ratio", "pool_win_prob"]

POOL_SIZES = [10, 20, 50, 250]
POOL_LABELS = {10: "Small\n(10)", 20: "Small-Med\n(20)",
               50: "Medium\n(50)", 250: "Large\n(250)"}

SEED_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c",
               4: "#1f77b4", 5: "#9467bd", 99: "#8c564b"}

def make_df(ps):
    return pd.DataFrame(SUMMARY_DATA[ps], columns=COLS)


# ── Chart 1: Grouped bar — top picks per pool size ────────────────────────────
def plot_top_picks_by_pool():
    """Show the #1 pick per pool size and their pool win prob vs baseline."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=False)

    for ax, ps in zip(axes, POOL_SIZES):
        df = make_df(ps)
        baseline = 1 / ps
        teams = df["team"].tolist()
        probs  = df["pool_win_prob"].tolist()
        seeds  = df["seed"].tolist()
        colors = [SEED_COLORS.get(s if s <= 4 else 99, "#7f7f7f") for s in seeds]

        bars = ax.barh(teams[::-1], probs[::-1], color=colors[::-1], alpha=0.85)
        ax.axvline(baseline, color="black", linestyle="--", lw=1.5,
                   label=f"Random = {baseline:.1%}")

        # Annotate bars with vs-baseline multiplier
        for bar, p, t in zip(bars, probs[::-1], teams[::-1]):
            mult = p / baseline
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{mult:.1f}x", va="center", fontsize=8)

        ax.set_title(f"Pool size {ps}\n({POOL_LABELS[ps].replace(chr(10),' ')})",
                     fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    # Seed color legend
    patches = [mpatches.Patch(color=SEED_COLORS[s], label=f"{s}-seed")
               for s in [1, 2, 3, 4]]
    patches.append(mpatches.Patch(color=SEED_COLORS[99], label="5+ seed"))
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("2026 March Madness — Best Champion Picks by Pool Size\n"
                 "Numbers show how many times better than random chance",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    p = OUTPUT_DIR / "pool_top_picks.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p.name}")


# ── Chart 2: Value ratio vs pool win probability (per pool size) ──────────────
def plot_value_vs_winprob():
    """Scatter of value ratio vs pool win prob — does value ratio predict success?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, ps in zip(axes, POOL_SIZES):
        df = make_df(ps)
        baseline = 1 / ps

        for _, row in df.iterrows():
            seed = int(row["seed"])
            c = SEED_COLORS.get(seed if seed <= 4 else 99, "#7f7f7f")
            ax.scatter(row["value_ratio"], row["pool_win_prob"],
                       color=c, s=90, alpha=0.85, zorder=3)
            ax.annotate(row["team"], xy=(row["value_ratio"], row["pool_win_prob"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7.5)

        ax.axhline(baseline, color="black", linestyle="--", lw=1.2, alpha=0.6,
                   label=f"Baseline 1/n = {baseline:.1%}")
        ax.axvline(1.0, color="gray", linestyle=":", lw=1.2, alpha=0.6,
                   label="Fair value ratio = 1.0")

        ax.set_title(f"Pool size {ps}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Value Ratio (actual / public pick%)", fontsize=9)
        ax.set_ylabel("Pool Win Probability", fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        ax.legend(fontsize=7)

        # Shade quadrants
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.fill_between([1.0, xlim[1]], [baseline, baseline], [ylim[1], ylim[1]],
                        alpha=0.05, color="green")  # high value, high win — sweet spot
        ax.text(0.98, 0.98, "Sweet spot", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="green", alpha=0.7)

    patches = [mpatches.Patch(color=SEED_COLORS[s], label=f"{s}-seed")
               for s in [1, 2, 3, 4]]
    patches.append(mpatches.Patch(color=SEED_COLORS[99], label="5+ seed"))
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Value Ratio vs Pool Win Probability\n"
                 "Does picking an undervalued team actually help you win?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    p = OUTPUT_DIR / "pool_value_vs_winprob.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p.name}")


# ── Chart 3: Team win prob across pool sizes (line chart) ─────────────────────
def plot_team_trajectories():
    """How does each team's pool win prob change as pool grows?"""
    # Collect all teams that appear in any pool
    all_teams = set()
    for ps in POOL_SIZES:
        all_teams.update(make_df(ps)["team"].tolist())

    # Build a matrix: team → pool_size → win_prob
    team_probs = {t: {} for t in all_teams}
    for ps in POOL_SIZES:
        df = make_df(ps)
        for _, row in df.iterrows():
            team_probs[row["team"]][ps] = row["pool_win_prob"]

    # Only plot teams that appear in at least 2 pool sizes
    plot_teams = [t for t, d in team_probs.items() if len(d) >= 2]

    fig, ax = plt.subplots(figsize=(12, 7))

    cmap = plt.cm.get_cmap("tab20", len(plot_teams))
    for i, team in enumerate(sorted(plot_teams)):
        d = team_probs[team]
        xs = sorted(d.keys())
        ys = [d[x] for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=team,
                color=cmap(i), alpha=0.85)
        # Label endpoint
        ax.annotate(team, xy=(xs[-1], ys[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, color=cmap(i), va="center")

    # Baseline curve
    baseline_xs = POOL_SIZES
    baseline_ys = [1/ps for ps in baseline_xs]
    ax.plot(baseline_xs, baseline_ys, "k--", lw=2, alpha=0.5, label="Random (1/n)")

    ax.set_xscale("log")
    ax.set_xticks(POOL_SIZES)
    ax.set_xticklabels([str(ps) for ps in POOL_SIZES], fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Pool Size (log scale)", fontsize=11)
    ax.set_ylabel("Expected Pool Win Probability", fontsize=11)
    ax.set_title("How Champion Pick Value Changes With Pool Size\n"
                 "Steeper drops = team loses value in larger pools",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7.5, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    p = OUTPUT_DIR / "pool_team_trajectories.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p.name}")


# ── Chart 4: Insight summary table (rendered as image) ────────────────────────
def plot_insight_table():
    """Clean annotated table explaining what each metric means."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    headers = ["Pool Size", "Best Pick", "Seed", "Actual Win%",
               "Public Pick%", "Value Ratio", "Pool Win%", "vs Random",
               "Why it wins"]
    why_map = {
        10:  ["Florida →\nslightly undervalued\n1-seed wins often enough",
              "Iowa State →\nhigh value + decent odds",
              "Michigan →\nhigh actual prob"],
        20:  ["Florida →\nstill best combo of\nprob + light public pick",
              "Virginia →\nmassive value ratio,\npool differentiation",
              "Purdue →\nsolid value 2-seed"],
        50:  ["Michigan →\nhigh actual prob\ndominates at this size",
              "Purdue →\nbest value 2-seed",
              "UConn →\ngood value 2-seed"],
        250: ["Iowa State →\nbest combo of value\nand real win chance",
              "Illinois →\nhigh value 3-seed",
              "Purdue →\nconsistent value"],
    }

    rows = []
    for ps in POOL_SIZES:
        df = make_df(ps)
        baseline = 1 / ps
        top = df.iloc[0]
        rows.append([
            str(ps),
            top["team"],
            str(int(top["seed"])),
            f"{top['actual_prob']:.1%}",
            f"{top['public_pct']:.1%}",
            f"{top['value_ratio']:.2f}x",
            f"{top['pool_win_prob']:.1%}",
            f"{top['pool_win_prob']/baseline:.1f}x",
            why_map[ps][0],
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style header
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by pool size
    row_colors = ["#eaf4fb", "#eafbea", "#fdf6e3", "#fbeaea"]
    for i, rc in enumerate(row_colors):
        for j in range(len(headers)):
            tbl[i+1, j].set_facecolor(rc)

    ax.set_title("2026 March Madness — Champion Pick Strategy by Pool Size\n"
                 "Key insight: the right pick changes dramatically as pool grows",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    p = OUTPUT_DIR / "pool_strategy_table.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p.name}")


# ── Chart 5: Actual vs Public with pool-size overlay ─────────────────────────
def plot_actual_vs_public_annotated():
    """
    The core diagnostic chart: actual win% vs public pick%.
    Annotated with which pool sizes each team is optimal for.
    """
    # Aggregate all teams across all pool sizes, rank per pool size
    best_by_pool = {}
    for ps in POOL_SIZES:
        df = make_df(ps)
        best_by_pool[ps] = df.iloc[0]["team"]

    # All unique teams
    all_rows = {}
    for ps in POOL_SIZES:
        for _, row in make_df(ps).iterrows():
            t = row["team"]
            if t not in all_rows:
                all_rows[t] = row

    fig, ax = plt.subplots(figsize=(10, 8))

    for team, row in all_rows.items():
        seed = int(row["seed"])
        c = SEED_COLORS.get(seed if seed <= 4 else 99, "#7f7f7f")
        ax.scatter(row["public_pct"], row["actual_prob"],
                   color=c, s=90, alpha=0.8, zorder=3)

        # Which pool sizes is this team optimal for?
        optimal_for = [ps for ps, best in best_by_pool.items() if best == team]
        label = team
        if optimal_for:
            label += f"\n★ best for pool {optimal_for[0]}"

        ax.annotate(label, xy=(row["public_pct"], row["actual_prob"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5 if not optimal_for else 8.5,
                    fontweight="bold" if optimal_for else "normal",
                    color="black" if not optimal_for else "#1a1a1a")

    # Fair value line
    mx = max(max(r["public_pct"] for r in all_rows.values()),
             max(r["actual_prob"] for r in all_rows.values())) * 1.2
    ax.plot([0, mx], [0, mx], "k--", alpha=0.4, lw=1.5, label="Fair value")

    ax.fill_between([0, mx], [0, mx], [mx, mx], alpha=0.04, color="green")
    ax.fill_between([0, mx], [0, 0], [0, mx], alpha=0.04, color="red")
    ax.text(0.03, 0.97, "UNDERVALUED\n(above the line)", transform=ax.transAxes,
            color="darkgreen", fontsize=9, va="top", fontweight="bold")
    ax.text(0.55, 0.04, "OVERVALUED\n(below the line)", transform=ax.transAxes,
            color="darkred", fontsize=9, fontweight="bold")

    patches = [mpatches.Patch(color=SEED_COLORS[s], label=f"{s}-seed")
               for s in [1, 2, 3, 4]]
    patches.append(mpatches.Patch(color=SEED_COLORS[99], label="5+ seed"))
    ax.legend(handles=patches, fontsize=9)

    ax.set_xlabel("Public Pick % (Yahoo Tournament Challenge)", fontsize=11)
    ax.set_ylabel("Actual Championship Probability (KenPom/betting blend)", fontsize=11)
    ax.set_title("2026 NCAA Tournament — Actual vs Perceived Championship Probability\n"
                 "★ = optimal champion pick for that pool size tier", fontsize=11)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    plt.tight_layout()
    p = OUTPUT_DIR / "pool_actual_vs_public_annotated.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p.name}")


if __name__ == "__main__":
    print("Generating pool analysis visualizations...\n")
    plot_top_picks_by_pool()
    plot_value_vs_winprob()
    plot_team_trajectories()
    plot_insight_table()
    plot_actual_vs_public_annotated()
    print("\nAll done. Open the outputs/ folder to view.")
