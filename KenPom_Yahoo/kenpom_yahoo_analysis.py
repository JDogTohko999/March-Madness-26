"""
KenPom vs Yahoo Pick Distribution Analysis — 2026 NCAA Tournament
Compares KenPom actual win probabilities against Yahoo public pick percentages
for Sweet 16 onward, to find value picks.

Assumes chalk (no upsets) through Round of 32.
Focuses on: Elite 8, Final Four, Championship game, Champion.
"""

import requests
import re
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── KenPom column → Yahoo round mapping ──────────────────────────────────────
# KenPom: probability (out of 1,000,000) of reaching that round
# Yahoo:  % of brackets picking that team to advance past that round
ROUND_MAP = {
    "Elite8":  {"kenpom_col": "Elite8",  "yahoo_round": "3", "label": "Elite 8",         "points": 80},
    "Final4":  {"kenpom_col": "Final4",  "yahoo_round": "4", "label": "Final Four",       "points": 160},
    "Final":   {"kenpom_col": "Final",   "yahoo_round": "5", "label": "Championship Game","points": 160},
    "Champ":   {"kenpom_col": "Champ",   "yahoo_round": "6", "label": "Champion",         "points": 320},
}

# Our team name normalizer: KenPom names → canonical names
KENPOM_NAME_MAP = {
    "Iowa St.":    "Iowa State",
    "Connecticut": "UConn",
    "Michigan St.":"Michigan State",
    "Gonzaga":     "Gonzaga",
    "St. John's":  "St. John's",
    "Virginia":    "Virginia",
    "Arkansas":    "Arkansas",
    "Miami FL":    "Miami (FL)",
    "N.C. State":  "NC State",
    "Texas A&M":   "Texas A&M",
    "Saint Mary's":"St. Mary's",
    "Saint Louis": "Saint Louis",
    "South Florida":"South Florida",
    "Cal Baptist": "California Baptist",
    "North Dakota St.": "North Dakota State",
    "Kennesaw St.": "Kennesaw State",
    "Tennessee St.":"Tennessee State",
    "Miami OH":    "Miami (OH)",
}

YAHOO_KEY_TO_TEAM = {
    "ncaab.t.173": "Duke",
    "ncaab.t.17":  "Arizona",
    "ncaab.t.357": "Michigan",
    "ncaab.t.210": "Florida",
    "ncaab.t.254": "Houston",
    "ncaab.t.129": "UConn",
    "ncaab.t.277": "Iowa State",
    "ncaab.t.474": "Purdue",
    "ncaab.t.358": "Michigan State",
    "ncaab.t.267": "Illinois",
    "ncaab.t.233": "Gonzaga",
    "ncaab.t.618": "Virginia",
    "ncaab.t.400": "Nebraska",
    "ncaab.t.6":   "Alabama",
    "ncaab.t.287": "Kansas",
    "ncaab.t.19":  "Arkansas",
    "ncaab.t.615": "Vanderbilt",
    "ncaab.t.534": "St. John's",
    "ncaab.t.592": "Texas Tech",
    "ncaab.t.657": "Wisconsin",
    "ncaab.t.68":  "BYU",
    "ncaab.t.606": "UCLA",
    "ncaab.t.443": "Ohio St.",
    "ncaab.t.367": "Missouri",
    "ncaab.t.230": "Georgia",
    "ncaab.t.314": "Louisville",
    "ncaab.t.355": "Miami (FL)",
    "ncaab.t.613": "VCU",
    "ncaab.t.580": "Tennessee",
    "ncaab.t.276": "Iowa",
    "ncaab.t.617": "Villanova",
    "ncaab.t.292": "Kentucky",
    "ncaab.t.413": "North Carolina",
    "ncaab.t.345": "McNeese",
    "ncaab.t.418": "Northern Iowa",
    "ncaab.t.611": "Utah State",
    "ncaab.t.3":   "Akron",
    "ncaab.t.120": "Clemson",
    "ncaab.t.515": "SMU",
    "ncaab.t.411": "NC State",
    "ncaab.t.515": "SMU",
    "ncaab.t.585": "Texas",
    "ncaab.t.103": "UCF",
    "ncaab.t.576": "TCU",
    "ncaab.t.526": "South Florida",
    "ncaab.t.536": "Saint Louis",
    "ncaab.t.538": "St. Mary's",
    "ncaab.t.587": "Texas A&M",
    "ncaab.t.502": "Santa Clara",
    "ncaab.t.252": "Hofstra",
    "ncaab.t.1009":"California Baptist",
    "ncaab.t.246": "Hawaii",
    "ncaab.t.596": "Troy",
    "ncaab.t.356": "Miami (OH)",
    "ncaab.t.220": "Furman",
    "ncaab.t.460": "Pennsylvania",
    "ncaab.t.263": "Idaho",
    "ncaab.t.303": "Lehigh",
    "ncaab.t.1374":"Kennesaw State",
    "ncaab.t.1455":"North Dakota State",
    "ncaab.t.582": "Tennessee State",
    "ncaab.t.346": "UMBC",
    "ncaab.t.512": "Siena",
    "ncaab.t.660": "Wright State",
    "ncaab.t.1112":"High Point",
    "ncaab.t.1237":"Queens",
}


def fetch_kenpom() -> pd.DataFrame:
    """Fetch KenPom 2026 tournament predictions from GitHub."""
    url = "https://raw.githubusercontent.com/kenpom/tourney_predictions/main/ncaa_2026.csv"
    print("Fetching KenPom data...")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Drop any trailing incomplete rows
    df = df.dropna(subset=["Champ"])
    # Normalize team names
    df["Team"] = df["Team"].apply(lambda t: KENPOM_NAME_MAP.get(t, t))
    # Convert to probabilities (divide by 1,000,000)
    for col in ["Rd2", "Swt16", "Elite8", "Final4", "Final", "Champ"]:
        df[col] = df[col].astype(float) / 1_000_000
    print(f"  {len(df)} teams loaded from KenPom")
    return df


def fetch_yahoo_rounds() -> dict:
    """Fetch all 6 rounds of Yahoo pick distribution."""
    url = "https://tournament.fantasysports.yahoo.com/mens-basketball-bracket/pickdistribution"
    print("Fetching Yahoo pick distribution...")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    big = None
    for s in soup.find_all("script"):
        content = s.string or ""
        if len(content) > 100000 and "pickDistribution" in content:
            big = content
            break

    if not big:
        raise RuntimeError("Yahoo App script not found")

    round_blocks = re.findall(
        r'\{"roundId":"(\d+)","distributionByTeam":\[([^\]]+)\]', big
    )

    results = {}
    for round_id, teams_str in round_blocks:
        entries = re.findall(
            r'"editorialTeamKey":"(ncaab\.t\.\d+)","percentage":([\d.]+)', teams_str
        )
        round_data = {}
        for key, pct in entries:
            name = YAHOO_KEY_TO_TEAM.get(key)
            if name:
                round_data[name] = float(pct) / 100.0
        results[round_id] = round_data

    print(f"  {len(round_blocks)} rounds scraped from Yahoo")
    return results


def build_merged_df(kenpom: pd.DataFrame, yahoo_rounds: dict) -> pd.DataFrame:
    """
    Merge KenPom and Yahoo data into a single DataFrame with value ratios
    for each round from Elite 8 onward.
    """
    rows = []
    for _, krow in kenpom.iterrows():
        team = krow["Team"]
        seed = int(krow["seed"]) if not pd.isna(krow["seed"]) else 99
        region = krow["Region"]

        row = {"team": team, "seed": seed, "region": region}

        for rkey, rmeta in ROUND_MAP.items():
            kp_prob = krow[rmeta["kenpom_col"]]
            yahoo_pct = yahoo_rounds.get(rmeta["yahoo_round"], {}).get(team, None)

            row[f"kenpom_{rkey}"] = kp_prob
            row[f"yahoo_{rkey}"] = yahoo_pct
            if yahoo_pct and yahoo_pct > 0:
                row[f"value_{rkey}"] = kp_prob / yahoo_pct
            else:
                row[f"value_{rkey}"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_round_analysis(df: pd.DataFrame, round_key: str, top_n: int = 12):
    """Print value analysis for a specific round."""
    meta = ROUND_MAP[round_key]
    label = meta["label"]
    pts = meta["points"]

    kp_col  = f"kenpom_{round_key}"
    yh_col  = f"yahoo_{round_key}"
    val_col = f"value_{round_key}"

    sub = df[df[val_col].notna() & (df[kp_col] > 0.005) & (df["seed"] <= 4)].copy()
    sub = sub.sort_values(val_col, ascending=False)

    print(f"\n{'='*68}")
    print(f"  {label.upper()}  ({pts} pts)  —  KenPom vs Yahoo")
    print(f"{'='*68}")
    print(f"  {'Team':22s} {'Seed':>4}  {'KenPom%':>8}  {'Yahoo%':>8}  {'Value':>7}  {'Status'}")
    print(f"  {'-'*60}")

    for _, row in sub.head(top_n).iterrows():
        vr = row[val_col]
        status = "GREAT VALUE" if vr >= 1.5 else ("undervalued" if vr >= 1.1 else
                 ("OVERVALUED" if vr < 0.7 else "slight OV" if vr < 0.9 else "~fair"))
        print(f"  {row['team']:22s} {int(row['seed']):>4}  "
              f"{row[kp_col]:>8.1%}  {row[yh_col]:>8.1%}  {vr:>7.2f}x  {status}")

    # Also show most overvalued
    print(f"\n  --- Most Overvalued (avoid in competitive pools) ---")
    for _, row in sub.sort_values(val_col).head(5).iterrows():
        vr = row[val_col]
        print(f"  {row['team']:22s} {int(row['seed']):>4}  "
              f"{row[kp_col]:>8.1%}  {row[yh_col]:>8.1%}  {vr:>7.2f}x")


def plot_round_scatter(df: pd.DataFrame, round_key: str):
    """Scatter: KenPom prob vs Yahoo pick% for one round, with fair-value line."""
    meta = ROUND_MAP[round_key]
    kp_col  = f"kenpom_{round_key}"
    yh_col  = f"yahoo_{round_key}"
    val_col = f"value_{round_key}"

    sub = df[df[val_col].notna() & (df[kp_col] > 0.005) & (df["seed"] <= 4)].copy()

    fig, ax = plt.subplots(figsize=(9, 7))

    seed_colors = {1:"#d62728", 2:"#ff7f0e", 3:"#2ca02c", 4:"#1f77b4", 99:"#9467bd"}
    for _, row in sub.iterrows():
        c = seed_colors.get(int(row["seed"]) if int(row["seed"]) <= 4 else 99, "#7f7f7f")
        ax.scatter(row[yh_col], row[kp_col], color=c, s=70, alpha=0.8, zorder=3)
        # Label teams with notable value
        if row[val_col] > 1.2 or row[val_col] < 0.7 or row[kp_col] > 0.08:
            ax.annotate(
                f"{row['team']} ({int(row['seed'])})",
                xy=(row[yh_col], row[kp_col]),
                xytext=(5, 4), textcoords="offset points", fontsize=7.5, alpha=0.9
            )

    mx = max(sub[yh_col].max(), sub[kp_col].max()) * 1.15
    ax.plot([0, mx], [0, mx], "k--", alpha=0.4, lw=1.5, label="Fair value")
    ax.fill_between([0, mx], [0, mx], [mx, mx], alpha=0.04, color="green")
    ax.fill_between([0, mx], [0, 0], [0, mx], alpha=0.04, color="red")
    ax.text(0.02, 0.97, "UNDERVALUED", transform=ax.transAxes,
            color="green", fontsize=9, va="top", fontweight="bold")
    ax.text(0.60, 0.04, "OVERVALUED", transform=ax.transAxes,
            color="red", fontsize=9, fontweight="bold")

    patches = [
        mpatches.Patch(color="#d62728", label="1-seeds"),
        mpatches.Patch(color="#ff7f0e", label="2-seeds"),
        mpatches.Patch(color="#2ca02c", label="3-seeds"),
        mpatches.Patch(color="#1f77b4", label="4-seeds"),
        mpatches.Patch(color="#9467bd", label="5+ seeds"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel(f"Yahoo Public Pick % (Round {meta['yahoo_round']})", fontsize=11)
    ax.set_ylabel(f"KenPom Win Probability", fontsize=11)
    ax.set_title(f"2026 NCAA Tournament — {meta['label']}\n"
                 f"KenPom Actual Probability vs Yahoo Public Pick %", fontsize=11)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    plt.tight_layout()

    path = OUTPUT_DIR / f"kenpom_yahoo_{round_key.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def plot_value_heatmap(df: pd.DataFrame):
    """
    Heatmap of value ratios across all late rounds for top teams.
    Shows at a glance which teams are undervalued across multiple rounds.
    """
    val_cols = [f"value_{r}" for r in ROUND_MAP]
    labels   = [ROUND_MAP[r]["label"] for r in ROUND_MAP]

    # Keep seeds 1-4 with KenPom Champ > 0.5% and at least one Yahoo value
    sub = df[(df["kenpom_Champ"] > 0.005) & (df["seed"] <= 4)].copy()
    sub = sub.dropna(subset=val_cols, how="all")
    sub = sub.sort_values("kenpom_Champ", ascending=False).head(20)

    mat = sub[val_cols].fillna(1.0).values
    teams = sub["team"].tolist()
    seeds = sub["seed"].tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=2.0, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels([f"{t} ({s})" for t, s in zip(teams, seeds)], fontsize=9)

    # Annotate cells
    for i in range(len(teams)):
        for j in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.6 < v < 1.6 else "white")

    plt.colorbar(im, ax=ax, label="Value Ratio (KenPom / Yahoo)  —  Green = Undervalued")
    ax.set_title("Value Ratio Heatmap — KenPom vs Yahoo Pick %\n"
                 "Top 20 teams by KenPom championship probability\n"
                 "(Assumes chalk through Round of 32)", fontsize=11)
    plt.tight_layout()

    path = OUTPUT_DIR / "kenpom_yahoo_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def print_final_recommendations(df: pd.DataFrame):
    """Print a concise actionable pick guide for Sweet 16 onward."""
    print("\n" + "="*68)
    print("  ACTIONABLE RECOMMENDATIONS — SWEET 16 ONWARD")
    print("  (Assuming chalk picks through Round of 32)")
    print("="*68)

    for rkey, meta in ROUND_MAP.items():
        val_col = f"value_{rkey}"
        kp_col  = f"kenpom_{rkey}"
        yh_col  = f"yahoo_{rkey}"

        sub = df[df[val_col].notna() & (df[kp_col] > 0.01) & (df["seed"] <= 4)].copy()
        best = sub.sort_values(val_col, ascending=False).head(3)

        print(f"\n  {meta['label']} ({meta['points']} pts):")
        print(f"  {'Pick':22s}  {'Seed':>4}  {'KenPom':>7}  {'Yahoo':>7}  {'Value':>7}")
        for _, row in best.iterrows():
            print(f"  {row['team']:22s}  {int(row['seed']):>4}  "
                  f"{row[kp_col]:>7.1%}  {row[yh_col]:>7.1%}  {row[val_col]:>7.2f}x")

    print()
    print("  KEY INSIGHT: Teams with value ratio > 1.3x are statistically worth")
    print("  picking over the public consensus pick, even if their absolute win")
    print("  probability is lower. The pool win math rewards uniqueness.")
    print()

    # Region-by-region summary
    print("  REGION SUMMARY (best undervalued team per region, Elite 8+):")
    for region, rname in [("E","East"), ("W","West"), ("MW","Midwest"), ("S","South")]:
        reg = df[df["region"] == region].copy()
        reg = reg[reg["value_Elite8"].notna() & (reg["kenpom_Elite8"] > 0.01) & (reg["seed"] <= 4)]
        if len(reg):
            best = reg.sort_values("value_Elite8", ascending=False).iloc[0]
            print(f"  {rname:8s}: {best['team']} (seed {int(best['seed'])})  "
                  f"E8 value={best['value_Elite8']:.2f}x  "
                  f"FF value={best.get('value_Final4', float('nan')):.2f}x")


def main():
    # 1. Fetch data
    kenpom = fetch_kenpom()
    yahoo_rounds = fetch_yahoo_rounds()

    # 2. Merge
    print("\nMerging datasets...")
    df = build_merged_df(kenpom, yahoo_rounds)
    print(f"  {len(df)} teams in merged dataset")
    print(f"  Teams with Yahoo+KenPom overlap: "
          f"{df['value_Champ'].notna().sum()}")

    # 3. Save merged CSV
    csv_path = OUTPUT_DIR / "kenpom_yahoo_merged.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")

    # 4. Round-by-round analysis
    for rkey in ROUND_MAP:
        print_round_analysis(df, rkey)

    # 5. Final recommendations
    print_final_recommendations(df)

    # 6. Charts
    print("\nGenerating charts...")
    for rkey in ROUND_MAP:
        plot_round_scatter(df, rkey)
    plot_value_heatmap(df)

    print("\nDone. All outputs saved to KenPom_Yahoo/")


if __name__ == "__main__":
    main()
