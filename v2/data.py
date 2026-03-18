"""
v2/data.py — Data layer for 2026 March Madness bracket optimizer.

Sources:
  - KenPom tourney_predictions (GitHub): per-round reach probabilities for all 68 teams
  - Yahoo Fantasy pick distribution: 6-round public pick percentages
  - Hardcoded betting futures: fallback championship probabilities

Handles First Four pairs by treating them as a pre-tournament round.
"""

import io
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# ── KenPom column names → round numbers ──────────────────────────────────────
# KenPom stores cumulative reach probability out of 1,000,000
KENPOM_COLS   = ["Rd2", "Swt16", "Elite8", "Final4", "Final", "Champ"]
KENPOM_ROUNDS = [1,      2,       3,        4,        5,       6]

# ── Team name normalisation: KenPom display name → canonical name ─────────────
KENPOM_NAME_MAP = {
    "Iowa St.":          "Iowa State",
    "Connecticut":       "UConn",
    "Michigan St.":      "Michigan State",
    "St. John's":        "St. John's",
    "Miami FL":          "Miami (FL)",
    "Miami OH":          "Miami (OH)",
    "N.C. State":        "NC State",
    "Saint Mary's":      "St. Mary's",
    "Saint Louis":       "Saint Louis",
    "South Florida":     "South Florida",
    "Cal Baptist":       "California Baptist",
    "North Dakota St.":  "North Dakota State",
    "Kennesaw St.":      "Kennesaw State",
    "Tennessee St.":     "Tennessee State",
    "Texas A&M":         "Texas A&M",
}

# ── Yahoo editorialTeamKey → canonical name ───────────────────────────────────
YAHOO_KEY_TO_TEAM = {
    "ncaab.t.173": "Duke",           "ncaab.t.17":  "Arizona",
    "ncaab.t.357": "Michigan",       "ncaab.t.210": "Florida",
    "ncaab.t.254": "Houston",        "ncaab.t.129": "UConn",
    "ncaab.t.277": "Iowa State",     "ncaab.t.474": "Purdue",
    "ncaab.t.358": "Michigan State", "ncaab.t.267": "Illinois",
    "ncaab.t.233": "Gonzaga",        "ncaab.t.618": "Virginia",
    "ncaab.t.400": "Nebraska",       "ncaab.t.6":   "Alabama",
    "ncaab.t.287": "Kansas",         "ncaab.t.19":  "Arkansas",
    "ncaab.t.615": "Vanderbilt",     "ncaab.t.534": "St. John's",
    "ncaab.t.592": "Texas Tech",     "ncaab.t.657": "Wisconsin",
    "ncaab.t.68":  "BYU",           "ncaab.t.606": "UCLA",
    "ncaab.t.443": "Ohio St.",       "ncaab.t.367": "Missouri",
    "ncaab.t.230": "Georgia",        "ncaab.t.314": "Louisville",
    "ncaab.t.355": "Miami (FL)",     "ncaab.t.613": "VCU",
    "ncaab.t.580": "Tennessee",      "ncaab.t.276": "Iowa",
    "ncaab.t.617": "Villanova",      "ncaab.t.292": "Kentucky",
    "ncaab.t.413": "North Carolina", "ncaab.t.345": "McNeese",
    "ncaab.t.418": "Northern Iowa",  "ncaab.t.611": "Utah State",
    "ncaab.t.3":   "Akron",         "ncaab.t.120": "Clemson",
    "ncaab.t.515": "SMU",           "ncaab.t.411": "NC State",
    "ncaab.t.585": "Texas",         "ncaab.t.103": "UCF",
    "ncaab.t.576": "TCU",           "ncaab.t.526": "South Florida",
    "ncaab.t.536": "Saint Louis",   "ncaab.t.538": "St. Mary's",
    "ncaab.t.587": "Texas A&M",     "ncaab.t.502": "Santa Clara",
    "ncaab.t.252": "Hofstra",       "ncaab.t.1009":"California Baptist",
    "ncaab.t.246": "Hawaii",        "ncaab.t.596": "Troy",
    "ncaab.t.356": "Miami (OH)",    "ncaab.t.220": "Furman",
    "ncaab.t.460": "Pennsylvania",  "ncaab.t.263": "Idaho",
    "ncaab.t.303": "Lehigh",        "ncaab.t.1374":"Kennesaw State",
    "ncaab.t.1455":"North Dakota State", "ncaab.t.582":"Tennessee State",
    "ncaab.t.346": "UMBC",          "ncaab.t.512": "Siena",
    "ncaab.t.660": "Wright State",  "ncaab.t.1112":"High Point",
    "ncaab.t.1237":"Queens",
}

# ── Betting futures fallback (American odds, ~March 17 2026) ──────────────────
BETTING_ODDS = {
    "Duke": 300, "Michigan": 337, "Arizona": 412, "Florida": 650,
    "Houston": 800, "UConn": 1000, "Iowa State": 1200, "Purdue": 1500,
    "Illinois": 1900, "Gonzaga": 2000, "Virginia": 2500,
    "Michigan State": 2500, "Alabama": 3000, "Nebraska": 3500,
    "Arkansas": 4000, "Kansas": 4000, "St. John's": 4500,
    "Vanderbilt": 5000, "Wisconsin": 5000, "Texas Tech": 5000,
}


# ─────────────────────────────────────────────────────────────────────────────
# KenPom
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kenpom() -> pd.DataFrame:
    """
    Fetch 2026 KenPom tourney predictions from GitHub.
    Returns DataFrame with columns:
        team, seed, region, first_four (bool),
        Rd2, Swt16, Elite8, Final4, Final, Champ  (all as 0-1 probabilities)
    """
    url = "https://raw.githubusercontent.com/kenpom/tourney_predictions/main/ncaa_2026.csv"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.dropna(subset=["Champ"])

    # Normalise names
    df["Team"] = df["Team"].apply(lambda t: KENPOM_NAME_MAP.get(str(t).strip(), str(t).strip()))
    df = df.rename(columns={"Team": "team", "Region": "region", "seed": "seed"})

    # Expand single-letter region codes to full names
    region_expand = {"E": "East", "W": "West", "MW": "Midwest", "S": "South"}
    df["region"] = df["region"].apply(lambda r: region_expand.get(str(r).strip(), str(r).strip()))

    # Convert to 0-1 probabilities
    for col in KENPOM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 1_000_000

    # Detect First Four pairs: same (seed, region) appearing twice
    df["first_four"] = df.duplicated(subset=["seed", "region"], keep=False)

    # Compute per-round conditional win probabilities
    # cond_r = P(win game in round r | reached round r)
    prev = pd.Series([1.0] * len(df), index=df.index)
    for col, rnd in zip(KENPOM_COLS, KENPOM_ROUNDS):
        reach = df[col]
        # For First Four teams: Round 1 conditional = their Rd2
        # (Rd2 already accounts for winning the First Four AND R64 games)
        df[f"cond_{rnd}"] = (reach / prev.clip(lower=1e-9)).clip(0, 1)
        prev = reach

    return df.reset_index(drop=True)


def resolve_first_four(kenpom_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    For each First Four pair, simulate the play-in game and keep one winner.
    Winner chosen probabilistically: P(A) = Rd2_A / (Rd2_A + Rd2_B).
    Returns a 64-team DataFrame (one team per seed-slot per region).
    """
    df = kenpom_df.copy()
    ff_pairs = df[df["first_four"]].groupby(["seed", "region"])

    drop_idx = []
    for (seed, region), group in ff_pairs:
        if len(group) != 2:
            continue
        idx_a, idx_b = group.index[0], group.index[1]
        rd2_a = df.loc[idx_a, "Rd2"]
        rd2_b = df.loc[idx_b, "Rd2"]
        total = rd2_a + rd2_b
        p_a = rd2_a / total if total > 0 else 0.5
        loser_idx = idx_b if rng.random() < p_a else idx_a
        drop_idx.append(loser_idx)

    df = df.drop(index=drop_idx).reset_index(drop=True)
    df["first_four"] = False
    return df


def build_teams_by_region(df_64: pd.DataFrame) -> dict:
    """
    Organise 64 teams into {region: [team_name * 16]}, sorted seed 1→16.
    Index position = seed - 1.
    """
    teams_by_region = {}
    for region, grp in df_64.groupby("region"):
        grp_sorted = grp.sort_values("seed")
        teams_by_region[region] = grp_sorted["team"].tolist()
    return teams_by_region


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_rounds() -> dict:
    """
    Scrape Yahoo Fantasy pick distribution (all 6 rounds).
    Returns {round_str: {team_name: fraction}}
    Round 1 = R64 advancement, Round 6 = Champion.
    """
    url = "https://tournament.fantasysports.yahoo.com/mens-basketball-bracket/pickdistribution"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    big = None
    for s in soup.find_all("script"):
        content = s.string or ""
        if len(content) > 100_000 and "pickDistribution" in content:
            big = content
            break

    if not big:
        raise RuntimeError("Yahoo App script not found — page may have changed structure")

    round_blocks = re.findall(
        r'\{"roundId":"(\d+)","distributionByTeam":\[([^\]]+)\]', big
    )

    results = {}
    for round_id, teams_str in round_blocks:
        entries = re.findall(
            r'"editorialTeamKey":"(ncaab\.t\.\d+)","percentage":([\d.]+)', teams_str
        )
        raw = {}
        for key, pct in entries:
            name = YAHOO_KEY_TO_TEAM.get(key)
            if name:
                raw[name] = float(pct) / 100.0
        # Normalise within round so values sum to 1
        total = sum(raw.values())
        results[round_id] = {t: v / total for t, v in raw.items()} if total > 0 else raw

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Betting odds helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_betting_probs() -> dict:
    """Return vig-free championship probabilities from hardcoded futures odds."""
    def american_to_raw(odds):
        return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
    raw = {t: american_to_raw(o) for t, o in BETTING_ODDS.items()}
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all(rng: np.random.Generator, verbose: bool = True) -> dict:
    """
    Fetch all data and return a dict ready for the simulator:
      kenpom_df     : full 68-team DataFrame (raw, includes First Four)
      yahoo_rounds  : {round_str: {team: fraction}}
      betting_probs : {team: vig-free champ prob}
    """
    if verbose:
        print("  [1/3] Fetching KenPom predictions...")
    kenpom_df = fetch_kenpom()
    if verbose:
        n_ff = kenpom_df["first_four"].sum()
        print(f"        {len(kenpom_df)} teams loaded, {n_ff} in First Four pairs")

    if verbose:
        print("  [2/3] Fetching Yahoo pick distribution...")
    yahoo_rounds = fetch_yahoo_rounds()
    n_champ = len(yahoo_rounds.get("6", {}))
    if verbose:
        print(f"        {len(yahoo_rounds)} rounds, {n_champ} teams with champion data")

    if verbose:
        print("  [3/3] Computing betting-derived probabilities...")
    betting_probs = get_betting_probs()
    if verbose:
        print(f"        {len(betting_probs)} teams with odds")

    return {
        "kenpom_df":    kenpom_df,
        "yahoo_rounds": yahoo_rounds,
        "betting_probs": betting_probs,
    }
