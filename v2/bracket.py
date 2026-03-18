"""
v2/bracket.py — Bracket structure, game simulation, and scoring.

Key design decisions:
  - Per-game win probability uses KenPom round-specific conditional probabilities
    via Bradley-Terry: P(A beats B in round r) = cond_r_A / (cond_r_A + cond_r_B)
  - All 6 rounds are scored (ESPN standard: 10/20/40/80/160/320)
  - "My bracket" enforces champion's full path; other games use value-optimal picks
  - Opponent brackets generated game-by-game using Yahoo pick distributions,
    maintaining coherence at every step
"""

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
REGIONS = ["East", "West", "Midwest", "South"]
ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

# Within each region, the 8 R64 seed-pair indices (0-indexed into seed-sorted list)
# Index 0 = seed-1 team, index 15 = seed-16 team
R64_PAIRS = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]

# R32: winners of adjacent R64 slots play each other
R32_GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7)]

# S16: winners of adjacent R32 slots play each other
S16_GROUPS = [(0, 1), (2, 3)]

# Final Four matchup: which region pairs face off
FF_PAIRS = [("East", "West"), ("Midwest", "South")]

# Round labels for display
ROUND_NAMES = {1: "R64", 2: "R32", 3: "Sweet 16", 4: "Elite 8",
               5: "Final Four", 6: "Championship"}


# ─────────────────────────────────────────────────────────────────────────────
# Strength / probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_cond_probs(kenpom_df: pd.DataFrame) -> dict:
    """
    Build {team: {round: conditional_win_prob}} from KenPom data.

    cond_r = P(win game in round r | team has reached round r)
    Already computed in data.fetch_kenpom() as cond_1 … cond_6.
    Returns a plain dict for fast lookup during simulation.
    """
    cond = {}
    for _, row in kenpom_df.iterrows():
        team = row["team"]
        cond[team] = {r: float(row.get(f"cond_{r}", 0.5)) for r in range(1, 7)}
    return cond


def h2h_prob(team_a: str, team_b: str, round_num: int,
             cond_probs: dict, floor: float = 0.05) -> float:
    """
    Bradley-Terry head-to-head probability for round `round_num`.
    P(A beats B) = cond_A / (cond_A + cond_B), floored/ceiled at `floor`.
    """
    ca = cond_probs.get(team_a, {}).get(round_num, 0.5)
    cb = cond_probs.get(team_b, {}).get(round_num, 0.5)
    total = ca + cb
    if total < 1e-12:
        return 0.5
    return float(np.clip(ca / total, floor, 1 - floor))


# ─────────────────────────────────────────────────────────────────────────────
# Tournament simulation (true outcomes)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_region(region: str, region_teams: list, cond_probs: dict,
                    rng: np.random.Generator) -> dict:
    """
    Simulate rounds 1-4 for a single region.
    Returns {game_key: winner_name} for all 15 games in the region.
    region_teams: 16-item list ordered by seed (index 0 = seed 1).
    """
    results = {}

    # Round 1 — R64 (8 games)
    r1_winners = []
    for slot, (i, j) in enumerate(R64_PAIRS):
        a, b = region_teams[i], region_teams[j]
        p = h2h_prob(a, b, 1, cond_probs)
        w = a if rng.random() < p else b
        results[f"1_{region}_{slot}"] = w
        r1_winners.append(w)

    # Round 2 — R32 (4 games)
    r2_winners = []
    for slot, (i, j) in enumerate(R32_GROUPS):
        a, b = r1_winners[i], r1_winners[j]
        p = h2h_prob(a, b, 2, cond_probs)
        w = a if rng.random() < p else b
        results[f"2_{region}_{slot}"] = w
        r2_winners.append(w)

    # Round 3 — Sweet 16 (2 games)
    r3_winners = []
    for slot, (i, j) in enumerate(S16_GROUPS):
        a, b = r2_winners[i], r2_winners[j]
        p = h2h_prob(a, b, 3, cond_probs)
        w = a if rng.random() < p else b
        results[f"3_{region}_{slot}"] = w
        r3_winners.append(w)

    # Round 4 — Elite 8 (1 game)
    a, b = r3_winners[0], r3_winners[1]
    p = h2h_prob(a, b, 4, cond_probs)
    w = a if rng.random() < p else b
    results[f"4_{region}"] = w

    return results


def simulate_tournament(teams_by_region: dict, cond_probs: dict,
                        rng: np.random.Generator) -> dict:
    """
    Simulate a full 63-game tournament. Returns {game_key: winner} for all rounds.
    """
    results = {}

    # Rounds 1-4 (within regions)
    region_winners = {}
    for region in REGIONS:
        r = simulate_region(region, teams_by_region[region], cond_probs, rng)
        results.update(r)
        region_winners[region] = results[f"4_{region}"]

    # Round 5 — Final Four (2 games)
    ff_winners = []
    for pair_idx, (r1, r2) in enumerate(FF_PAIRS):
        a, b = region_winners[r1], region_winners[r2]
        p = h2h_prob(a, b, 5, cond_probs)
        w = a if rng.random() < p else b
        results[f"5_{pair_idx}"] = w
        ff_winners.append(w)

    # Round 6 — Championship
    a, b = ff_winners[0], ff_winners[1]
    p = h2h_prob(a, b, 6, cond_probs)
    w = a if rng.random() < p else b
    results["6"] = w

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_bracket(picks: dict, results: dict) -> int:
    """
    Score a bracket against actual results. All 63 game keys checked.
    picks and results are both {game_key: team_name}.
    """
    score = 0
    for key, true_winner in results.items():
        if picks.get(key) == true_winner:
            round_num = int(key.split("_")[0])
            score += ROUND_POINTS[round_num]
    return score


# ─────────────────────────────────────────────────────────────────────────────
# "My bracket" — value-optimal with forced champion path
# ─────────────────────────────────────────────────────────────────────────────

def _chalk_pick(team_a: str, team_b: str, round_num: int, cond_probs: dict) -> str:
    """
    Pick the higher-KenPom-probability team (chalk).
    Used for all non-champion games to maximise expected base score.
    Value theory only applies to the champion selection, not every game.
    """
    ca = cond_probs.get(team_a, {}).get(round_num, 0.5)
    cb = cond_probs.get(team_b, {}).get(round_num, 0.5)
    return team_a if ca >= cb else team_b


def build_my_bracket(champion: str, teams_by_region: dict,
                     cond_probs: dict, yahoo_rounds: dict = None) -> dict:
    """
    Build a coherent bracket with `champion` forced to win every game on their path.
    All other games use value-optimal picks (KenPom / Yahoo ratio).

    Returns {game_key: my_pick} for all 63 games.
    """
    picks = {}
    champ_region = None
    for region, teams in teams_by_region.items():
        if champion in teams:
            champ_region = region
            break
    if champ_region is None:
        raise ValueError(f"Champion '{champion}' not found in any region")

    # ── Rounds 1-4: per region ────────────────────────────────────────────────
    region_picks = {}  # region → E8 winner pick

    for region in REGIONS:
        region_teams = teams_by_region[region]
        forced = (region == champ_region)

        # Round 1
        r1_picks = []
        for slot, (i, j) in enumerate(R64_PAIRS):
            a, b = region_teams[i], region_teams[j]
            if forced and champion in (a, b):
                w = champion
            else:
                w = _chalk_pick(a, b, 1, cond_probs)
            picks[f"1_{region}_{slot}"] = w
            r1_picks.append(w)

        # Round 2
        r2_picks = []
        for slot, (i, j) in enumerate(R32_GROUPS):
            a, b = r1_picks[i], r1_picks[j]
            if forced and champion in (a, b):
                w = champion
            else:
                w = _chalk_pick(a, b, 2, cond_probs)
            picks[f"2_{region}_{slot}"] = w
            r2_picks.append(w)

        # Round 3
        r3_picks = []
        for slot, (i, j) in enumerate(S16_GROUPS):
            a, b = r2_picks[i], r2_picks[j]
            if forced and champion in (a, b):
                w = champion
            else:
                w = _chalk_pick(a, b, 3, cond_probs)
            picks[f"3_{region}_{slot}"] = w
            r3_picks.append(w)

        # Round 4
        a, b = r3_picks[0], r3_picks[1]
        if forced and champion in (a, b):
            w = champion
        else:
            w = _chalk_pick(a, b, 4, cond_probs)
        picks[f"4_{region}"] = w
        region_picks[region] = w

    # ── Round 5: Final Four ───────────────────────────────────────────────────
    ff_picks = []
    for pair_idx, (r1, r2) in enumerate(FF_PAIRS):
        a, b = region_picks[r1], region_picks[r2]
        if champion in (a, b):
            w = champion
        else:
            w = _chalk_pick(a, b, 5, cond_probs)
        picks[f"5_{pair_idx}"] = w
        ff_picks.append(w)

    # ── Round 6: Championship ─────────────────────────────────────────────────
    a, b = ff_picks[0], ff_picks[1]
    w = champion if champion in (a, b) else _chalk_pick(a, b, 6, cond_probs)
    picks["6"] = w

    return picks


# ─────────────────────────────────────────────────────────────────────────────
# Opponent bracket generation — coherent, game-by-game from Yahoo distributions
# ─────────────────────────────────────────────────────────────────────────────

def _yahoo_pick(team_a: str, team_b: str, round_num: int,
                yahoo_rounds: dict, rng: np.random.Generator,
                virginia_mult: float = 1.0) -> str:
    """
    Draw one team from a Yahoo-weighted distribution for a specific game.
    Applies virginia_mult to Virginia's pick weight before normalising.
    """
    r_key = str(round_num)
    yahoo_r = yahoo_rounds.get(r_key, {})

    def weight(team):
        w = yahoo_r.get(team, 0.01)  # small floor so unknown teams aren't excluded
        if team == "Virginia":
            w *= virginia_mult
        return w

    wa, wb = weight(team_a), weight(team_b)
    total = wa + wb
    return team_a if rng.random() < wa / total else team_b


def build_opponent_bracket(teams_by_region: dict, yahoo_rounds: dict,
                            cond_probs: dict, rng: np.random.Generator,
                            virginia_mult: float = 1.0) -> dict:
    """
    Generate one coherent opponent bracket drawn from Yahoo pick distributions.
    At each game, the two candidates are whoever the opponent picked to reach
    that game (from prior rounds), and the draw uses Yahoo round-specific weights.

    virginia_mult > 1 inflates Virginia's pick probability (local pool calibration).
    """
    picks = {}
    region_picks = {}

    # ── Rounds 1-4: per region ────────────────────────────────────────────────
    for region in REGIONS:
        region_teams = teams_by_region[region]

        # Round 1
        r1_picks = []
        for slot, (i, j) in enumerate(R64_PAIRS):
            a, b = region_teams[i], region_teams[j]
            w = _yahoo_pick(a, b, 1, yahoo_rounds, rng, virginia_mult)
            picks[f"1_{region}_{slot}"] = w
            r1_picks.append(w)

        # Round 2
        r2_picks = []
        for slot, (i, j) in enumerate(R32_GROUPS):
            a, b = r1_picks[i], r1_picks[j]
            w = _yahoo_pick(a, b, 2, yahoo_rounds, rng, virginia_mult)
            picks[f"2_{region}_{slot}"] = w
            r2_picks.append(w)

        # Round 3
        r3_picks = []
        for slot, (i, j) in enumerate(S16_GROUPS):
            a, b = r2_picks[i], r2_picks[j]
            w = _yahoo_pick(a, b, 3, yahoo_rounds, rng, virginia_mult)
            picks[f"3_{region}_{slot}"] = w
            r3_picks.append(w)

        # Round 4
        a, b = r3_picks[0], r3_picks[1]
        w = _yahoo_pick(a, b, 4, yahoo_rounds, rng, virginia_mult)
        picks[f"4_{region}"] = w
        region_picks[region] = w

    # ── Round 5: Final Four ───────────────────────────────────────────────────
    ff_picks = []
    for pair_idx, (r1, r2) in enumerate(FF_PAIRS):
        a, b = region_picks[r1], region_picks[r2]
        w = _yahoo_pick(a, b, 5, yahoo_rounds, rng, virginia_mult)
        picks[f"5_{pair_idx}"] = w
        ff_picks.append(w)

    # ── Round 6: Championship ─────────────────────────────────────────────────
    a, b = ff_picks[0], ff_picks[1]
    w = _yahoo_pick(a, b, 6, yahoo_rounds, rng, virginia_mult)
    picks["6"] = w

    return picks
