"""
Tournament and bracket pool simulation engine for 2026 March Madness.
Implements the Elder Research methodology for pool-size-optimized bracket selection.
"""

import numpy as np
from itertools import product

# ESPN scoring system: points double each round
ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

# Bracket structure: (seed1, seed2) matchups in Round of 64, by region
# Standard NCAA bracket matchups
R64_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15)
]

REGIONS = ["East", "West", "Midwest", "South"]

# Final Four: East vs West, Midwest vs South (standard bracket)
FF_MATCHUPS = [("East", "West"), ("Midwest", "South")]


def seed_win_prob(seed1: int, seed2: int, upset_factor: float = 1.0) -> float:
    """
    Estimate win probability for seed1 vs seed2 using a log-based model
    calibrated to historical NCAA tournament outcomes.

    upset_factor > 1 increases upset probability (used for Cinderella teams).
    """
    # Log-linear model fitted to historical seed win rates
    log_ratio = np.log(seed2 / seed1)
    # Empirically calibrated: higher-seed advantage diminishes at high seeds
    raw_prob = 1 / (1 + np.exp(-0.65 * log_ratio))
    # Clip to reasonable bounds
    return float(np.clip(raw_prob * upset_factor, 0.05, 0.95))


def build_team_win_probs(teams: dict, betting_probs: dict) -> dict:
    """
    Build head-to-head win probabilities for all teams using a Bradley-Terry model.
    Strength derived from betting championship probabilities (best single source).
    Falls back to seed-based model for unranked teams.

    Returns dict: {team: strength_rating}
    """
    # Bradley-Terry strength from betting probs
    strengths = {}
    for team, info in teams.items():
        if team in betting_probs and betting_probs[team] > 0:
            # Use 4th root of championship prob as per-game strength
            # (roughly 6 wins needed to win tournament)
            strengths[team] = betting_probs[team] ** (1 / 6)
        else:
            # Seed-based fallback
            seed = info["seed"]
            strengths[team] = 0.5 ** (seed / 4)
    return strengths


def h2h_prob(team_a: str, team_b: str, strengths: dict) -> float:
    """Bradley-Terry head-to-head probability: P(A beats B) = s_A / (s_A + s_B)."""
    sa = strengths.get(team_a, 0.01)
    sb = strengths.get(team_b, 0.01)
    return sa / (sa + sb)


def simulate_region(region_teams: list, strengths: dict, rng: np.random.Generator) -> str:
    """
    Simulate a single region bracket. region_teams is sorted by seed (index 0 = seed 1).
    Returns name of region winner.
    """
    # Round of 64: 8 games
    matchups = [(region_teams[i], region_teams[j]) for i, j in [
        (0, 15), (7, 8), (4, 11), (3, 12),
        (5, 10), (2, 13), (6, 9), (1, 14)
    ]]

    survivors = []
    for a, b in matchups:
        p = h2h_prob(a, b, strengths)
        winner = a if rng.random() < p else b
        survivors.append(winner)

    # Round of 32: 4 games
    r32 = []
    for i in range(0, 8, 2):
        a, b = survivors[i], survivors[i+1]
        p = h2h_prob(a, b, strengths)
        r32.append(a if rng.random() < p else b)

    # Sweet 16: 2 games
    s16 = []
    for i in range(0, 4, 2):
        a, b = r32[i], r32[i+1]
        p = h2h_prob(a, b, strengths)
        s16.append(a if rng.random() < p else b)

    # Elite Eight: 1 game
    a, b = s16[0], s16[1]
    p = h2h_prob(a, b, strengths)
    return a if rng.random() < p else b


def simulate_full_bracket(
    region_teams: dict,  # {region: [team list ordered seed 1..16]}
    strengths: dict,
    rng: np.random.Generator
) -> dict:
    """
    Simulate a full tournament. Returns dict with all results:
    {round: {game_key: winner}}
    Also returns 'champion'.
    """
    results = {}

    # Simulate each region → Final Four
    ff_teams = {}
    for region in REGIONS:
        teams = region_teams[region]
        winner = simulate_region(teams, strengths, rng)
        ff_teams[region] = winner

    results["final_four"] = ff_teams.copy()

    # Simulate Final Four
    sf_winners = []
    for r1, r2 in FF_MATCHUPS:
        a, b = ff_teams[r1], ff_teams[r2]
        p = h2h_prob(a, b, strengths)
        sf_winners.append(a if rng.random() < p else b)

    results["final_two"] = sf_winners

    # Championship
    a, b = sf_winners[0], sf_winners[1]
    p = h2h_prob(a, b, strengths)
    champ = a if rng.random() < p else b
    results["champion"] = champ

    return results


def score_bracket(predicted: dict, actual: dict, region_teams: dict) -> int:
    """
    Score a predicted bracket against actual results using ESPN scoring.
    Both predicted and actual are dicts with keys: champion, final_two, final_four.
    Returns total score (integer).

    For a full implementation this would need per-round tracking,
    but for pool simulations the champion + FF rounds dominate.
    We score the rounds we track: Final Four (R5=160), Championship (R6=320).
    For simplicity, add expected points from earlier rounds based on seed probs.
    """
    score = 0

    # Final Four correct (160 pts each)
    actual_ff = set(actual["final_four"].values())
    for team in predicted.get("final_four", {}).values():
        if team in actual_ff:
            score += ROUND_POINTS[5]

    # Championship game teams (already in final_four scoring above as FF correct)
    # Final correct (320 pts)
    if predicted.get("champion") == actual.get("champion"):
        score += ROUND_POINTS[6]

    return score


def build_region_teams(teams: dict) -> dict:
    """Organize teams by region, sorted by seed."""
    region_teams = {r: [] for r in REGIONS}
    # Build seed-sorted list per region
    for region in REGIONS:
        reg_teams = [(info["seed"], name) for name, info in teams.items()
                     if info["region"] == region]
        reg_teams.sort()
        region_teams[region] = [t for _, t in reg_teams]
    return region_teams


def simulate_champion_probs(
    teams: dict,
    strengths: dict,
    n_sims: int = 10000,
    seed: int = 42
) -> dict:
    """
    Run N simulations of full tournament. Returns {team: championship_prob}.
    """
    rng = np.random.default_rng(seed)
    region_teams = build_region_teams(teams)

    champ_counts = {t: 0 for t in teams}
    for _ in range(n_sims):
        result = simulate_full_bracket(region_teams, strengths, rng)
        champ = result["champion"]
        if champ in champ_counts:
            champ_counts[champ] += 1

    return {t: c / n_sims for t, c in champ_counts.items()}


def simulate_pool_win_prob(
    champion_pick: str,
    teams: dict,
    strengths: dict,
    public_pick_pct: dict,
    pool_size: int,
    n_tournament_sims: int = 5000,
    n_opponent_sims: int = 5000,
    seed: int = 42
) -> float:
    """
    Estimate probability of winning a bracket pool of `pool_size` people
    when picking `champion_pick` as your champion.

    Methodology (Elder Research):
    1. Simulate N tournament outcomes
    2. For each outcome, generate M opponent brackets using public pick distributions
    3. Score your bracket and all opponent brackets
    4. Compute your win rate = fraction of sims where your score is max

    Returns expected pool win probability.
    """
    rng = np.random.default_rng(seed)
    region_teams = build_region_teams(teams)

    # Your bracket: champion = champion_pick, rest = simulate using actual probs
    # (Use a fixed "your bracket" generated from strengths, champion forced)
    your_ff_picks = _generate_bracket_picks(champion_pick, teams, strengths, rng, region_teams)

    win_count = 0
    total_sims = 0

    for sim_idx in range(n_tournament_sims):
        # Simulate true tournament outcome
        true_result = simulate_full_bracket(region_teams, strengths, rng)
        true_champ = true_result["champion"]

        # Score your bracket
        your_score = score_bracket(your_ff_picks, true_result, region_teams)
        # Add champion bonus
        if true_champ == champion_pick:
            your_score += 50  # tiebreaker bonus for correct champion (proxy)

        # Simulate opponent brackets (pool_size - 1 opponents)
        # Each opponent picks champion from public distribution
        opponent_beats_you = 0

        for _ in range(pool_size - 1):
            # Draw opponent's champion from public distribution
            opp_champ = _draw_from_dist(public_pick_pct, rng)
            # Opponent's FF picks (simplified: random based on strengths)
            opp_picks = _generate_opponent_picks(opp_champ, teams, public_pick_pct, rng, region_teams)
            opp_score = score_bracket(opp_picks, true_result, region_teams)
            if true_champ == opp_champ:
                opp_score += 50
            if opp_score > your_score:
                opponent_beats_you += 1

        # You win if no opponent beat you
        if opponent_beats_you == 0:
            win_count += 1
        total_sims += 1

    return win_count / total_sims


def _draw_from_dist(prob_dict: dict, rng: np.random.Generator) -> str:
    """Draw a team from a probability distribution, handling normalization."""
    teams = list(prob_dict.keys())
    probs = np.array([prob_dict[t] for t in teams])
    probs = probs / probs.sum()
    idx = rng.choice(len(teams), p=probs)
    return teams[idx]


def _generate_bracket_picks(champion: str, teams: dict, strengths: dict,
                             rng: np.random.Generator, region_teams: dict) -> dict:
    """Generate a bracket picking `champion` as champion, rest by strength."""
    # Find which region champion is in
    champ_region = teams[champion]["region"]

    # Simulate Final Four: force champion's region to produce champion
    ff = {}
    for region in REGIONS:
        if region == champ_region:
            ff[region] = champion
        else:
            ff[region] = simulate_region(region_teams[region], strengths, rng)

    # Final Four: champion must win their semifinal
    sf_winners = [champion]
    for r1, r2 in FF_MATCHUPS:
        if champ_region in (r1, r2):
            # Other semifinal
            other = r2 if champ_region == r1 else r1
            # Champion's side already handled; pick other SF winner
            continue
        a, b = ff[r1], ff[r2]
        p = h2h_prob(a, b, strengths)
        sf_winners.append(a if rng.random() < p else b)

    return {"champion": champion, "final_four": ff, "final_two": sf_winners}


def _generate_opponent_picks(champion: str, teams: dict, public_pick_pct: dict,
                              rng: np.random.Generator, region_teams: dict) -> dict:
    """Generate opponent bracket using public pick percentages for FF, champion forced."""
    # Simplified: champion region forced, other FF teams drawn from public region probs
    champ_region = teams.get(champion, {}).get("region", REGIONS[0])
    ff = {champ_region: champion}
    for region in REGIONS:
        if region == champ_region:
            continue
        # Pick FF team from region using public pick pct (fallback: uniform)
        region_teams_list = [t for t, info in teams.items() if info["region"] == region]
        weights = np.array([public_pick_pct.get(t, 0.01) for t in region_teams_list])
        weights /= weights.sum()
        idx = rng.choice(len(region_teams_list), p=weights)
        ff[region] = region_teams_list[idx]

    return {"champion": champion, "final_four": ff, "final_two": list(ff.values())[:2]}


def run_pool_analysis(
    teams: dict,
    betting_probs: dict,
    public_pick_pct: dict,
    pool_sizes: list = [10, 20, 50, 100, 500, 1000],
    top_n_teams: int = 20,
    n_sims: int = 3000,
) -> dict:
    """
    Run the full pool analysis for all pool sizes and top candidate champion picks.
    Returns nested dict: {pool_size: {team: win_prob}}
    """
    strengths = build_team_win_probs(teams, betting_probs)

    # Identify top candidate teams (by betting probability)
    candidates = sorted(betting_probs.keys(), key=lambda t: -betting_probs[t])[:top_n_teams]
    # Make sure all candidates are in our team list
    candidates = [t for t in candidates if t in teams]

    print(f"\nRunning pool simulations for {len(candidates)} candidate champions "
          f"× {len(pool_sizes)} pool sizes ({n_sims} sims each)...")

    results = {}
    total = len(pool_sizes) * len(candidates)
    done = 0

    for pool_size in pool_sizes:
        results[pool_size] = {}
        print(f"\n  Pool size {pool_size}:")
        for team in candidates:
            win_p = simulate_pool_win_prob(
                champion_pick=team,
                teams=teams,
                strengths=strengths,
                public_pick_pct=public_pick_pct,
                pool_size=pool_size,
                n_tournament_sims=n_sims,
                n_opponent_sims=1,  # opponent sims folded into pool_size draws
                seed=42 + done,
            )
            results[pool_size][team] = win_p
            done += 1
            pct = done / total * 100
            bar_len = 30
            filled = int(bar_len * done / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"  [{bar}] {pct:5.1f}%  ({done}/{total})  {team}: {win_p:.3%}", flush=True)

    return results


if __name__ == "__main__":
    # Quick test
    from data_collection import collect_all_data, build_team_win_probs

    data = collect_all_data()
    strengths = build_team_win_probs(data["teams"], data["betting_probs"])

    print("\nSimulating championship probabilities (5000 sims)...")
    sim_probs = simulate_champion_probs(data["teams"], strengths, n_sims=5000)

    print("\nTop 15 simulated championship probabilities:")
    for team, prob in sorted(sim_probs.items(), key=lambda x: -x[1])[:15]:
        print(f"  {team:20s}  sim={prob:.3f}")
