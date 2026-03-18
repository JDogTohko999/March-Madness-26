"""
v2/pool_sim.py — Pool win probability simulation engine.

For each candidate champion × pool size:
  1. Build a coherent "my bracket" (champion path forced, rest value-optimal)
  2. Simulate N true tournament outcomes using KenPom-calibrated probabilities
  3. For each outcome, generate (pool_size - 1) coherent opponent brackets
     drawn from Yahoo pick distributions (with optional Virginia multiplier)
  4. Score all brackets across all 6 rounds
  5. Pool win probability = fraction of sims where my score beats all opponents
"""

import numpy as np
import pandas as pd
from data import resolve_first_four, build_teams_by_region
from bracket import (
    build_cond_probs, simulate_tournament, build_my_bracket,
    build_opponent_bracket, score_bracket, REGIONS
)


# ─────────────────────────────────────────────────────────────────────────────
# Single champion × single pool size
# ─────────────────────────────────────────────────────────────────────────────

def sim_pool_win_prob(
    champion: str,
    kenpom_df: pd.DataFrame,
    yahoo_rounds: dict,
    pool_size: int,
    n_sims: int = 5000,
    virginia_mult: float = 1.0,
    seed: int = 42,
) -> float:
    """
    Estimate P(win pool of `pool_size`) when picking `champion` as your champion.

    For each simulation:
      - Resolve First Four games (random)
      - Simulate true tournament using KenPom conditional probabilities
      - Score "my bracket" against true results
      - Generate (pool_size - 1) opponent brackets from Yahoo distributions
      - Score each opponent bracket
      - Win if my score strictly exceeds all opponents

    Returns expected win probability.
    """
    rng = np.random.default_rng(seed)
    wins = 0

    # Pre-build my bracket once per champion (uses expected First Four resolution)
    # We'll rebuild each sim to account for First Four randomness
    for _ in range(n_sims):
        # Resolve First Four for this sim
        df_64 = resolve_first_four(kenpom_df, rng)
        teams_by_region = build_teams_by_region(df_64)
        cond_probs = build_cond_probs(df_64)

        # Skip sim if champion didn't survive First Four
        champ_present = any(champion in tlist for tlist in teams_by_region.values())
        if not champ_present:
            continue

        # Build my bracket for this sim's bracket structure
        my_picks = build_my_bracket(champion, teams_by_region, cond_probs, yahoo_rounds)

        # Simulate the true tournament
        true_results = simulate_tournament(teams_by_region, cond_probs, rng)

        # Score my bracket (all 6 rounds)
        my_score = score_bracket(my_picks, true_results)

        # Generate and score opponent brackets
        my_win = True
        for _ in range(pool_size - 1):
            opp_picks = build_opponent_bracket(
                teams_by_region, yahoo_rounds, cond_probs, rng, virginia_mult
            )
            opp_score = score_bracket(opp_picks, true_results)
            if opp_score >= my_score:   # tie goes to opponent (conservative)
                my_win = False
                break

        if my_win:
            wins += 1

    return wins / n_sims


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis: all candidates × all pool sizes
# ─────────────────────────────────────────────────────────────────────────────

def run_all_pools(
    kenpom_df: pd.DataFrame,
    yahoo_rounds: dict,
    pool_sizes: list,
    n_sims: int = 5000,
    top_n: int = 15,
    virginia_mult: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Run pool simulations for the top `top_n` teams by KenPom Champ probability
    across all pool sizes.

    Returns {pool_size: {team: win_prob}}.
    Prints a progress bar to stdout.
    """
    # Select candidates: top N by KenPom Champ, excluding First-Four-only teams
    # Use the non-FF-resolved df to get all teams
    candidates_df = (
        kenpom_df[~kenpom_df["first_four"]]
        .sort_values("Champ", ascending=False)
        .head(top_n)
    )
    # Also include the higher-prob First Four team from each pair (they can win)
    ff_winners = (
        kenpom_df[kenpom_df["first_four"]]
        .sort_values("Rd2", ascending=False)
        .drop_duplicates(subset=["seed", "region"])
    )
    candidates_df = pd.concat([candidates_df, ff_winners]).drop_duplicates("team")
    candidates = candidates_df["team"].tolist()

    total_combos = len(pool_sizes) * len(candidates)
    done = 0
    results = {ps: {} for ps in pool_sizes}

    print(f"\n  Simulating {len(candidates)} candidates × "
          f"{len(pool_sizes)} pool sizes × {n_sims} sims each")
    print(f"  Virginia multiplier: {virginia_mult:.1f}x\n")

    for pool_size in pool_sizes:
        for team in candidates:
            win_p = sim_pool_win_prob(
                champion=team,
                kenpom_df=kenpom_df,
                yahoo_rounds=yahoo_rounds,
                pool_size=pool_size,
                n_sims=n_sims,
                virginia_mult=virginia_mult,
                seed=seed + done,
            )
            results[pool_size][team] = win_p
            done += 1

            # Progress bar
            frac = done / total_combos
            bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
            print(f"  [{bar}] {frac*100:5.1f}%  pool={pool_size:>4}  "
                  f"{team:<22}  win={win_p:.3%}", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Value metrics table (champion pick analysis)
# ─────────────────────────────────────────────────────────────────────────────

def build_value_table(
    kenpom_df: pd.DataFrame,
    yahoo_rounds: dict,
    betting_probs: dict,
) -> pd.DataFrame:
    """
    Build a team-level value metrics DataFrame for champion picks.
    Columns: team, seed, region, kenpom_champ, betting_prob, public_pct,
             actual_prob (blend), value_ratio
    """
    yahoo_champ = yahoo_rounds.get("6", {})
    rows = []

    for _, row in kenpom_df.iterrows():
        if row["first_four"]:
            continue  # skip duplicates
        team = row["team"]
        kp   = float(row["Champ"])
        bet  = betting_probs.get(team, 0.0)
        pub  = yahoo_champ.get(team, 0.001)

        # Actual prob: blend KenPom (independent) + betting (independent market)
        # Weight KenPom more since it's a distinct model, not circular
        actual = 0.6 * kp + 0.4 * bet
        value  = actual / pub if pub > 0 else 0.0

        rows.append({
            "team":         team,
            "seed":         int(row["seed"]),
            "region":       row["region"],
            "kenpom_champ": kp,
            "betting_prob": bet,
            "actual_prob":  actual,
            "public_pct":   pub,
            "value_ratio":  value,
        })

    df = pd.DataFrame(rows).sort_values("actual_prob", ascending=False)
    return df.reset_index(drop=True)


def build_round_value_table(
    kenpom_df: pd.DataFrame,
    yahoo_rounds: dict,
) -> pd.DataFrame:
    """
    Build value metrics for all rounds (Elite 8 onward), seeds 1-4 only.
    """
    kenpom_round_map = {3: "Elite8", 4: "Final4", 5: "Final", 6: "Champ"}
    rows = []

    for _, row in kenpom_df.iterrows():
        if row["first_four"] or int(row["seed"]) > 4:
            continue
        team = row["team"]
        for rnum, kp_col in kenpom_round_map.items():
            kp  = float(row[kp_col])
            yh  = yahoo_rounds.get(str(rnum), {}).get(team, None)
            vr  = (kp / yh) if (yh and yh > 0 and kp > 0) else None
            rows.append({
                "team":       team,
                "seed":       int(row["seed"]),
                "region":     row["region"],
                "round":      rnum,
                "round_name": {3:"Elite 8", 4:"Final Four",
                               5:"Champ Game", 6:"Champion"}[rnum],
                "kenpom_prob": kp,
                "yahoo_pct":   yh,
                "value_ratio": vr,
            })

    return pd.DataFrame(rows)


def print_summary_tables(results: dict, value_df: pd.DataFrame, pool_sizes: list):
    """Print ranked champion pick tables for each pool size."""
    POOL_LABELS = {10:"Small (~10)", 20:"Small-Med (~20)",
                   50:"Medium (~50)", 250:"Large (~250)"}

    print("\n" + "="*72)
    print("  CHAMPION PICK RECOMMENDATIONS BY POOL SIZE")
    print("="*72)

    for ps in pool_sizes:
        ps_results = results.get(ps, {})
        baseline   = 1 / ps

        ranked = sorted(ps_results.items(), key=lambda x: -x[1])
        print(f"\n  Pool size {ps:>3}  ({POOL_LABELS.get(ps,'')})   "
              f"baseline = {baseline:.1%}")
        print(f"  {'Rank':<5} {'Team':<22} {'Seed':>4}  "
              f"{'KenPom':>8} {'Public':>8} {'Value':>7} "
              f"{'Pool Win':>9} {'vs Baseline':>11}")
        print("  " + "-"*78)

        for rank, (team, win_p) in enumerate(ranked[:8], 1):
            row = value_df[value_df["team"] == team]
            if row.empty:
                continue
            row = row.iloc[0]
            print(f"  {rank:<5} {team:<22} {int(row['seed']):>4}  "
                  f"{row['kenpom_champ']:>8.2%} {row['public_pct']:>8.2%} "
                  f"{row['value_ratio']:>7.2f} "
                  f"{win_p:>9.3%} {win_p/baseline:>10.1f}x")

    print()
