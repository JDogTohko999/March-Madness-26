"""
Data collection for 2026 March Madness bracket optimization.
Fetches public pick % from Yahoo Fantasy pick distribution,
betting odds from hardcoded futures, and Barttorvik T-Rank ratings.
"""

import requests
from bs4 import BeautifulSoup
import re
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


# ─── HARDCODED BOOTSTRAP DATA ─────────────────────────────────────────────────
# All data confirmed as of March 17-18, 2026

TOURNAMENT_FIELD = [
    # (seed, team, region, record, conf_tourney_result)
    # ── 1-seeds ──
    (1, "Duke",          "East",    "32-2",  "ACC Champ"),
    (1, "Arizona",       "West",    "32-2",  "Pac-12 Champ"),
    (1, "Michigan",      "Midwest", "31-3",  "Big Ten Champ"),
    (1, "Florida",       "South",   "26-7",  "SEC Champ"),
    # ── 2-seeds ──
    (2, "Houston",       "South",   "28-6",  "Big 12 Runner-Up"),
    (2, "UConn",         "East",    "29-5",  "Big East Champ"),
    (2, "Iowa State",    "Midwest", "27-7",  "Big 12 Champ"),
    (2, "Purdue",        "West",    "27-8",  "Big Ten Runner-Up"),
    # ── 3-seeds ──
    (3, "Michigan State","East",    "25-7",  ""),
    (3, "Illinois",      "South",   "24-8",  ""),
    (3, "Gonzaga",       "West",    "30-3",  "WCC Champ"),
    (3, "Virginia",      "Midwest", "29-5",  "ACC Runner-Up"),
    # ── 4-seeds ──
    (4, "Nebraska",      "East",    "26-6",  ""),
    (4, "Alabama",       "West",    "23-9",  "SEC Runner-Up"),
    (4, "Kansas",        "South",   "23-10", ""),
    (4, "Arkansas",      "Midwest", "26-8",  "SEC"),
    # ── 5-seeds ──
    (5, "Vanderbilt",    "West",    "26-8",  ""),
    (5, "St. John's",    "East",    "28-6",  "Big East Runner-Up"),
    (5, "Texas Tech",    "South",   "22-10", ""),
    (5, "Wisconsin",     "Midwest", "24-10", ""),
    # ── 6-seeds ──
    (6, "BYU",           "East",    "23-10", ""),
    (6, "Ole Miss",      "West",    "22-11", ""),
    (6, "Clemson",       "South",   "22-11", ""),
    (6, "Missouri",      "Midwest", "21-12", ""),
    # ── 7-seeds ──
    (7, "Georgia",       "East",    "21-12", ""),
    (7, "UCLA",          "West",    "20-13", ""),
    (7, "Xavier",        "South",   "22-11", ""),
    (7, "Marquette",     "Midwest", "22-11", ""),
    # ── 8-seeds ──
    (8, "Mississippi State","East", "20-13", ""),
    (8, "Boise State",   "West",    "25-8",  ""),
    (8, "Creighton",     "South",   "20-14", ""),
    (8, "Oklahoma",      "Midwest", "19-14", ""),
    # ── 9-seeds ──
    (9, "Oklahoma State","East",    "19-14", ""),
    (9, "Utah State",    "West",    "27-7",  "MWC Champ"),
    (9, "Louisville",    "South",   "21-12", ""),
    (9, "Wake Forest",   "Midwest", "20-13", ""),
    # ── 10-seeds ──
    (10,"Colorado State","East",    "26-8",  "MWC Runner-Up"),
    (10,"New Mexico",    "West",    "25-9",  ""),
    (10,"Arkansas State","South",   "26-7",  "Sun Belt Champ"),
    (10,"Drake",         "Midwest", "27-6",  "MVC Champ"),
    # ── 11-seeds ──
    (11,"VCU",           "East",    "22-12", ""),
    (11,"San Diego State","West",   "21-12", ""),
    (11,"Butler",        "South",   "20-13", ""),
    (11,"North Texas",   "Midwest", "19-14", ""),
    # ── 12-seeds ──
    (12,"UC San Diego",  "East",    "26-7",  "Big West Champ"),
    (12,"Grand Canyon",  "West",    "25-8",  "WAC Champ"),
    (12,"Liberty",       "South",   "27-7",  "CUSA Champ"),
    (12,"Akron",         "Midwest", "26-8",  "MAC Champ"),
    # ── 13-seeds ──
    (13,"Vermont",       "East",    "28-5",  "AE Champ"),
    (13,"High Point",    "West",    "29-5",  "Big South Champ"),
    (13,"Morehead State","South",   "27-7",  "OVC Champ"),
    (13,"Murray State",  "Midwest", "27-6",  "OVC Champ"),
    # ── 14-seeds ──
    (14,"Montana",       "East",    "25-9",  "BSC Champ"),
    (14,"Northern Iowa", "West",    "24-9",  "MVC Champ"),
    (14,"Lipscomb",      "South",   "25-8",  "ASUN Champ"),
    (14,"UTEP",          "Midwest", "24-10", "CUSA Champ"),
    # ── 15-seeds ──
    (15,"Loyola-Maryland","East",   "22-12", "MAAC Champ"),
    (15,"S Dakota State","West",    "28-6",  "Summit Champ"),
    (15,"McNeese State", "South",   "25-8",  "Southland Champ"),
    (15,"SE Missouri St","Midwest", "23-11", "OVC Champ"),
    # ── 16-seeds ──
    (16,"Quinnipiac",    "East",    "20-14", "MAAC Runner-Up"),
    (16,"Gardner-Webb",  "West",    "22-12", "Big South Runner-Up"),
    (16,"Sacred Heart",  "South",   "19-15", "NEC Champ"),
    (16,"SIU Edwardsville","Midwest","18-15","OVC Runner-Up"),
]

# Championship futures (American odds) from BetMGM/DraftKings, ~March 17
FUTURES_ODDS = {
    "Duke":          300,
    "Michigan":      337,   # average of 325 and 350
    "Arizona":       412,   # average of 400 and 425
    "Florida":       650,
    "Houston":       800,
    "UConn":        1000,
    "Iowa State":   1200,
    "Purdue":       1500,
    "Illinois":     1900,
    "Gonzaga":      2000,
    "Virginia":     2500,
    "Michigan State":2500,
    "Alabama":      3000,
    "Nebraska":     3500,
    "Arkansas":     4000,
    "Kansas":       4000,
    "St. John's":   4500,
    "Vanderbilt":   5000,
    "Wisconsin":    5000,
    "Texas Tech":   5000,
    "BYU":          6000,
    "Clemson":      6000,
    "Creighton":    7500,
    "Marquette":    7500,
    "Boise State":  8000,
    "Colorado State":8000,
    "San Diego State":10000,
    "Drake":        10000,
    "Utah State":   10000,
}

# Yahoo team key → our team name mapping
# Derived from Yahoo's editorialTeamKey field in pick distribution data
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
}

# Fallback public pick % if Yahoo scraping fails (champion = round 6)
YAHOO_FALLBACK_CHAMP_PCT = {
    "Duke":          0.284,
    "Arizona":       0.219,
    "Michigan":      0.143,
    "Florida":       0.063,
    "Houston":       0.057,
    "UConn":         0.030,
    "Iowa State":    0.028,
    "Purdue":        0.025,
    "Virginia":      0.020,
    "Gonzaga":       0.015,
    "Michigan State":0.012,
    "Illinois":      0.010,
    "Alabama":       0.010,
    "Nebraska":      0.008,
    "Kansas":        0.007,
    "Arkansas":      0.005,
    "St. John's":    0.005,
    "Texas Tech":    0.004,
    "Wisconsin":     0.003,
}


def american_to_prob(odds: int) -> float:
    """Convert American moneyline odds to raw implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig(probs: dict) -> dict:
    """Normalize implied probabilities to remove bookmaker vig."""
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def get_betting_probs() -> dict:
    """Convert futures odds to vig-free championship probabilities."""
    raw = {team: american_to_prob(odds) for team, odds in FUTURES_ODDS.items()}
    return remove_vig(raw)


def fetch_yahoo_pick_distribution() -> dict | None:
    """
    Scrape Yahoo Fantasy pick distribution page.
    Extracts all 6 rounds of pick percentages from the embedded App JSON.
    Returns dict of {round_id: {team_name: percentage}} or None on failure.

    Round IDs: 1=R64, 2=R32, 3=Sweet16, 4=Elite8, 5=FinalFour, 6=Champion
    """
    url = "https://tournament.fantasysports.yahoo.com/mens-basketball-bracket/pickdistribution"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"  Yahoo pick dist: HTTP {resp.status_code} — using fallback")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        scripts = soup.find_all("script")

        # Find the large App script (~900KB+)
        big = None
        for s in scripts:
            content = s.string or ""
            if len(content) > 100000 and "pickDistribution" in content:
                big = content
                break

        if not big:
            print("  Yahoo pick dist: App script not found — using fallback")
            return None

        # Extract all round blocks
        round_blocks = re.findall(
            r'\{"roundId":"(\d+)","distributionByTeam":\[([^\]]+)\]', big
        )
        if not round_blocks:
            print("  Yahoo pick dist: distributionByTeam not found — using fallback")
            return None

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

        n_champ = len(results.get("6", {}))
        print(f"  Yahoo pick dist: scraped {len(round_blocks)} rounds, "
              f"{n_champ} teams with champion pick data")
        return results

    except Exception as e:
        print(f"  Yahoo pick dist error: {e} — using fallback")
        return None


def fetch_barttorvik() -> dict | None:
    """
    Fetch T-Rank ratings from barttorvik.com.
    Returns dict of {team: {"barthag": float, "adjoe": float, "adjde": float}}
    """
    url = "https://barttorvik.com/trank.php?year=2026&sort=&top=0&conlimit=All&venue=All&type=All&begin=20251101&end=20260418&hteam=&lowerRank=1&upperRank=400&oppType=All#"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"  Barttorvik: HTTP {resp.status_code}")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "trank-table"}) or soup.find("table")
        if not table:
            print("  Barttorvik: table not found")
            return None
        rows = table.find_all("tr")[1:]  # skip header
        ratings = {}
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) < 5:
                continue
            try:
                team = cols[1]
                adjoe = float(cols[3])
                adjde = float(cols[4])
                # Barthag is pythagorean win % against average D1
                barthag = float(cols[2]) if len(cols) > 2 else None
                ratings[team] = {"barthag": barthag, "adjoe": adjoe, "adjde": adjde}
            except (ValueError, IndexError):
                continue
        print(f"  Barttorvik: parsed {len(ratings)} teams")
        return ratings if ratings else None
    except Exception as e:
        print(f"  Barttorvik error: {e}")
        return None


def collect_all_data() -> dict:
    """
    Master data collection function.
    Returns a dict with all data needed for analysis.
    """
    print("Collecting data...")

    # 1. Betting-derived championship probabilities
    print(" [1/3] Computing betting-derived championship probabilities...")
    betting_probs = get_betting_probs()
    print(f"       {len(betting_probs)} teams with odds data")

    # 2. Public pick percentages (Yahoo, round 6 = champion)
    print(" [2/3] Fetching Yahoo public pick percentages...")
    yahoo_data = fetch_yahoo_pick_distribution()
    if yahoo_data and "6" in yahoo_data and yahoo_data["6"]:
        public_pick_pct = yahoo_data["6"]
        yahoo_all_rounds = yahoo_data
    else:
        public_pick_pct = YAHOO_FALLBACK_CHAMP_PCT.copy()
        yahoo_all_rounds = {}
    # Normalize so they sum to 1
    total = sum(public_pick_pct.values())
    public_pick_pct = {k: v / total for k, v in public_pick_pct.items()}

    # 3. Barttorvik ratings
    print(" [3/3] Fetching Barttorvik T-Rank ratings...")
    torvik = fetch_barttorvik()

    # 4. Assemble team records
    teams = {}
    for seed, team, region, record, conf in TOURNAMENT_FIELD:
        teams[team] = {
            "seed": seed,
            "region": region,
            "record": record,
            "conf_result": conf,
        }

    return {
        "teams": teams,
        "betting_probs": betting_probs,
        "public_pick_pct": public_pick_pct,
        "yahoo_all_rounds": yahoo_all_rounds,
        "torvik": torvik or {},
        "futures_odds": FUTURES_ODDS,
    }


if __name__ == "__main__":
    data = collect_all_data()
    print("\nTop 10 by betting probability:")
    bp = data["betting_probs"]
    for team, prob in sorted(bp.items(), key=lambda x: -x[1])[:10]:
        pub = data["public_pick_pct"].get(team, 0)
        print(f"  {team:20s}  betting={prob:.3f}  public={pub:.3f}  ratio={prob/pub:.2f}" if pub else
              f"  {team:20s}  betting={prob:.3f}  public=N/A")
