#!/usr/bin/env python3
"""
LangGraph agent + tool for football prediction data harvesting.

Flow:
  user prompt -> [Agent (Haiku) parses entities] -> [Tool hits API-FOOTBALL] -> answer

What the tool fetches (when a fixture is resolved):
  - fixture info
  - standings (league table), team statistics (home/away)
  - last H2H meetings
  - lineups (if posted)
  - events (historical / after kickoff)
  - **pre-match odds** (if bookmakers have posted them) via /v3/football/odds?fixture={id}

Also supports single-team queries (profile, recent results, upcoming + pre-match odds).

Setup
  pip install langgraph langchain anthropic requests pydantic python-dotenv
  export API_FOOTBALL_KEY="your_api_sports_key"   # header: x-apisports-key
  export ANTHROPIC_API_KEY="your_anthropic_key"

Run
  python football_prediction_langgraph.py "I would like data for Chelsea vs Brighton this weekend"
  python football_prediction_langgraph.py "Panathinaikos data please"
"""

import os
import re
import sys
import json
import time
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple, TypedDict

import requests
from pydantic import BaseModel, Field
import re
from dotenv import load_dotenv

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
# LangGraph checkpointer (in-memory)
try:
    from langgraph.checkpoint.memory import MemorySaver  # newer versions
except Exception:
    try:
        from langgraph.checkpoint import MemorySaver  # older versions
    except Exception:
        MemorySaver = None
from langchain.tools import tool
import math
import urllib.parse
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import warnings

# ------------------ ENV & CONFIG ------------------

load_dotenv()

# Allow a graph-only mode (no API keys required) via CLI flags or env
_argv = sys.argv[1:] if isinstance(sys.argv, list) else []
GRAPH_ONLY = (
    any(a in ("--draw-graph", "--graph", "--diagram") for a in _argv)
    or os.getenv("GRAPH_ONLY", "0").lower() in ("1", "true", "yes")
)

API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
if not API_KEY and not GRAPH_ONLY:
    print("ERROR: API_FOOTBALL_KEY is not set (x-apisports-key).")
    sys.exit(1)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY and not GRAPH_ONLY:
    print("ERROR: ANTHROPIC_API_KEY is not set.")
    sys.exit(1)

BASE = "https://v3.football.api-sports.io"
HDRS = {"x-apisports-key": API_KEY}

# Simple debug toggle (set DEBUG=1 env var to enable verbose prints)
DEBUG = os.getenv("DEBUG", "0") not in ("0", "false", "False", "")

# Suppress noisy warnings unless DEBUG is enabled
SUPPRESS_WARNINGS = os.getenv("SUPPRESS_WARNINGS", "1" if not DEBUG else "0").lower() not in ("0", "false", "")
if SUPPRESS_WARNINGS:
    import warnings as _warnings
    try:
        from sklearn.exceptions import InconsistentVersionWarning as _InconsistentVersionWarning
        _warnings.filterwarnings("ignore", category=_InconsistentVersionWarning)
    except Exception:
        pass
    # Pydantic serializer noise
    try:
        _warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*", category=UserWarning)
    except Exception:
        pass
    # XGBoost empty dataset noise
    try:
        _warnings.filterwarnings("ignore", message=r".*Empty dataset at worker.*", category=UserWarning)
    except Exception:
        pass

def dbg(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ------------------ SMALL UTILS ------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _sanitize_search(s: str) -> str:
    """API-FOOTBALL 'search' fields accept only alphanumeric and spaces. Strip other chars.

    Example: 'Brighton & Hove Albion' -> 'Brighton  Hove Albion' -> 'Brighton Hove Albion'
    """
    s = s or ""
    # Replace non-alnum/space with space, then collapse spaces
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def current_season(today: Optional[dt.date] = None) -> int:
    """Return the football season year expected by API-FOOTBALL.

    API-FOOTBALL uses the starting year of the season (e.g. 2024 for 2024-2025).
    Heuristic: if month >= 7 (July onward) use current calendar year, else use previous year.
    """
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1

def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    dbg(f"HTTP GET {url} params={params}")
    r = requests.get(url, headers=HDRS, params=params, timeout=20)
    if r.status_code == 429:
        raise RuntimeError("Rate limited by API-FOOTBALL (429).")
    if r.status_code >= 400:
        raise RuntimeError(f"API error [{r.status_code}]: {r.text}")
    data = r.json()
    dbg(f"HTTP RESP keys={list(data.keys())} err={data.get('errors')}")
    if data.get("errors"):
        raise RuntimeError(f"API returned errors: {data['errors']}")
    return data

def _date_range_from_hint(date_hint: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Loose parser for 'today', 'tomorrow', 'saturday', 'this weekend', or YYYY-MM-DD."""
    if not date_hint:
        return (None, None)
    hint = date_hint.lower().strip()
    today = dt.date.today()
    dows = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

    if hint == "today":
        return (today.isoformat(), today.isoformat())
    if hint == "tomorrow":
        d = today + dt.timedelta(days=1)
        return (d.isoformat(), d.isoformat())
    if hint == "yesterday":
        d = today - dt.timedelta(days=1)
        return (d.isoformat(), d.isoformat())
    if "weekend" in hint:
        # Weekend window
        if "last" in hint:
            # Previous weekend (Saturday-Sunday) relative to today
            offset = ((5 - today.weekday()) % 7) - 7
        else:
            # Upcoming/current weekend: Saturday-Sunday
            offset = (5 - today.weekday()) % 7
        start = today + dt.timedelta(days=offset)
        end = start + dt.timedelta(days=1)
        return (start.isoformat(), end.isoformat())
    # last <weekday>
    mlast = re.match(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", hint)
    if mlast:
        idx = dows.index(mlast.group(1))
        # Compute the most recent past occurrence strictly before today
        diff = (today.weekday() - idx) % 7
        diff = diff if diff != 0 else 7
        target = today - dt.timedelta(days=diff)
        return (target.isoformat(), target.isoformat())
    if hint in dows:
        target = today + dt.timedelta(days=(dows.index(hint) - today.weekday()) % 7)
        return (target.isoformat(), target.isoformat())
    # Month YYYY (e.g., 'march 2024' or 'in march 2024')
    months = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12,
    }
    mm = re.search(r"(?:^|\b)in\s+([a-zA-Z]+)\s+(\d{4})(?:\b|$)", hint)
    if not mm:
        mm = re.search(r"(?:^|\b)([a-zA-Z]+)\s+(\d{4})(?:\b|$)", hint)
    if mm:
        mon_name = mm.group(1).lower()
        year = int(mm.group(2))
        mon = months.get(mon_name)
        if mon:
            import calendar
            start = dt.date(year, mon, 1)
            last_day = calendar.monthrange(year, mon)[1]
            end = dt.date(year, mon, last_day)
            return (start.isoformat(), end.isoformat())

    m = re.search(r"\d{4}-\d{2}-\d{2}", hint)
    if m:
        d = m.group(0)
        return (d, d)
    return (None, None)

def _is_past_only_window(date_hint: Optional[str]) -> bool:
    """Return True if the provided date_hint resolves to a window strictly before today.

    We consider a window "past-only" when the computed end date (or the single date)
    is strictly earlier than today's date (in local time). If the hint is empty or
    cannot be parsed, return False (not considered past-only).
    """
    if not date_hint:
        return False
    dfrom, dto = _date_range_from_hint(date_hint)
    if not dfrom and not dto:
        return False
    try:
        # pick the end of window; if only one date, both are that date
        end_s = dto or dfrom
        if not end_s:
            return False
        end_d = dt.date.fromisoformat(end_s[:10])
        today = dt.date.today()
        return end_d < today
    except Exception:
        return False

# ------------------ API-FOOTBALL HELPERS ------------------

def search_team_by_name(name: str) -> Optional[Dict[str, Any]]:
    q_primary = _sanitize_search(name)
    data = _get(f"{BASE}/teams", {"search": q_primary})
    items = data.get("response", [])
    # Fallbacks for tricky names (e.g., 'Brighton & Hove Albion' -> 'Brighton')
    if not items:
        simplifieds = []
        if "&" in name:
            simplifieds.append(_sanitize_search(name.split("&")[0]))
        tokens = q_primary.split()
        if tokens:
            simplifieds.append(tokens[0])
        tried = set([q_primary])
        for q in simplifieds:
            if not q or q in tried:
                continue
            tried.add(q)
            dbg(f"Retry team search with simplified query: {q}")
            data2 = _get(f"{BASE}/teams", {"search": q})
            items = data2.get("response", [])
            if items:
                break
    if not items:
        return None
    name_n = _norm(name)
    for t in items:
        if _norm(t["team"]["name"]) == name_n:
            return t
    return items[0]

def get_team_recent_fixtures(team_id: int, last: int = 5) -> List[Dict[str, Any]]:
    return _get(f"{BASE}/fixtures", {"team": team_id, "last": last}).get("response", [])

def get_team_upcoming(team_id: int, nxt: int = 3) -> List[Dict[str, Any]]:
    return _get(f"{BASE}/fixtures", {"team": team_id, "next": nxt}).get("response", [])

def compute_team_form_as_of(team_id: int, league_id: int, season: int, iso_dt: str, last: int = 5) -> Dict[str, Any]:
    """Compute a team's form snapshot as of a given kickoff time.

    Pulls a window of recent fixtures for the team in the same season, filters
    out any fixtures strictly after the given iso datetime, and summarizes the
    results over the last N before that time.

    Returns a dict like:
      {
        "form": "WDLLW",
        "sample": [<the N prior fixtures objects>],
        "counts": {"wins": 2, "draws": 1, "losses": 2}
      }
    """
    try:
        cutoff = dt.datetime.fromisoformat(iso_dt.replace("Z", "+00:00"))
    except Exception:
        cutoff = None
    # Get a larger sample to filter from, since API supports last by team but not by time
    window = _get(f"{BASE}/fixtures", {"team": team_id, "season": season, "league": league_id, "last": max(last * 3, 10)}).get("response", [])
    # Keep fixtures at or before cutoff (if provided)
    filtered = []
    for fx in window:
        try:
            fdt = dt.datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
        except Exception:
            fdt = None
        if cutoff is None or (fdt and fdt <= cutoff):
            filtered.append(fx)
    # Sort by date descending, then take last N prior fixtures
    filtered.sort(key=lambda r: r["fixture"]["date"], reverse=True)
    sample = filtered[:last]
    # Build form string from perspective of team_id
    form_chars = []
    w = d = l = 0
    for fx in sample:
        teams = fx.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        goals = fx.get("goals", {})
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            # skip unfinished
            continue
        if team_id == home.get("id"):
            if hg > ag:
                form_chars.append("W"); w += 1
            elif hg == ag:
                form_chars.append("D"); d += 1
            else:
                form_chars.append("L"); l += 1
        elif team_id == away.get("id"):
            if ag > hg:
                form_chars.append("W"); w += 1
            elif ag == hg:
                form_chars.append("D"); d += 1
            else:
                form_chars.append("L"); l += 1
        else:
            # if the fixture doesn't involve the team, ignore (shouldn't happen)
            continue
    # Form is in reverse chronological order currently; reverse to oldest->newest
    form = "".join(reversed(form_chars))
    return {"form": form, "sample": sample, "counts": {"wins": w, "draws": d, "losses": l}}

def head_to_head(team1_id: int, team2_id: int, last: int = 10) -> List[Dict[str, Any]]:
    return _get(f"{BASE}/fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last}).get("response", [])

def find_fixture_by_teams_date(team1_id: int, team2_id: int, date_from: Optional[str], date_to: Optional[str], season: Optional[int], league_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Search fixtures between two teams within an optional date window.

    API-FOOTBALL does not support 'team2'. To narrow, we fetch fixtures for team1
    (optionally filtered by season/date range) then locally filter opponent == team2.
    """
    params: Dict[str, Any] = {"team": team1_id}
    if season:
        params["season"] = season
    if league_id:
        params["league"] = league_id
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to
    dbg(f"Searching fixtures team={team1_id} vs {team2_id} season={season} league={league_id} window=({date_from},{date_to})")
    resp = _get(f"{BASE}/fixtures", params).get("response", [])
    # Filter for opponent
    filtered = []
    for fx in resp:
        home_id = fx["teams"]["home"]["id"]
        away_id = fx["teams"]["away"]["id"]
        if (home_id == team1_id and away_id == team2_id) or (home_id == team2_id and away_id == team1_id):
            filtered.append(fx)
    if not filtered:
        return None
    filtered.sort(key=lambda r: r["fixture"]["date"])
    return filtered[0]

def find_fixture_for_team_by_hint(team_id: int, date_from: Optional[str], date_to: Optional[str], season: Optional[int], league_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Find a single fixture for a team using an optional date window and season.

    - If date window provided, query within window and return the earliest by date.
    - Else, fetch the next fixture (next=1).
    """
    if date_from or date_to:
        params: Dict[str, Any] = {"team": team_id}
        if season:
            params["season"] = season
        if league_id:
            params["league"] = league_id
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        dbg(f"Searching team fixtures team={team_id} season={season} league={league_id} window=({date_from},{date_to})")
        resp = _get(f"{BASE}/fixtures", params).get("response", [])
        if not resp:
            return None
        resp.sort(key=lambda r: r["fixture"]["date"])
        return resp[0]
    else:
        dbg(f"Fetching next fixture for team={team_id}")
        params: Dict[str, Any] = {"team": team_id, "next": 1}
        if league_id:
            params["league"] = league_id
        nxt = _get(f"{BASE}/fixtures", params).get("response", [])
        return nxt[0] if nxt else None

def search_league_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Search leagues by name and return best match entry."""
    data = _get(f"{BASE}/leagues", {"search": _sanitize_search(name)})
    items = data.get("response", [])
    if not items:
        return None
    name_n = _norm(name)
    # Prefer exact name match; otherwise first
    for it in items:
        if _norm(it.get("league", {}).get("name") or "") == name_n:
            return it
    return items[0]

def get_league_fixtures(league_id: int, season: int, date_from: Optional[str], date_to: Optional[str], nxt: int = 20) -> List[Dict[str, Any]]:
    """Fetch fixtures for a league within a date window if provided, otherwise next N fixtures."""
    params: Dict[str, Any] = {"league": league_id, "season": season}
    if date_from or date_to:
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
    else:
        params["next"] = nxt
    return _get(f"{BASE}/fixtures", params).get("response", [])

def league_season_from_fixture(fx: Dict[str, Any]) -> Tuple[int, int]:
    return fx["league"]["id"], fx["league"]["season"]

def get_standings(league_id: int, season: int) -> Dict[str, Any]:
    return _get(f"{BASE}/standings", {"league": league_id, "season": season}).get("response", [])

def get_team_statistics(league_id: int, season: int, team_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/teams/statistics", {"league": league_id, "season": season, "team": team_id}).get("response", {})

def get_injuries(team_id: int, season: int) -> List[Dict[str, Any]]:
    return _get(f"{BASE}/injuries", {"team": team_id, "season": season}).get("response", [])

def get_lineups(fixture_id: int) -> List[Dict[str, Any]]:
    return _get(f"{BASE}/fixtures/lineups", {"fixture": fixture_id}).get("response", [])

def get_events(fixture_id: int) -> List[Dict[str, Any]]:
    return _get(f"{BASE}/fixtures/events", {"fixture": fixture_id}).get("response", [])

def get_prematch_odds_for_fixture(fixture_id: int) -> List[Dict[str, Any]]:
    """Pre-match odds for a fixture when books have posted them."""
    raw = _get(f"{BASE}/odds", {"fixture": fixture_id}).get("response", [])
    return filter_relevant_odds(raw)

# ------------------ PREDICTION PACKET ------------------

class PredictionPacket(BaseModel):
    mode: str = Field(..., description="'team' or 'match'")
    query: str
    resolved: Dict[str, Any]
    # team-centric
    team_profile: Optional[Dict[str, Any]] = None
    team_recent_fixtures: Optional[List[Dict[str, Any]]] = None
    team_upcoming: Optional[List[Dict[str, Any]]] = None
    team_injuries: Optional[List[Dict[str, Any]]] = None
    # match-centric
    fixture: Optional[Dict[str, Any]] = None
    league_table: Optional[Dict[str, Any]] = None
    home_stats: Optional[Dict[str, Any]] = None
    away_stats: Optional[Dict[str, Any]] = None
    h2h_last: Optional[List[Dict[str, Any]]] = None
    lineups: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Dict[str, Any]]] = None
    odds: Optional[List[Dict[str, Any]]] = None
    # extended match context
    home_recent: Optional[List[Dict[str, Any]]] = None
    away_recent: Optional[List[Dict[str, Any]]] = None
    home_next_after: Optional[List[Dict[str, Any]]] = None
    away_next_after: Optional[List[Dict[str, Any]]] = None
    # historical form snapshot as of fixture kickoff (past or future)
    home_form_at: Optional[Dict[str, Any]] = None
    away_form_at: Optional[Dict[str, Any]] = None
    # league-centric (when user asks only about a league)
    league_info: Optional[Dict[str, Any]] = None
    league_fixtures: Optional[List[Dict[str, Any]]] = None

# ------------------ FEATURE EXPORT (schema-driven) ------------------

def _parse_iso_dt(s: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None

def _parse_round_number(round_str: Optional[str]) -> Optional[int]:
    if not round_str:
        return None
    m = re.search(r"(\d+)$", str(round_str))
    return int(m.group(1)) if m else None

def _extract_rank_from_standings(standings: Any, team_id: int) -> Optional[int]:
    """API-FOOTBALL /standings response -> find rank for team_id.
    Expected shape: [ { 'league': { 'standings': [ [ { 'rank':.., 'team': {'id':..}}, ... ] ] } } ]
    """
    try:
        arr = standings or []
        if not arr:
            return None
        league = arr[0].get('league', {})
        grids = league.get('standings') or []
        # standings can be nested list-of-lists (groups)
        for group in grids:
            for row in group:
                t = (row or {}).get('team', {})
                if t.get('id') == team_id:
                    return row.get('rank')
        return None
    except Exception:
        return None

def _points_for(team_id: int, fx: Dict[str, Any]) -> Optional[int]:
    goals = fx.get('goals', {})
    hg, ag = goals.get('home'), goals.get('away')
    if hg is None or ag is None:
        return None
    home = fx.get('teams', {}).get('home', {}).get('id')
    away = fx.get('teams', {}).get('away', {}).get('id')
    if team_id == home:
        if hg > ag: return 3
        if hg == ag: return 1
        return 0
    if team_id == away:
        if ag > hg: return 3
        if ag == hg: return 1
        return 0
    return None

def _goals_for_against(team_id: int, fx: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    goals = fx.get('goals', {})
    hg, ag = goals.get('home'), goals.get('away')
    if hg is None or ag is None:
        return (None, None)
    home = fx.get('teams', {}).get('home', {}).get('id')
    away = fx.get('teams', {}).get('away', {}).get('id')
    if team_id == home:
        return (hg, ag)
    if team_id == away:
        return (ag, hg)
    return (None, None)

def _avg(x: List[float]) -> Optional[float]:
    arr = [v for v in x if isinstance(v, (int, float))]
    return (sum(arr) / len(arr)) if arr else None

def _rate(numer: int, denom: int) -> Optional[float]:
    return (numer / denom) if denom and denom > 0 else None

def _cards_avg_from_stats(stats: Dict[str, Any], color: str) -> Optional[float]:
    try:
        fx_played = (((stats or {}).get('fixtures') or {}).get('played') or {}).get('total') or 0
        cards = (stats or {}).get('cards', {})
        per_min = (cards or {}).get(color) or {}
        total = 0
        for k, v in per_min.items():
            if isinstance(v, dict) and v.get('total') is not None:
                try:
                    total += int(v.get('total') or 0)
                except Exception:
                    continue
        return _rate(total, fx_played)
    except Exception:
        return None

def _penalty_success_rate(stats: Dict[str, Any]) -> Optional[float]:
    try:
        pen = (stats or {}).get('penalty') or {}
        scored = ((pen.get('scored') or {}).get('total')) or 0
        missed = ((pen.get('missed') or {}).get('total')) or 0
        att = (scored or 0) + (missed or 0)
        return _rate(scored, att)
    except Exception:
        return None

def export_match_features_from_packet(pkt: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (features, meta) for a single match packet dict. Skips non-match packets."""
    if (pkt or {}).get('mode') != 'match' or not pkt.get('fixture'):
        return (None, None)
    fx = pkt['fixture']
    league_obj = fx.get('league', {})
    round_str = league_obj.get('round')
    round_num = _parse_round_number(round_str)
    kickoff = _parse_iso_dt(fx.get('fixture', {}).get('date'))
    home = fx.get('teams', {}).get('home', {})
    away = fx.get('teams', {}).get('away', {})
    home_id, away_id = home.get('id'), away.get('id')
    league_id = league_obj.get('id')
    season = league_obj.get('season')

    # standings ranks
    home_rank = _extract_rank_from_standings(pkt.get('league_table'), home_id)
    away_rank = _extract_rank_from_standings(pkt.get('league_table'), away_id)
    league_rank_diff = (away_rank - home_rank) if (home_rank and away_rank) else None

    # form window samples from historical snapshot if available; else fall back to home_recent/away_recent
    h_sample = ((pkt.get('home_form_at') or {}).get('sample')) or (pkt.get('home_recent') or [])
    a_sample = ((pkt.get('away_form_at') or {}).get('sample')) or (pkt.get('away_recent') or [])

    def summarize_form(sample: List[Dict[str, Any]], team_id: int) -> Dict[str, Optional[float]]:
        pts = []
        gf = []
        ga = []
        clean = 0
        fail = 0
        done = 0
        for fx in sample:
            p = _points_for(team_id, fx)
            if p is None:
                continue
            g_for, g_against = _goals_for_against(team_id, fx)
            pts.append(p)
            if isinstance(g_for, int) and isinstance(g_against, int):
                gf.append(g_for)
                ga.append(g_against)
                clean += 1 if g_against == 0 else 0
                fail += 1 if g_for == 0 else 0
                done += 1
        return {
            'points_avg': _avg(pts),
            'gd_avg': (_avg([ (gf[i] - ga[i]) for i in range(len(gf)) ]) if gf and ga and len(gf)==len(ga) else None),
            'clean_sheet_rate': _rate(clean, done),
            'failed_to_score_rate': _rate(fail, done),
            'goals_avg': _avg(gf),
            'concede_avg': _avg(ga),
            'n': done,
        }

    h_form = summarize_form(h_sample, home_id)
    a_form = summarize_form(a_sample, away_id)

    # days since last match (use most recent past in sample)
    def days_since_last(sample: List[Dict[str, Any]]) -> Optional[float]:
        if not kickoff or not sample:
            return None
        # find latest fixture before kickoff
        past = []
        for fx in sample:
            dtp = _parse_iso_dt((fx.get('fixture') or {}).get('date'))
            if dtp and dtp < kickoff:
                past.append(dtp)
        if not past:
            return None
        last_dt = max(past)
        return (kickoff - last_dt).days

    dsl_home = days_since_last(h_sample)
    dsl_away = days_since_last(a_sample)

    # H2H summary
    h2h = pkt.get('h2h_last') or []
    h_w = h_d = h_l = 0
    diffs = []
    for rec in h2h:
        goals = rec.get('goals', {})
        hg, ag = goals.get('home'), goals.get('away')
        if hg is None or ag is None:
            continue
        # map to current fixture's home/away ids
        r_home_id = rec.get('teams', {}).get('home', {}).get('id')
        r_away_id = rec.get('teams', {}).get('away', {}).get('id')
        # Convert to perspective of current home vs away
        if r_home_id == home_id and r_away_id == away_id:
            diff = hg - ag
        elif r_home_id == away_id and r_away_id == home_id:
            diff = ag - hg  # swap perspective
        else:
            # if different teams (shouldn't), skip
            continue
        diffs.append(diff)
        if diff > 0: h_w += 1
        elif diff == 0: h_d += 1
        else: h_l += 1
    total_h2h = len(diffs)
    h2h_home_win_rate = _rate(h_w, total_h2h)
    h2h_away_win_rate = _rate(h_l, total_h2h)
    h2h_draw_rate = _rate(h_d, total_h2h)
    h2h_goal_diff_avg = _avg(diffs)

    # Cards averages and penalty success rates from season stats
    home_yc_avg = _cards_avg_from_stats(pkt.get('home_stats') or {}, 'yellow')
    away_yc_avg = _cards_avg_from_stats(pkt.get('away_stats') or {}, 'yellow')
    home_rc_avg = _cards_avg_from_stats(pkt.get('home_stats') or {}, 'red')
    away_rc_avg = _cards_avg_from_stats(pkt.get('away_stats') or {}, 'red')
    home_pen_rate = _penalty_success_rate(pkt.get('home_stats') or {})
    away_pen_rate = _penalty_success_rate(pkt.get('away_stats') or {})

    features = {
        # core
        'home_advantage': 1,
        'days_since_last_match_home': dsl_home,
        'days_since_last_match_away': dsl_away,
        'round_number': round_num,
        'league_rank_diff': league_rank_diff,
        # form-based
        'home_form_points_avg': h_form['points_avg'],
        'away_form_points_avg': a_form['points_avg'],
        'home_goal_diff_avg': h_form['gd_avg'],
        'away_goal_diff_avg': a_form['gd_avg'],
        'home_clean_sheet_rate': h_form['clean_sheet_rate'],
        'away_clean_sheet_rate': a_form['clean_sheet_rate'],
        'home_failed_to_score_rate': h_form['failed_to_score_rate'],
        'away_failed_to_score_rate': a_form['failed_to_score_rate'],
        'home_goals_avg': h_form['goals_avg'],
        'away_goals_avg': a_form['goals_avg'],
        'home_concede_avg': h_form['concede_avg'],
        'away_concede_avg': a_form['concede_avg'],
        # h2h
        'h2h_home_win_rate': h2h_home_win_rate,
        'h2h_away_win_rate': h2h_away_win_rate,
        'h2h_draw_rate': h2h_draw_rate,
        'h2h_goal_diff_avg': h2h_goal_diff_avg,
        # discipline & penalties
        'home_yellow_cards_avg': home_yc_avg,
        'away_yellow_cards_avg': away_yc_avg,
        'home_red_cards_avg': home_rc_avg,
        'away_red_cards_avg': away_rc_avg,
        'home_penalty_success_rate': home_pen_rate,
        'away_penalty_success_rate': away_pen_rate,
    }

    # meta for later joins / comms
    meta = {
        'match_id': fx.get('fixture', {}).get('id'),
        'datetime': fx.get('fixture', {}).get('date'),
        'venue': (fx.get('fixture', {}) or {}).get('venue'),
        'referee': (fx.get('fixture', {}) or {}).get('referee'),
        'league': league_obj,
        'season': season,
        'round_str': round_str,
        'round_number': round_num,
        'home_team': home,
        'away_team': away,
        'odds': pkt.get('odds'),
    }
    return (features, meta)

def export_features_from_packets(packets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    feats = []
    metas = []
    for pkt in packets or []:
        f, m = export_match_features_from_packet(pkt)
        if f is not None:
            feats.append(f)
            metas.append(m)
    return feats, metas

# ------------------ VALUE TOOLS (MODEL + IMPLIED) ------------------

# Paths for model artifacts (same as infer.py expectations)
MODEL_PATH = "xgb_model.ubj"  # or "xgb_model.json"
META_PATH = "xgb_meta.json"
IMPUTER_PATH = "imputer.joblib"

_model_cache = {
    "booster": None,
    "feature_names": None,
    "classes": None,
    "imputer": None,
}

def _load_model_artifacts():
    # Cache to avoid re-loading between tool calls in same process
    if _model_cache["booster"] is not None:
        return (_model_cache["booster"], _model_cache["feature_names"], _model_cache["classes"], _model_cache["imputer"])
    booster = xgb.Booster()
    # Try UBJ first, then fall back to JSON if present
    try:
        booster.load_model(MODEL_PATH)
    except Exception as e:
        # If a JSON model exists, try that before failing
        alt = "xgb_model.json"
        if os.path.exists(alt):
            try:
                booster.load_model(alt)
                dbg(f"[model] Loaded fallback JSON model: {alt}")
            except Exception:
                raise e
        else:
            raise e
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    classes = np.array(meta["classes"])  # e.g., ['A','D','H']
    imputer = None
    try:
        if os.path.exists(IMPUTER_PATH):
            imputer = joblib.load(IMPUTER_PATH)
            try:
                n_fit = getattr(imputer, "n_features_in_", None)
                if n_fit is not None and n_fit != len(feature_names):
                    imputer = None
            except Exception:
                imputer = None
    except Exception:
        imputer = None
    _model_cache.update({
        "booster": booster,
        "feature_names": feature_names,
        "classes": classes,
        "imputer": imputer,
    })
    return booster, feature_names, classes, imputer

def _prepare_matrix(rows: List[Dict[str, Any]], feature_names: List[str], imputer=None):
    df = pd.DataFrame(rows)
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    X_df = df[feature_names]
    # If the imputer was fitted with feature names, pass a DataFrame to avoid sklearn warnings
    if imputer is not None:
        try:
            # Some older sklearn imputers may not accept DataFrames; try DataFrame first
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_trans = imputer.transform(X_df)
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_trans = imputer.transform(X_df.to_numpy())
            except Exception:
                # Give up on the imputer if totally incompatible
                dbg("[model] Imputer transform failed; proceeding without imputation")
                X_trans = X_df.to_numpy()
    else:
        X_trans = X_df.to_numpy()
    dmat = xgb.DMatrix(X_trans)
    return dmat

class PredictModelInput(BaseModel):
    features: List[Dict[str, Any]] = Field(..., description="List of feature dicts extracted from packets, in the model's expected shape.")
    metas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metas with match_id/fixture to align outputs.")

@tool("predict_model_probs", args_schema=PredictModelInput)
def predict_model_probs(features: List[Dict[str, Any]], metas: Optional[List[Dict[str, Any]]] = None) -> str:
    """Predict class probabilities (e.g., H/D/A) for provided features using the XGBoost model.

    If metas are provided, they will be echoed alongside the probabilities for alignment.
    """
    try:
        booster, feat_names, classes, imputer = _load_model_artifacts()
        dmat = _prepare_matrix(features, feat_names, imputer)
        proba = booster.predict(dmat)
        results = []
        for i, row in enumerate(proba):
            entry = {str(classes[j]): float(row[j]) for j in range(len(classes))}
            if metas and i < len(metas):
                entry = {**entry, "meta": metas[i]}
            results.append(entry)
        return json.dumps({"ok": True, "probs": results})
    except Exception as e:
        try:
            dbg(f"[predict_model_probs] ERROR: {e}")
        except Exception:
            pass
        return json.dumps({"ok": False, "error": str(e)})

class ImpliedAndDeltasInput(BaseModel):
    # expects packets or a simplified list that includes fixture_id and odds
    packets: Optional[List[Dict[str, Any]]] = Field(None, description="List of packets that contain odds.")
    # alternatively, provide a mapping {fixture_id: odds_list}
    odds_map: Optional[Dict[str, Any]] = Field(None, description="Optional explicit odds map keyed by fixture_id.")
    # optional model probabilities aligned by fixture index
    model_probs: Optional[List[Dict[str, Any]]] = Field(None, description="Optional list of model probability dicts per fixture to compute deltas vs implied.")

@tool("implied_and_deltas", args_schema=ImpliedAndDeltasInput)
def implied_and_deltas(packets: Optional[List[Dict[str, Any]]] = None, odds_map: Optional[Dict[str, Any]] = None, model_probs: Optional[List[Dict[str, Any]]] = None) -> str:
    """Compute implied probabilities from odds (1/odd) for 1X2 markets per fixture and optional deltas vs provided model_probs.

    Returns: { ok: true, results: [ { fixture_id, implied: {H,D,A}, probs: {...} | null, deltas: {...} | null } ] }
    """
    try:
        results = []
        # Build a flat list of (fixture_id, odds) entries
        entries = []
        if isinstance(packets, list):
            for p in packets:
                if (p or {}).get("mode") == "match" and p.get("fixture"):
                    fid = (p.get("fixture", {}) or {}).get("fixture", {}).get("id")
                    odds = p.get("odds")
                    if fid and odds:
                        entries.append((fid, odds))
        if isinstance(odds_map, dict):
            for fid, odds in odds_map.items():
                entries.append((fid, odds))
        # Helper to find a 1X2 market tuple (home, draw, away) per bookmaker entry
        def extract_1x2(odds_obj):
            # odds_obj is expected to be list of bookmakers with markets
            for bk in (odds_obj or []):
                markets = (bk or {}).get("markets") or {}
                if "1X2" in markets:
                    trio = markets["1X2"]
                    h = trio.get("home")
                    d = trio.get("draw")
                    a = trio.get("away")
                    if h and d and a:
                        return (h, d, a)
            return (None, None, None)
        # Compute implied and deltas
        for idx, (fid, odds_obj) in enumerate(entries):
            h, d, a = extract_1x2(odds_obj)
            if not (h and d and a):
                results.append({"fixture_id": fid, "implied": None, "probs": (model_probs[idx] if model_probs and idx < len(model_probs) else None), "deltas": None})
                continue
            try:
                H = 1.0 / float(h)
                D = 1.0 / float(d)
                A = 1.0 / float(a)
                # normalize to sum to 1 to account for overround if desired
                s = H + D + A
                implied = {"H": H / s, "D": D / s, "A": A / s} if s > 0 else {"H": H, "D": D, "A": A}
            except Exception:
                implied = None
            probs = model_probs[idx] if model_probs and idx < len(model_probs) else None
            deltas = None
            if implied and probs:
                deltas = {k: float(probs.get(k, 0.0)) - float(implied.get(k, 0.0)) for k in ("H", "D", "A")}
            results.append({"fixture_id": fid, "implied": implied, "probs": probs, "deltas": deltas})
        return json.dumps({"ok": True, "results": results})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

# ------------------ HAiKU VALUE AGENT ------------------

VALUE_SYSTEM = """
You are a small tools-only assistant. The user will not be asked for anything; use the provided packets/features/metas from state to:
1) Predict model probabilities with predict_model_probs
2) Compute implied probabilities and deltas with implied_and_deltas

Return only a compact JSON with { ok, probs, implied_and_deltas }.
"""

def value_agent_run(state: "GraphState") -> Dict[str, Any]:
    # Pull features/metas/packets for processing
    feats = state.get("retained_features") or []
    metas = state.get("retained_metas") or []
    dbg(f"[value_agent] features={len(feats)} metas={len(metas)}")
    # Rebuild packets view from tool_json if needed
    packets = []
    try:
        if state.get("tool_json"):
            obj = json.loads(state["tool_json"])
            if isinstance(obj, dict) and obj.get("ok"):
                packets = obj.get("packets") or []
    except Exception:
        packets = []
    dbg(f"[value_agent] packets={len(packets)}")
    # Step 1: model probabilities
    probs_raw = predict_model_probs.invoke({"features": feats, "metas": metas})
    probs = json.loads(probs_raw)
    dbg(f"[value_agent] predict_model_probs ok={probs.get('ok')} n={len(probs.get('probs') or [])}")
    if not probs.get("ok"):
        dbg(f"[value_agent] model error: {probs.get('error')}")
    # Step 2: implied + deltas, align by index
    model_probs = probs.get("probs") if probs.get("ok") else None
    implied_raw = implied_and_deltas.invoke({"packets": packets, "model_probs": model_probs})
    implied = json.loads(implied_raw)
    dbg(f"[value_agent] implied_and_deltas ok={implied.get('ok')} n={len(implied.get('results') or [])}")
    return {"ok": bool(probs.get("ok") and implied.get("ok")), "probs": probs, "implied_and_deltas": implied}
# ------------------ ODDS FILTERING (retain only 1X2 & Over/Under) ------------------

def filter_relevant_odds(raw_odds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter API-FOOTBALL odds payload to only include full time result (1X2) and goals over/under markets.

    Output structure (list of bookmakers with normalized markets):
    [
      {
        'bookmaker': 'Bet365',
        'last_update': '2025-09-26T12:00:00+00:00',
        'markets': {
            '1X2': {'home': '2.05', 'draw': '3.50', 'away': '3.60'},
            'over_under': {
                'lines': {
                   '2.5': {'over': '1.95', 'under': '1.85'},
                   '3.5': {'over': '2.90', 'under': '1.40'}
                }
            }
        }
      }, ...
    ]
    """
    filtered: List[Dict[str, Any]] = []
    for item in raw_odds:
        update_ts = item.get('update')
        for bk in item.get('bookmakers', []) or []:
            markets: Dict[str, Any] = {}
            for bet in bk.get('bets', []) or []:
                name = (bet.get('name') or '').lower()
                vals = bet.get('values', []) or []
                # Match Winner / 1X2
                if 'match winner' in name or name.strip() in ('1x2', 'winner'):
                    trio = {'home': None, 'draw': None, 'away': None}
                    for v in vals:
                        label = (v.get('value') or '').lower()
                        if label in ('home','1'):
                            trio['home'] = v.get('odd')
                        elif label in ('draw','x'):
                            trio['draw'] = v.get('odd')
                        elif label in ('away','2'):
                            trio['away'] = v.get('odd')
                    markets['1X2'] = trio
                # Goals Over/Under
                elif 'over/under' in name or 'goals over/under' in name or name.startswith('o/u'):
                    lines: Dict[str, Dict[str, Optional[str]]] = {}
                    for v in vals:
                        val = (v.get('value') or '').strip()
                        m = re.match(r'(Over|Under)\s+(\d+(?:\.\d)?)', val, re.I)
                        if not m:
                            continue
                        ou = m.group(1).lower()
                        line = m.group(2)
                        slot = lines.setdefault(line, {'over': None, 'under': None})
                        if ou == 'over':
                            slot['over'] = v.get('odd')
                        else:
                            slot['under'] = v.get('odd')
                    if lines:
                        markets['over_under'] = {'lines': lines}
            if markets:
                filtered.append({
                    'bookmaker': bk.get('name'),
                    'last_update': update_ts,
                    'markets': markets,
                })
    dbg(f"Filtered odds -> kept {len(filtered)} bookmaker entries (1X2 & O/U only)")
    return filtered

def assemble_team_packet(team_name: str, season: Optional[int]) -> PredictionPacket:
    dbg(f"Assemble team packet team={team_name} season_in={season}")
    team = search_team_by_name(team_name)
    if not team:
        raise ValueError(f"Team not found: {team_name}")
    tid = team["team"]["id"]

    season_guess = season or current_season()
    recent = get_team_recent_fixtures(tid, last=5)
    upcoming = get_team_upcoming(tid, nxt=3)

    # attach pre-match odds to each upcoming fixture when present
    upcoming_with_odds = []
    for fx in upcoming:
        fx_id = fx["fixture"]["id"]
        try:
            odds = get_prematch_odds_for_fixture(fx_id)
        except Exception:
            odds = []
        fx_copy = dict(fx)
        fx_copy["pre_match_odds"] = odds
        upcoming_with_odds.append(fx_copy)

    league_id = None
    if recent:
        league_id = recent[0]["league"]["id"]
        season_guess = recent[0]["league"]["season"]
    elif upcoming:
        league_id = upcoming[0]["league"]["id"]
        season_guess = upcoming[0]["league"]["season"]

    inj = get_injuries(tid, season_guess) if season_guess else []
    stats = get_team_statistics(league_id, season_guess, tid) if league_id else {}
    dbg(f"Team packet built tid={tid} league_id={league_id} season={season_guess} upcoming={len(upcoming_with_odds)}")

    return PredictionPacket(
        mode="team",
        query=team_name,
        resolved={"team_id": tid, "season": season_guess, "league_id": league_id},
        team_profile=team,
        team_recent_fixtures=recent,
        team_upcoming=upcoming_with_odds,
        team_injuries=inj,
        home_stats=stats if stats else None,
    )

def assemble_match_packet(home_team: str, away_team: str, date_hint: Optional[str], season: Optional[int], league_id: Optional[int] = None) -> PredictionPacket:
    dbg(f"Assemble match packet home={home_team} away={away_team} date_hint={date_hint} season_in={season}")
    th = search_team_by_name(home_team)
    ta = search_team_by_name(away_team)
    if not th or not ta:
        raise ValueError(f"Could not resolve both teams: {home_team} vs {away_team}")
    th_id, ta_id = th["team"]["id"], ta["team"]["id"]

    dfrom, dto = _date_range_from_hint(date_hint)
    search_season = season or current_season()
    fx = find_fixture_by_teams_date(th_id, ta_id, dfrom, dto, search_season, league_id=league_id)
    if not fx:
        # fallback: broaden without date window (season only)
        fx = find_fixture_by_teams_date(th_id, ta_id, None, None, search_season, league_id=league_id)

    pkt = PredictionPacket(
        mode="match",
        query=f"{home_team} vs {away_team}",
        resolved={"home_team_id": th_id, "away_team_id": ta_id, "date_from": dfrom, "date_to": dto},
    )

    pkt.h2h_last = head_to_head(th_id, ta_id, last=10)

    if fx:
        dbg(f"Fixture resolved id={fx['fixture']['id']} date={fx['fixture']['date']}")
        pkt.fixture = fx
        fx_id = fx["fixture"]["id"]
        league_id, season_id = league_season_from_fixture(fx)
        pkt.league_table = get_standings(league_id, season_id)
        pkt.home_stats = get_team_statistics(league_id, season_id, th_id)
        pkt.away_stats = get_team_statistics(league_id, season_id, ta_id)
        # compute historical form as of kickoff
        try:
            pkt.home_form_at = compute_team_form_as_of(th_id, league_id, season_id, fx["fixture"]["date"], last=5)
        except Exception:
            pkt.home_form_at = None
        try:
            pkt.away_form_at = compute_team_form_as_of(ta_id, league_id, season_id, fx["fixture"]["date"], last=5)
        except Exception:
            pkt.away_form_at = None
        pkt.lineups = get_lineups(fx_id)
        # recent fixtures for each team (exclude this fixture if already played/upcoming)
        try:
            pkt.home_recent = get_team_recent_fixtures(th_id, last=10)
        except Exception:
            pkt.home_recent = None
        try:
            pkt.away_recent = get_team_recent_fixtures(ta_id, last=10)
        except Exception:
            pkt.away_recent = None
        # next fixtures for season (to gauge scheduling congestion). We fetch next=2 and drop the current fixture id.
        try:
            h_next_all = get_team_upcoming(th_id, nxt=3)
            pkt.home_next_after = [f for f in h_next_all if f["fixture"]["id"] != fx_id][:2]
        except Exception:
            pkt.home_next_after = None
        try:
            a_next_all = get_team_upcoming(ta_id, nxt=3)
            pkt.away_next_after = [f for f in a_next_all if f["fixture"]["id"] != fx_id][:2]
        except Exception:
            pkt.away_next_after = None
        try:
            pkt.events = get_events(fx_id)
        except Exception:
            pkt.events = None
        try:
            pkt.odds = get_prematch_odds_for_fixture(fx_id)  # pre-match odds
        except Exception:
            pkt.odds = None
    else:
        pkt.home_stats = {}
        pkt.away_stats = {}
        pkt.league_table = {}
    return pkt

def assemble_match_packet_from_fixture(fx: Dict[str, Any]) -> PredictionPacket:
    """Build a full match packet given an already resolved fixture object.

    Avoids re-searching by team names. Gathers standings, team stats, lineups,
    events, odds, and recent/next fixtures.
    """
    home = fx["teams"]["home"]
    away = fx["teams"]["away"]
    th_id, ta_id = home["id"], away["id"]
    home_name = home.get("name") or str(th_id)
    away_name = away.get("name") or str(ta_id)
    pkt = PredictionPacket(
        mode="match",
        query=f"{home_name} vs {away_name}",
        resolved={"home_team_id": th_id, "away_team_id": ta_id},
    )
    pkt.fixture = fx
    fx_id = fx["fixture"]["id"]
    league_id, season_id = league_season_from_fixture(fx)
    try:
        pkt.league_table = get_standings(league_id, season_id)
    except Exception:
        pkt.league_table = None
    try:
        pkt.home_stats = get_team_statistics(league_id, season_id, th_id)
    except Exception:
        pkt.home_stats = None
    try:
        pkt.away_stats = get_team_statistics(league_id, season_id, ta_id)
    except Exception:
        pkt.away_stats = None
    # historical form as of kickoff
    try:
        pkt.home_form_at = compute_team_form_as_of(th_id, league_id, season_id, fx["fixture"]["date"], last=5)
    except Exception:
        pkt.home_form_at = None
    try:
        pkt.away_form_at = compute_team_form_as_of(ta_id, league_id, season_id, fx["fixture"]["date"], last=5)
    except Exception:
        pkt.away_form_at = None
    try:
        pkt.lineups = get_lineups(fx_id)
    except Exception:
        pkt.lineups = None
    try:
        pkt.h2h_last = head_to_head(th_id, ta_id, last=10)
    except Exception:
        pkt.h2h_last = None
    try:
        pkt.home_recent = get_team_recent_fixtures(th_id, last=10)
    except Exception:
        pkt.home_recent = None
    try:
        pkt.away_recent = get_team_recent_fixtures(ta_id, last=10)
    except Exception:
        pkt.away_recent = None
    try:
        h_next_all = get_team_upcoming(th_id, nxt=3)
        pkt.home_next_after = [f for f in h_next_all if f["fixture"]["id"] != fx_id][:2]
    except Exception:
        pkt.home_next_after = None
    try:
        a_next_all = get_team_upcoming(ta_id, nxt=3)
        pkt.away_next_after = [f for f in a_next_all if f["fixture"]["id"] != fx_id][:2]
    except Exception:
        pkt.away_next_after = None
    try:
        pkt.events = get_events(fx_id)
    except Exception:
        pkt.events = None
    try:
        pkt.odds = get_prematch_odds_for_fixture(fx_id)
    except Exception:
        pkt.odds = None
    return pkt

# ------------------ TOOL (LangChain Tool) ------------------

# ------------------ TOOL (LangChain Tool) ------------------

class TeamSearchInput(BaseModel):
    team: str
    date_hint: Optional[str] = Field(None)
    season: Optional[int] = Field(None)
    league: Optional[str] = Field(None, description="Optional league constraint")

@tool("team_search", args_schema=TeamSearchInput)
def team_search(team: str, date_hint: Optional[str] = None, season: Optional[int] = None, league: Optional[str] = None) -> str:
    """Given a team name, resolve its next (or hinted) fixture and compute a full match packet; fall back to a team packet if no fixture is found."""
    try:
        if date_hint and _is_past_only_window(date_hint):
            playful = (
                "Future mode only ⚽🔮 — ask for 'today', 'tomorrow', or a coming window and I’ll gather the goods."
            )
            return json.dumps({"ok": False, "error": playful})
        resolved_season = season if season is not None else current_season()
        league_entry = search_league_by_name(league) if league else None
        league_id = (league_entry or {}).get("league", {}).get("id") if league_entry else None
        t = search_team_by_name(team)
        if not t:
            return json.dumps({"ok": False, "error": f"Team not found: {team}"})
        tid = t["team"]["id"]
        dfrom, dto = _date_range_from_hint(date_hint)
        fx = find_fixture_for_team_by_hint(tid, dfrom, dto, resolved_season, league_id=league_id)
        if fx:
            pkt = assemble_match_packet_from_fixture(fx)
            return json.dumps({"ok": True, "packets": [json.loads(pkt.model_dump_json())]})
        # fallback to team-centric
        pkt = assemble_team_packet(team, resolved_season)
        return json.dumps({"ok": True, "packets": [json.loads(pkt.model_dump_json())]})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

class FixtureSearchInput(BaseModel):
    home_team: str
    away_team: str
    date_hint: Optional[str] = Field(None)
    season: Optional[int] = Field(None)
    league: Optional[str] = Field(None)

@tool("fixture_search", args_schema=FixtureSearchInput)
def fixture_search(home_team: str, away_team: str, date_hint: Optional[str] = None, season: Optional[int] = None, league: Optional[str] = None) -> str:
    """Given two teams (and optional date/league), resolve the fixture and compute a full match packet."""
    try:
        if date_hint and _is_past_only_window(date_hint):
            playful = (
                "I’m tuned to upcoming kickoffs only — try 'today', 'tomorrow', 'this weekend', or a specific future date."
            )
            return json.dumps({"ok": False, "error": playful})
        resolved_season = season if season is not None else current_season()
        league_entry = search_league_by_name(league) if league else None
        league_id = (league_entry or {}).get("league", {}).get("id") if league_entry else None
        pkt = assemble_match_packet(home_team, away_team, date_hint, resolved_season, league_id=league_id)
        return json.dumps({"ok": True, "packets": [json.loads(pkt.model_dump_json())]})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

class LeagueSearchInput(BaseModel):
    league: str
    date_hint: Optional[str] = Field(None, description="Date window like 'this weekend', 'today', or 'YYYY-MM-DD'")
    season: Optional[int] = Field(None)
    max_fixtures: Optional[int] = Field(
        None,
        description="Max fixtures to expand into full packets. If omitted, uses 60 for month-year windows (e.g., 'March 2024'), otherwise 10.",
    )

def _is_month_year_hint(hint: Optional[str]) -> bool:
    """Determine whether a date hint targets a whole month in a given year.

    Accepts forms like:
    - "March 2024"
    - "in March 2024" (leading "in" is allowed)

    Matching is case-insensitive and ignores commas. Returns True only when the
    hint matches a full month-year (e.g., "April 2025").

    Args:
        hint: Free-form date hint string from the user.

    Returns:
        bool: True if the hint represents a month-year window, False otherwise.
    """
    if not hint:
        return False
    s = hint.strip()
    # Allow prefixes like 'in March 2024'
    if s.lower().startswith("in "):
        s = s[3:].strip()
    # Normalize multiple spaces and optional comma
    s = re.sub(r",", "", s)
    # Match '<Month> YYYY' case-insensitively
    return re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$", s, re.IGNORECASE) is not None

@tool("league_search", args_schema=LeagueSearchInput)
def league_search(league: str, date_hint: Optional[str] = None, season: Optional[int] = None, max_fixtures: Optional[int] = None) -> str:
    """Agent tool: expand fixtures for a league within a date window and return packets.

    Behavior:
    - Resolves the league by name and determines a date window using the parser.
    - Fetches fixtures within the window (or next N if no window supported by API) and expands
      up to `max_fixtures` into full match packets. A league meta packet is included first.
    - If `max_fixtures` is None, a dynamic default is applied: 60 for month-year windows
      (e.g., "March 2024"), otherwise 10.

    Args:
        league: The league name (e.g., "Premier League", "Super League 1").
        date_hint: A relative/absolute hint like "today", "this weekend", "YYYY-MM-DD",
                   or month-year (e.g., "March 2024").
        season: Optional season starting year (API-FOOTBALL convention). Defaults to current season.
        max_fixtures: Max number of fixtures to expand. If None, uses dynamic default as above.

    Returns:
        str: JSON string with shape {"ok": true, "packets": [...] } on success or
             {"ok": false, "error": "..."} on failure.
    """
    # Dynamic cap: 60 for month-year windows, else 10 (unless explicitly provided)
    if max_fixtures is None:
        max_fixtures = 60 if _is_month_year_hint(date_hint) else 10
    try:
        # If a date hint is provided and it's strictly in the past, refuse
        if date_hint and _is_past_only_window(date_hint):
            playful = (
                "Let's keep our eyes on the road ahead — I only look to upcoming fixtures. "
                "Try a future window like 'today', 'tomorrow', 'this weekend', or a specific date."
            )
            return json.dumps({"ok": False, "error": playful})
        resolved_season = season if season is not None else current_season()
        league_entry = search_league_by_name(league)
        if not league_entry:
            return json.dumps({"ok": False, "error": f"League not found: {league}"})
        league_id = league_entry["league"]["id"]
        dfrom, dto = _date_range_from_hint(date_hint) if date_hint else (None, None)
        fixtures = get_league_fixtures(league_id, resolved_season, dfrom, dto, nxt=max_fixtures)
        packets: List[Dict[str, Any]] = []
        for fx in fixtures[:max_fixtures]:
            try:
                pkt = assemble_match_packet_from_fixture(fx)
                packets.append(json.loads(pkt.model_dump_json()))
            except Exception as e:
                dbg(f"League fixture expansion failed: {e}")
        meta = PredictionPacket(
            mode="league",
            query=league,
            resolved={"league_id": league_id, "season": resolved_season, "date_from": dfrom, "date_to": dto},
            league_info=league_entry,
            league_fixtures=fixtures,
        )
        return json.dumps({"ok": True, "packets": [json.loads(meta.model_dump_json())] + packets})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

class MultiLeagueSearchInput(BaseModel):
    leagues: List[str]
    date_hint: Optional[str] = Field(None, description="Date window like 'this weekend', 'today', or 'YYYY-MM-DD'")
    season: Optional[int] = Field(None)
    max_fixtures: Optional[int] = Field(
        None,
        description="Max fixtures per league to expand into full packets. If omitted, uses 60 for month-year windows (e.g., 'March 2024'), otherwise 10.",
    )

@tool("multi_league_search", args_schema=MultiLeagueSearchInput)
def multi_league_search(leagues: List[str], date_hint: Optional[str] = None, season: Optional[int] = None, max_fixtures: Optional[int] = None) -> str:
    """Agent tool: expand fixtures for multiple leagues within a shared date window.

    Behavior:
    - For each league name, resolves league and fetches fixtures within the parsed date window.
    - Emits a league meta packet followed by expanded match packets for up to `max_fixtures` fixtures.
    - Applies a dynamic default for `max_fixtures` when None: 60 for month-year windows,
      otherwise 10.

    Args:
        leagues: List of league names (e.g., ["Premier League", "Super League 1"]).
        date_hint: Shared date window for all leagues ("today", "last weekend", "2025-09-26",
                   or month-year like "April 2025").
        season: Optional season starting year (API-FOOTBALL convention). Defaults to current season.
        max_fixtures: Max fixtures per league to expand; uses dynamic default if None.

    Returns:
        str: JSON string with shape {"ok": true, "packets": [...] } or an error envelope
             when validation or lookups fail.
    """
    # Dynamic cap: 60 for month-year windows, else 10 (unless explicitly provided)
    if max_fixtures is None:
        max_fixtures = 60 if _is_month_year_hint(date_hint) else 10
    try:
        if not leagues:
            return json.dumps({"ok": False, "error": "No leagues provided."})
        if date_hint and _is_past_only_window(date_hint):
            playful = (
                "Time only flows forward here — I fetch future fixtures, not recaps. "
                "Try 'today', 'tomorrow', 'this weekend', or a specific upcoming date."
            )
            return json.dumps({"ok": False, "error": playful})
        resolved_season = season if season is not None else current_season()
        dfrom, dto = _date_range_from_hint(date_hint) if date_hint else (None, None)
        all_packets: List[Dict[str, Any]] = []
        for lg in leagues:
            try:
                league_entry = search_league_by_name(lg)
                if not league_entry:
                    dbg(f"League not found in multi search: {lg}")
                    continue
                league_id = league_entry["league"]["id"]
                fixtures = get_league_fixtures(league_id, resolved_season, dfrom, dto, nxt=max_fixtures)
                # league meta packet
                meta = PredictionPacket(
                    mode="league",
                    query=lg,
                    resolved={"league_id": league_id, "season": resolved_season, "date_from": dfrom, "date_to": dto},
                    league_info=league_entry,
                    league_fixtures=fixtures,
                )
                all_packets.append(json.loads(meta.model_dump_json()))
                # expand fixtures
                for fx in fixtures[:max_fixtures]:
                    try:
                        pkt = assemble_match_packet_from_fixture(fx)
                        all_packets.append(json.loads(pkt.model_dump_json()))
                    except Exception as ex:
                        dbg(f"Fixture expansion failed for {lg}: {ex}")
            except Exception as inner:
                dbg(f"Multi-league inner error for {lg}: {inner}")
                continue
        if not all_packets:
            return json.dumps({"ok": False, "error": "No data found for the requested leagues."})
        return json.dumps({"ok": True, "packets": all_packets})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

class BatchSearchInput(BaseModel):
    queries: List[Dict[str, Any]]
    season: Optional[int] = Field(None)

@tool("batch_search", args_schema=BatchSearchInput)
def batch_search(queries: List[Dict[str, Any]], season: Optional[int] = None) -> str:
    """Run fixture/team searches for multiple parsed queries and return all packets."""
    try:
        packets: List[Dict[str, Any]] = []
        resolved_season = season if season is not None else current_season()
        for q in (queries or []):
            qtype = (q.get("type") or "").lower()
            if qtype == "match" and q.get("home_team") and q.get("away_team"):
                out = json.loads(fixture_search.invoke({
                    "home_team": q.get("home_team"),
                    "away_team": q.get("away_team"),
                    "date_hint": q.get("date_hint"),
                    "season": resolved_season,
                }))
            elif qtype == "team" and q.get("team"):
                out = json.loads(team_search.invoke({
                    "team": q.get("team"),
                    "date_hint": q.get("date_hint"),
                    "season": resolved_season,
                }))
            elif qtype == "league" and q.get("league"):
                out = json.loads(league_search.invoke({
                    "league": q.get("league"),
                    "date_hint": q.get("date_hint"),
                    "season": resolved_season,
                }))
            else:
                out = {"ok": False, "error": f"Unsupported or incomplete query: {q}"}
            if out.get("ok"):
                packets.extend(out.get("packets") or [])
        if not packets:
            return json.dumps({"ok": False, "error": "No valid items in batch."})
        return json.dumps({"ok": True, "packets": packets})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

# ------------------ AGENT (Anthropic Haiku) ------------------

import anthropic
HAIKU_MODEL = "claude-3-haiku-20240307"  # previous model
# HAIKU_MODEL = "claude-3-5-sonnet-20240620"  # previous model
# HAIKU_MODEL = "claude-4-sonnet-20250514"
SONNET4_MODEL = "claude-4-sonnet-20250514"

EXTRACTION_SYSTEM = """
You extract structured data from user queries about association football (soccer).

You will receive a request about:
- one match, or
- one team, or
- a batch of matches/teams, or
- a league.

OUTPUT FORMAT — ALWAYS return ONLY valid JSON matching this schema:

{
  "queries": [
    {
      "type": "match" | "team" | "league",
      "home_team": string | null,
      "away_team": string | null,
      "team": string | null,
      "league": string | null,
      "date_hint": string | null
    }
  ]
}

Rules:
- For a MATCH: set type="match", fill home_team and away_team; set team and league to null.
  - Home/away inference:
    - If text says "X vs Y", X is home.
    - If text uses "@", "at", or "away at": subject team is away, the other is home.
    - If text says "X hosts Y", X is home.
    - If unclear, first mentioned is home.
- For a TEAM-only request: set type="team", set team, home_team, away_team to null.
- For a LEAGUE request: set type="league", set league, team, home_team, away_team to null.
- For BATCH requests: return one object per item in "queries".
- Names must be as close as possible to official club or league names (e.g., "Manchester United", "Real Madrid", "Premier League").
  - Normalize common aliases:
    - "Man Utd" → "Manchester United"
    - "Man City" → "Manchester City"
    - "PSG" → "Paris Saint-Germain"
    - "AEK" / "A.E.K." → "AEK Athens"
    - "PAOK" → "PAOK"
    - "Oly" / "Olympiacos" → "Olympiacos"
  - Do not invent leagues. Do not translate club names across languages; prefer the club’s standard international/English form when widely used (e.g., "Olympiacos", "Panathinaikos").
- date_hint: copy any explicit or relative temporal phrase (e.g., "Saturday", "tomorrow", "this weekend", "2025-09-26"). If none, use null.
- Do NOT add fields. Do NOT include explanations or comments. Output ONLY JSON.

Examples:

Input: "Chelsea vs Brighton this weekend"
Output: { "queries":[{ "type":"match","home_team":"Chelsea","away_team":"Brighton & Hove Albion","team":null,"league":null,"date_hint":"this weekend"}] }

Input: "Olympiacos this Sunday"
Output: { "queries":[{ "type":"team","home_team":null,"away_team":null,"team":"Olympiacos","league":null,"date_hint":"this Sunday"}] }

Input: "Aek vs PAOK and Aris vs Panseraikos this week"
Output: { "queries":[
  { "type":"match","home_team":"AEK Athens","away_team":"PAOK","team":null,"league":null,"date_hint":"this week"},
  { "type":"match","home_team":"Aris Thessaloniki","away_team":"Panserraikos","team":null,"league":null,"date_hint":"this week"}
] }

Input: "Premier League"
Output: { "queries":[{ "type":"league","home_team":null,"away_team":null,"team":null,"league":"Premier League","date_hint":null}] }

Input: "Greek Superleague and Premier League this Saturday"
Output: { "queries":[
    { "type":"league","home_team":null,"away_team":null,"team":null,"league":"Super League 1","date_hint":"Saturday"},
    { "type":"league","home_team":null,"away_team":null,"team":null,"league":"Premier League","date_hint":"Saturday"}
] }

Input: "What happened in Premier League and Greek Super League last weekend"
Output: { "queries":[
    { "type":"league","home_team":null,"away_team":null,"team":null,"league":"Premier League","date_hint":"last weekend"},
    { "type":"league","home_team":null,"away_team":null,"team":null,"league":"Super League 1","date_hint":"last weekend"}
] }
"""

def haiku_extract(nl: str) -> Dict[str, Optional[str]]:
    dbg(f"Haiku extract start prompt={nl!r}")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=300,
        temperature=0,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": nl}],
    )
    content = "".join([b.text for b in msg.content if hasattr(b, "text")]) or (msg.content[0].text if msg.content else "")
    m = re.search(r"\{.*\}\s*$", content, re.DOTALL)
    raw = m.group(0) if m else content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"queries": [{"type": None, "home_team": None, "away_team": None, "team": None, "league": None, "date_hint": None}]}
    # Normalize to always return a dict with "queries": list
    if isinstance(data, dict) and "queries" in data and isinstance(data.get("queries"), list):
        queries = data["queries"] or []
    else:
        # Backward compatibility: flatten to a single query object
        queries = [{
            "type": None,
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "team": data.get("team"),
            "league": data.get("league"),
            "date_hint": data.get("date_hint"),
        }]
    # sanitize
    normed = []
    for q in queries:
        qn = {k: (v.strip() if isinstance(v, str) else v) for k, v in q.items()}
        normed.append(qn)
    out = {"queries": normed}
    dbg(f"Haiku parsed -> {out}")
    return out

# ------------------ SONNET 4 CLASSIFIER ------------------

SONNET_CLASSIFIER_SYSTEM = """
You are a strict router for a football betting assistant. Only allow these intents:
1) best_value_markets: Identify best value markets for 1X2 (home/draw/away) only, for a match OR league OR multiple leagues OR a batch (mixture of items)
2) xn_bet_request: A request to build an xN bet (e.g., 4x, 5x) composed ONLY of 1X2 selections, scoped to a league or group of leagues

If the prompt does not clearly match one of the above, you MUST refuse by outputting {"allow": false} only.
Keep in mind it could still be something football related like "When was the premier league founded?" or "Who won the World Cup in 2018?" or "How many goals did chelsea goal last week" — these are NOT allowed.

When allowed, output strictly the JSON below with no commentary:
{
  "allow": true,
  "intent": "best_value_markets" | "xn_bet_request",
  "xn": number | null,
  "queries": [
    {
      "type": "match" | "league",
      "home_team": string | null,
      "away_team": string | null,
      "league": string | null,
      "date_hint": string | null
    }
  ]
}

Rules:
- For group of leagues: return one object per league in "queries" (type="league").
- For a batch request mixing matches and leagues: include each item.
- For xn_bet_request, set "xn" to the requested integer; otherwise use null.
- Names should be close to official names (e.g., "Premier League", "Brighton & Hove Albion").
- IMPORTANT (1X2 ONLY): If the prompt asks for any market other than 1X2 (e.g., Over/Under, Both Teams To Score/BTTS, handicaps including Asian Handicap, Draw No Bet, Double Chance, correct score, player or team props like corners/cards/shots/scorers, HT/FT), you MUST refuse by outputting {"allow": false}.
- IMPORTANT (No futures): If the prompt asks for outright/future markets (season winners, title/league winner, top scorer/golden boot, relegation/promotion, etc.), you MUST refuse by outputting {"allow": false}. Only per-match or per-round best value markets and xN bets are in scope.
 - If the market type is not specified but the request is otherwise in scope, assume 1X2 by default (do NOT refuse for lack of explicit '1X2').
"""

def sonnet_classify(nl: str) -> Dict[str, Any]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=SONNET4_MODEL,
        max_tokens=300,
        temperature=0,
        system=SONNET_CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": nl}],
    )
    content = "".join([b.text for b in msg.content if hasattr(b, "text")]) or (msg.content[0].text if msg.content else "")
    m = re.search(r"\{.*\}\s*$", content, re.DOTALL)
    raw = m.group(0) if m else content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {"allow": False}
    # Basic shape enforcement
    allow = bool(data.get("allow"))
    if not allow:
        return {"allow": False}
    intent = data.get("intent")
    xn = data.get("xn") if isinstance(data.get("xn"), (int, float)) else None
    queries = data.get("queries") if isinstance(data.get("queries"), list) else []
    # Normalize strings
    normed = []
    for q in queries:
        normed.append({
            "type": (q.get("type") or "").strip() or None,
            "home_team": (q.get("home_team") or None),
            "away_team": (q.get("away_team") or None),
            "league": (q.get("league") or None),
            "date_hint": (q.get("date_hint") or None),
        })
    return {"allow": True, "intent": intent, "xn": xn, "queries": normed}

# ------------------ GRAPH STATE & NODES ------------------

class GraphState(TypedDict):
    user_prompt: str
    # parsed
    queries: List[Dict[str, Any]]
    # classifier
    intent: Optional[str]
    xn: Optional[int]
    # io
    tool_json: Optional[str]
    final_text: Optional[str]
    # human-rendered text summary
    retained_text: Optional[str]
    # retention payloads (written by early/late nodes; persisted via checkpointer)
    retained_intent: Optional[str]
    retained_xn: Optional[int]
    retained_odds: Optional[List[Dict[str, Any]]]
    retained_matches: Optional[List[Dict[str, Any]]]
    retained_features: Optional[List[Dict[str, Any]]]
    retained_metas: Optional[List[Dict[str, Any]]]
    retained_value: Optional[List[Dict[str, Any]]]
    retained_value_rows: Optional[List[Dict[str, Any]]]
    retained_selection: Optional[Dict[str, Any]]

def agent_node(state: GraphState) -> GraphState:
    dbg("Agent node: parsing user prompt")
    parsed = haiku_extract(state["user_prompt"])  # returns {"queries": [...]}
    dbg(f"Agent node parsed: {parsed}")
    return {**state, "queries": parsed.get("queries") or []}

def sonnet_entry_node(state: GraphState) -> GraphState:
    """Entry classifier that only allows supported intents, else produces a polite refusal."""
    nl = state.get("user_prompt") or ""
    # Light normalization for common typos
    nl = re.sub(r"\bpics\b", "picks", nl, flags=re.I)
    nl = re.sub(r"\bpikcs\b|\bpikcs\b|\bpkcs\b", "picks", nl, flags=re.I)
    # Deterministic guard: refuse outright/future winner/top-scorer style requests
    outright_terms = [
        "outright", "title winner", "league winner", "to win the league", "champion",
        "top scorer", "golden boot", "relegation", "promotion", "to finish top",
        "win the cup", "outrights"
    ]
    low = nl.lower()
    if any(term in low for term in outright_terms):
        return {**state, "final_text": "Unfortunately I can’t help with that. I only handle 1X2 (home/draw/away) best value markets or xN 1X2 bet requests scoped to a league, group of leagues, batch, or a single match."}
    # Deterministic guard: refuse non-1X2 market types
    non_1x2_terms = [
        "over/under", "over under", "o/u", "total goals", "btts", "both teams to score",
        "handicap", "asian handicap", "ahc", "draw no bet", "dnb", "double chance",
        "correct score", "cs", "half time/full time", "ht/ft", "corners", "cards",
        "shots", "assists", "first scorer", "anytime scorer", "player props", "team props"
    ]
    if any(term in low for term in non_1x2_terms):
        return {**state, "final_text": "Unfortunately I can’t help with that. I only handle 1X2 (home/draw/away) best value markets or xN 1X2 bet requests scoped to a league, group of leagues, batch, or a single match."}
    cls = sonnet_classify(nl)
    if not cls.get("allow"):
        # hard stop with refusal matching the user's constraints
        return {**state, "final_text": "Unfortunately I can’t help with that. I only handle best value markets or xN bet requests scoped to a league, group of leagues, batch, or a single match."}
    # Validate allowed intent and queries are present and supported
    intent = (cls.get("intent") or "").strip()
    if intent not in ("best_value_markets", "xn_bet_request"):
        return {**state, "final_text": "Unfortunately I can’t help with that. I only handle best value markets or xN bet requests scoped to a league, group of leagues, batch, or a single match."}
    raw_qs = cls.get("queries") or []
    # keep only supported types and minimally valid entries
    qs = []
    for q in raw_qs:
        qtype = (q.get("type") or "").lower()
        if qtype not in ("match", "league"):
            continue
        if qtype == "match" and not (q.get("home_team") and q.get("away_team")):
            continue
        if qtype == "league" and not q.get("league"):
            continue
        qs.append(q)
    if not qs:
        # refuse if nothing actionable
        return {**state, "final_text": "Unfortunately I can’t help with that. I only handle best value markets or xN bet requests scoped to a league, group of leagues, batch, or a single match."}
    # If allowed, stash intent/xn and normalized queries, then proceed
    nxn = (int(cls.get("xn")) if cls.get("xn") is not None else None)
    new_state = {**state, "intent": intent, "xn": nxn, "queries": qs}
    # Retain question type for downstream/next runs (MemorySaver will checkpoint this)
    new_state["retained_intent"] = intent
    new_state["retained_xn"] = nxn
    # Ensure retention containers exist
    new_state.setdefault("retained_odds", [])
    new_state.setdefault("retained_matches", [])
    new_state.setdefault("retained_features", [])
    new_state.setdefault("retained_metas", [])
    new_state.setdefault("retained_value", [])
    new_state.setdefault("retained_value_rows", [])
    new_state.setdefault("retained_selection", None)
    return new_state

def tool_node(state: GraphState) -> GraphState:
    # If a previous node (e.g., classifier) already produced a final message, do not override it
    if state.get("final_text"):
        return state
    qs = state.get("queries") or []
    if not qs:
        return {**state, "final_text": "I couldn’t identify a team, match, or league. Please specify e.g. 'Chelsea vs Brighton', 'Panathinaikos', or 'Premier League'."}
    dbg("Tool node routing to appropriate search tool")
    # Block past-only windows with a playful response
    try:
        date_hints = [q.get("date_hint") for q in qs if q.get("date_hint")]
        if date_hints and all(_is_past_only_window(h) for h in date_hints):
            msg = (
                "I only gaze forward — the crystal ball’s tuned for upcoming football, not history books. "
                "Ask me about today or the future (e.g., 'this weekend', 'tomorrow', a specific YYYY-MM-DD), and I’ll fetch everything for you!"
            )
            return {**state, "final_text": msg}
    except Exception:
        pass
    if len(qs) > 1:
        # Specialized multi-league path: all items are leagues and share the same date_hint
        all_leagues = [q.get("league") for q in qs if (q.get("type") or "").lower() == "league" and q.get("league")]
        if all_leagues and len(all_leagues) == len(qs):
            dhints = set([(q.get("date_hint") or None) for q in qs])
            if len(dhints) == 1 and list(dhints)[0]:
                payload = multi_league_search.invoke({
                    "leagues": all_leagues,
                    "date_hint": list(dhints)[0],
                })
            else:
                payload = batch_search.invoke({"queries": qs})
        else:
            payload = batch_search.invoke({"queries": qs})
    else:
        q = qs[0]
        qtype = (q.get("type") or "").lower()
        if qtype == "match" and q.get("home_team") and q.get("away_team"):
            payload = fixture_search.invoke({
                "home_team": q.get("home_team"),
                "away_team": q.get("away_team"),
                "date_hint": q.get("date_hint"),
            })
        elif qtype == "team" and q.get("team"):
            payload = team_search.invoke({
                "team": q.get("team"),
                "date_hint": q.get("date_hint"),
            })
        elif qtype == "league" and q.get("league"):
            payload = league_search.invoke({
                "league": q.get("league"),
                "date_hint": q.get("date_hint"),
            })
        else:
            return {**state, "final_text": "I couldn’t identify a team, match, or league. Please specify e.g. 'Chelsea vs Brighton', 'Panathinaikos', or 'Premier League'."}
    dbg("Tool node received payload")
    return {**state, "tool_json": payload}

def answer_node(state: GraphState) -> GraphState:
    if state.get("final_text"):
        return state
    raw = state.get("tool_json")
    dbg("Answer node processing tool output")
    try:
        obj = json.loads(raw)
    except Exception:
        dbg("Answer node JSON decode failed")
        return {**state, "final_text": "Unexpected tool error."}

    if not obj.get("ok"):
        dbg(f"Tool indicated failure: {obj.get('error')}")
        return {**state, "final_text": f"❌ {obj.get('error', 'Unknown error.')}"}
    # Support both single and batch outputs via unified 'packets'
    packets = obj.get("packets")
    if packets is None:
        # Back-compat: single packet under 'prediction_packet'
        packets = [obj.get("prediction_packet")]
    packets = [p for p in (packets or []) if p]
    if not packets:
        return {**state, "final_text": "❌ No data returned."}
    # Build a concise summary
    lines: List[str] = []
    # Persist odds + basic per-match info + extracted features for retention
    try:
        # Extract basic match info and odds
        basics: List[Dict[str, Any]] = []
        odds_list: List[Dict[str, Any]] = []
        for pkt in packets:
            if (pkt or {}).get("mode") == "match" and pkt.get("fixture"):
                fx = pkt["fixture"]
                league_obj = fx.get("league", {})
                basics.append({
                    "fixture_id": fx.get("fixture", {}).get("id"),
                    "kickoff": fx.get("fixture", {}).get("date"),
                    "home": (fx.get("teams", {}).get("home", {}) or {}).get("name"),
                    "away": (fx.get("teams", {}).get("away", {}) or {}).get("name"),
                    "league": {
                        "id": league_obj.get("id"),
                        "name": league_obj.get("name"),
                        "round": league_obj.get("round"),
                        "season": league_obj.get("season"),
                    },
                })
                odds_list.append({
                    "fixture_id": fx.get("fixture", {}).get("id"),
                    "odds": pkt.get("odds"),
                })
        # Compute features from packets
        feats, _metas = export_features_from_packets(packets)
        # Merge into state (append-style to preserve across multiple tool calls if any)
        prev_basics = state.get("retained_matches") or []
        prev_odds = state.get("retained_odds") or []
        prev_feats = state.get("retained_features") or []
        prev_metas = state.get("retained_metas") or []
        state = {**state,
                 "retained_matches": prev_basics + basics,
                 "retained_odds": prev_odds + odds_list,
                 "retained_features": prev_feats + feats,
                 "retained_metas": prev_metas + _metas}
    except Exception:
        pass

    if len(packets) == 1:
        pkt = packets[0]
        mode = pkt.get("mode")
        if mode == "match":
            has_fixture = bool(pkt.get("fixture"))
            has_odds = bool(pkt.get("odds"))
            lines = [
                "✅ Match packet ready.",
                f"- Fixture resolved: {'yes' if has_fixture else 'no'}",
                f"- Pre-match odds: {'available' if has_odds else 'not yet posted'}",
                f"- H2H size: {len(pkt['h2h_last']) if pkt.get('h2h_last') else 0}",
                f"- Lineups: {len(pkt['lineups']) if pkt.get('lineups') else 0}",
            ]
        elif mode == "league":
            league_name = (pkt.get("league_info") or {}).get("league", {}).get("name") or pkt.get("query") or "League"
            fixtures = pkt.get("league_fixtures") or []
            lines = [
                f"✅ League fixtures for {league_name}.",
                f"- Fixtures returned: {len(fixtures)}",
            ]
        else:
            team_name = (pkt.get("team_profile") or {}).get("team", {}).get("name") or pkt.get("query") or "Team"
            upc = pkt.get("team_upcoming") or []
            with_odds = sum(1 for fx in upc if fx.get("pre_match_odds"))
            lines = [
                f"✅ Team packet for {team_name} ready.",
                f"- Recent fixtures: {len(pkt.get('team_recent_fixtures') or [])}",
                f"- Upcoming fixtures: {len(upc)} (with pre-match odds on {with_odds})",
                f"- Injuries records: {len(pkt.get('team_injuries') or [])}",
            ]
        return {**state, "final_text": "\n".join(lines)}
    # For multiple packets, return a concise summary and the enriched state
    lines = [f"✅ Packets ready: {len(packets)} (including {sum(1 for p in packets if (p or {}).get('mode')=='match')} matches)"]
    prev = state.get("final_text") or ""
    return {**state, "final_text": (prev + ("\n" if prev else "") + "\n".join(lines))}

def value_node(state: GraphState) -> GraphState:
    """Run value tools: model probabilities and implied/deltas; append results to final_text."""
    try:
        result = value_agent_run(state)
        prev = state.get("final_text") or ""
        # Summarize without dumping JSON
        try:
            implied = result.get("implied_and_deltas") or {}
            rows = implied.get("results") or []
        except Exception:
            rows = []
        header = f"\n\nValue analysis complete for {len(rows)} fixtures.\n"
        # Extract per-match deltas and store in retained_value for CLI consumption
        # Build fixture_id -> names map from retained_matches
        names_map = {}
        for b in (state.get("retained_matches") or []):
            fid = (b or {}).get("fixture_id")
            if fid:
                names_map[fid] = {
                    "home": (b.get("home") if isinstance(b, dict) else None),
                    "away": (b.get("away") if isinstance(b, dict) else None),
                }
        deltas_list: List[Dict[str, Any]] = []
        for r in rows:
            fid = (r or {}).get("fixture_id")
            deltas = (r or {}).get("deltas")
            if fid and isinstance(deltas, dict):
                nm = names_map.get(fid, {})
                deltas_list.append({
                    "fixture_id": fid,
                    "home": nm.get("home"),
                    "away": nm.get("away"),
                    "deltas": {
                        "H": deltas.get("H"),
                        "D": deltas.get("D"),
                        "A": deltas.get("A"),
                    }
                })
        dbg(f"[value_node] rows={len(rows)} with_deltas={len(deltas_list)}")
        return {**state, "final_text": f"{prev}{header}".rstrip(), "retained_value": deltas_list, "retained_value_rows": rows}
    except Exception as e:
        prev = state.get("final_text") or ""
        return {**state, "final_text": f"{prev}\n\n[WARN] Value agent failed: {e}"}

# ------------------ COMPOSE (Claude Sonnet 4) ------------------

def _best_1x2_odds_map(odds_obj: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return best decimal odds per outcome across all bookmakers for a fixture odds list.

    Returns: { 'H': {'odd': float, 'bookmaker': str}, 'D': {...}, 'A': {...} }
    Missing outcomes omitted.
    """
    best = {"H": {"odd": None, "bookmaker": None},
            "D": {"odd": None, "bookmaker": None},
            "A": {"odd": None, "bookmaker": None}}
    for bk in (odds_obj or []):
        markets = (bk or {}).get("markets") or {}
        trio = markets.get("1X2") or {}
        h, d, a = trio.get("home"), trio.get("draw"), trio.get("away")
        name = (bk or {}).get("bookmaker") or (bk or {}).get("name")
        try:
            if h:
                val = float(h)
                if best["H"]["odd"] is None or val > best["H"]["odd"]:
                    best["H"] = {"odd": val, "bookmaker": name}
        except Exception:
            pass
        try:
            if d:
                val = float(d)
                if best["D"]["odd"] is None or val > best["D"]["odd"]:
                    best["D"] = {"odd": val, "bookmaker": name}
        except Exception:
            pass
        try:
            if a:
                val = float(a)
                if best["A"]["odd"] is None or val > best["A"]["odd"]:
                    best["A"] = {"odd": val, "bookmaker": name}
        except Exception:
            pass
    return best

def _parse_target_odds_from_prompt(prompt: str) -> Optional[float]:
    """Parse a target odds multiplier from the user's prompt, e.g., 'at least 10x', 'target 8x', 'odds 6'.

    Heuristics; returns None when not found.
    """
    low = (prompt or "").lower()
    # Explicit cue words with x
    m = re.search(r"(target|min(?:imum)?|at\s+least|over|>=|reach|hit)\s*(\d+(?:\.\d+)?)\s*x", low)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            return None
    # 'odds 10' or 'price 10'
    m = re.search(r"\b(odds|price|return)\s*(\d+(?:\.\d+)?)\b", low)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            return None
    # Standalone '10x' - avoid confusing with '4x acca' legs; prefer values >= 5x as targets
    m_all = re.findall(r"\b(\d+(?:\.\d+)?)x\b", low)
    if m_all:
        try:
            nums = [float(x) for x in m_all]
            # choose any >= 5.0 as likely target
            for v in nums:
                if v >= 5.0:
                    return v
        except Exception:
            return None
    return None

def compose_node(state: GraphState) -> GraphState:
    """Use highest deltas and best available odds to answer:
    - best_value_markets: return single best market (max delta)
    - xn_bet_request: build acca by adding top-delta markets until product(odds) >= target_odds (if provided),
      otherwise pick top XN legs.
    The textual rendering is produced by Claude Sonnet 4.
    """
    try:
        intent = state.get("intent") or state.get("retained_intent")
        xn = state.get("xn") or state.get("retained_xn")
        rows = state.get("retained_value_rows") or []
        basics = state.get("retained_matches") or []
        odds_entries = state.get("retained_odds") or []
        # Build maps for easy lookup
        name_by_fid = {}
        for b in basics:
            fid = (b or {}).get("fixture_id")
            if fid:
                name_by_fid[fid] = {"home": b.get("home"), "away": b.get("away")}
        odds_by_fid = {o.get("fixture_id"): o.get("odds") for o in odds_entries if o.get("fixture_id")}
        # Candidates list
        candidates = []
        for r in rows:
            fid = (r or {}).get("fixture_id")
            deltas = (r or {}).get("deltas") or {}
            # treat deltas as missing if empty or all None
            deltas_all_none = (isinstance(deltas, dict) and deltas and all(v is None for v in deltas.values()))
            if not fid or not isinstance(deltas, dict) or deltas_all_none:
                # If deltas missing but implied present, derive a pseudo-delta vs uniform baseline (1/3)
                implied = (r or {}).get("implied") or None
                if implied and isinstance(implied, dict):
                    try:
                        baseline = 1.0/3.0
                        # choose the outcome with max (implied - baseline)
                        pick = max([(k, (implied.get(k) or 0.0) - baseline) for k in ("H","D","A")], key=lambda kv: kv[1])
                        outcome, delta_val = pick
                        best_map = _best_1x2_odds_map((state.get("retained_odds_map", {}) or {}).get(fid) or odds_by_fid.get(fid) or [])
                        info = best_map.get(outcome)
                        odd = (info or {}).get("odd")
                        book = (info or {}).get("bookmaker")
                        nm = name_by_fid.get(fid, {})
                        candidates.append({
                            "fixture_id": fid,
                            "home": nm.get("home"),
                            "away": nm.get("away"),
                            "pick": outcome,
                            "delta": float(delta_val),
                            "odd": float(odd) if isinstance(odd, (int,float)) else (float(odd) if isinstance(odd, str) and odd else None),
                            "bookmaker": book,
                        })
                    except Exception:
                        pass
                continue
            # pick outcome with maximum delta
            pick = max([(k, v) for k, v in deltas.items() if v is not None], key=lambda kv: kv[1], default=(None, None))
            outcome, delta = pick
            if outcome is None or delta is None:
                continue
            # Get best odds for this outcome
            best_map = _best_1x2_odds_map(odds_by_fid.get(fid) or [])
            info = best_map.get(outcome)
            odd = (info or {}).get("odd")
            book = (info or {}).get("bookmaker")
            nm = name_by_fid.get(fid, {})
            candidates.append({
                "fixture_id": fid,
                "home": nm.get("home"),
                "away": nm.get("away"),
                "pick": outcome,  # 'H' | 'D' | 'A'
                "delta": float(delta),
                "odd": float(odd) if isinstance(odd, (int, float)) else (float(odd) if isinstance(odd, str) and odd else None),
                "bookmaker": book,
            })
        # Sort by delta descending
        candidates.sort(key=lambda x: (x.get("delta") if x.get("delta") is not None else -999), reverse=True)
        selection = {"mode": intent, "items": [], "product_odds": None, "achieved": None}
        if intent == "best_value_markets":
            best = next((c for c in candidates if (c.get("delta") is not None and (c.get("odd") or 0) > 0)), None)
            if best is None and candidates:
                best = candidates[0]
            if best:
                selection["items"] = [best]
                selection["product_odds"] = best.get("odd")
                selection["achieved"] = True
        elif intent == "xn_bet_request":
            target_odds = _parse_target_odds_from_prompt(state.get("user_prompt") or "")
            acca = []
            product = 1.0
            for c in candidates:
                if c.get("odd") is None or c.get("odd") <= 1.0:
                    continue
                acca.append(c)
                product *= float(c.get("odd"))
                if target_odds and product >= target_odds:
                    break
                if xn and len(acca) >= int(xn):
                    # Respect requested leg count; may stop before reaching target
                    break
            selection["items"] = acca
            selection["product_odds"] = product if acca else None
            selection["achieved"] = (target_odds is None) or (product is not None and product >= target_odds)
            if target_odds:
                selection["target_odds"] = target_odds
            if xn:
                selection["xn"] = int(xn)
        else:
            # Unknown intent: do nothing
            return state

        # Render with Claude Sonnet 4
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            SYSTEM = (
                "You format concise betting recommendations strictly for 1X2 selections. "
                "When mode is 'best_value_markets', output a terse one-liner naming the pick with delta and best odds. "
                "When mode is 'xn_bet_request', list legs as short bullets (Home/Draw/Away) with delta and odds, then show the product. "
                "Avoid any extra prose, disclaimers, or emojis."
            )
            content_obj = {
                "mode": selection.get("mode"),
                "product_odds": selection.get("product_odds"),
                "achieved": selection.get("achieved"),
                "target_odds": selection.get("target_odds"),
                "xn": selection.get("xn"),
                "items": selection.get("items"),
            }
            msg = client.messages.create(
                model=SONNET4_MODEL,
                max_tokens=300,
                temperature=0,
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(content_obj)}],
            )
            text = "".join([b.text for b in msg.content if hasattr(b, "text")]) if msg.content else ""
        except Exception:
            # Fallback simple rendering
            if selection.get("mode") == "best_value_markets" and selection.get("items"):
                it = selection["items"][0]
                text = f"Best market: {it.get('home')} vs {it.get('away')} — pick {it.get('pick')} (Δ={it.get('delta'):.3f}), best odds {it.get('odd')} at {it.get('bookmaker')}"
            else:
                lines = ["Acca:"]
                for it in selection.get("items") or []:
                    lines.append(f"- {it.get('home')} vs {it.get('away')}: {it.get('pick')} (Δ={it.get('delta'):.3f}) @ {it.get('odd')} {it.get('bookmaker') or ''}")
                po = selection.get("product_odds")
                tgt = selection.get("target_odds")
                lines.append(f"Product odds: {po}{' (target '+str(tgt)+')' if tgt else ''}")
                text = "\n".join(lines)

        prev = state.get("final_text") or ""
        return {**state, "final_text": f"{prev}\n\n{text}", "retained_selection": selection}
    except Exception:
        return state
    else:
        # Batch summary
        counts = {"match": 0, "team": 0, "league": 0}
        for p in packets:
            m = (p or {}).get("mode")
            if m in counts:
                counts[m] += 1
        lines = [
            "✅ Batch completed.",
            f"- Total packets: {len(packets)}",
            f"- Matches: {counts['match']}, Teams: {counts['team']}, Leagues: {counts['league']}",
        ]
        lines.append("\nRaw JSON packets follow below.")
        pretty = json.dumps(packets, indent=2)
        return {**state, "final_text": "\n".join(lines) + "\n" + pretty}

# ------------------ BUILD GRAPH ------------------

# Descriptive node names
NODE_INTENT_CLASSIFIER = "intent_classifier_sonnet4"
NODE_ENTITY_EXTRACTOR = "entity_extractor_haiku"  # optional; not on main path
NODE_SEARCH_TOOLS = "search_tools"
NODE_ASSEMBLE_ANSWER = "assemble_answer"
NODE_VALUE_ANALYSIS = "value_analysis"
NODE_COMPOSE_SELECTION = "compose_selection"
NODE_VERBALIZE_OUTPUT = "verbalize_output"

graph = StateGraph(GraphState)
graph.add_node(NODE_INTENT_CLASSIFIER, sonnet_entry_node)
graph.add_node(NODE_ENTITY_EXTRACTOR, agent_node)
graph.add_node(NODE_SEARCH_TOOLS, tool_node)
graph.add_node(NODE_ASSEMBLE_ANSWER, answer_node)
graph.add_node(NODE_VALUE_ANALYSIS, value_node)
graph.add_node(NODE_COMPOSE_SELECTION, compose_node)
def verbalize_output(state: GraphState) -> GraphState:
    """Generate a concise human-readable summary using Claude Sonnet 4.

    Prefers retained_selection (acca or best pick). Falls back to retained_value rows.
    Stores text in retained_text and final_text.
    """
    try:
        selection = state.get("retained_selection") or {}
        intent = state.get("intent") or state.get("retained_intent")
        xn = state.get("xn") or state.get("retained_xn")
        basics = state.get("retained_matches") or []
        value_rows = state.get("retained_value") or []
        # Build a compact content object to drive LLM formatting
        content_obj: Dict[str, Any] = {}
        if selection and isinstance(selection, dict) and (selection.get("items") or []):
            content_obj = {
                "mode": selection.get("mode"),
                "xn": selection.get("xn") or xn,
                "target_odds": selection.get("target_odds"),
                "product_odds": selection.get("product_odds"),
                "achieved": selection.get("achieved"),
                "items": selection.get("items"),
            }
        else:
            # Fallback: top value rows with deltas only
            top = []
            for r in (value_rows or [])[:5]:
                top.append(r)
            content_obj = {"mode": intent or "best_value_markets", "items": top}
        # Try Sonnet rendering
        text = None
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            SYSTEM = (
                "You turn model output into crisp, human-friendly betting notes. "
                "If content contains 'items' with home/away/pick/delta/odd, produce: "
                "- One short intro line stating the task (best value or xN acca and whether target reached). "
                "- 1-5 bullet lines: '<Home> vs <Away> — <pick> (Δ=+0.123) @ <odd> <bookmaker>'. "
                "- One closing line summarizing combined/product odds when present. "
                "No emojis, no hedging, no extra caveats. Keep it concise."
            )
            msg = client.messages.create(
                model=SONNET4_MODEL,
                max_tokens=300,
                temperature=0,
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(content_obj)}],
            )
            text = "".join([b.text for b in msg.content if hasattr(b, "text")]) if msg.content else ""
        except Exception:
            pass
        # Deterministic fallback
        if not text:
            mode = content_obj.get("mode")
            items = content_obj.get("items") or []
            lines = []
            if mode == "xn_bet_request":
                nlegs = len(items)
                po = selection.get("product_odds") if selection else None
                tgt = selection.get("target_odds") if selection else None
                lines.append(f"x{xn or nlegs} acca suggestions:")
                for it in items:
                    lines.append(
                        f"- {it.get('home')} vs {it.get('away')} — {it.get('pick')} (Δ={it.get('delta'):.3f}) @ {it.get('odd')} {it.get('bookmaker') or ''}".strip()
                    )
                if po:
                    lines.append(f"Combined odds: {po}{' (target ' + str(tgt) + ')' if tgt else ''}")
            else:
                # best value markets list (top 3)
                lines.append("Top value 1X2 edges:")
                for it in items[:3]:
                    # retained_value entries carry 'deltas' map; format best available
                    ds = (it or {}).get("deltas") or {}
                    pick = max([(k, v) for k, v in ds.items() if v is not None], key=lambda kv: kv[1], default=(None, None))[0]
                    lines.append(f"- {it.get('home')} vs {it.get('away')} — {pick} (Δ={ds.get(pick):.3f} if pick else 'n/a')")
            text = "\n".join(lines)
        prev = state.get("final_text") or ""
        combined = (prev + ("\n\n" if prev and text else "") + (text or "")).strip()
        return {**state, "final_text": combined, "retained_text": combined}
    except Exception as e:
        return state

graph.add_node(NODE_VERBALIZE_OUTPUT, verbalize_output)

graph.set_entry_point(NODE_INTENT_CLASSIFIER)
graph.add_edge(NODE_INTENT_CLASSIFIER, NODE_SEARCH_TOOLS)
graph.add_edge(NODE_ENTITY_EXTRACTOR, NODE_SEARCH_TOOLS)
graph.add_edge(NODE_SEARCH_TOOLS, NODE_ASSEMBLE_ANSWER)
graph.add_edge(NODE_ASSEMBLE_ANSWER, NODE_VALUE_ANALYSIS)
graph.add_edge(NODE_VALUE_ANALYSIS, NODE_COMPOSE_SELECTION)
graph.add_edge(NODE_COMPOSE_SELECTION, NODE_VERBALIZE_OUTPUT)
graph.add_edge(NODE_VERBALIZE_OUTPUT, END)

# Compile with in-memory checkpointer for retention when available
if MemorySaver is not None:
    try:
        _memory = MemorySaver()
        app = graph.compile(checkpointer=_memory)
    except Exception:
        app = graph.compile()
else:
    app = graph.compile()

# ------------------ CLI ------------------

if __name__ == "__main__":
    def write_graph_diagram(gobj=None) -> str:
        """Export the LangGraph diagram.

        Preference order:
        1) Use compiled app's graph if available
        2) Fall back to builder graph
        Try PNG first; if unavailable, always write a Mermaid .mmd and return that path.
        Returns the actual path written, or empty string on failure.
        """
        ts_graph = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_png = f"graph-{ts_graph}.png"
        out_mmd = f"graph-{ts_graph}.mmd"
        # Choose a source object that has get_graph()
        src = None
        try:
            if gobj and hasattr(gobj, "get_graph"):
                src = gobj
            elif 'app' in globals() and hasattr(app, "get_graph"):
                src = app
            elif 'graph' in globals() and hasattr(graph, "get_graph"):
                src = graph
        except Exception:
            src = None
        if not src:
            return ""
        try:
            gviz = src.get_graph()
        except Exception:
            gviz = None
        if not gviz:
            return ""
        # Try PNG outputs first
        try:
            for meth in ("draw_png", "write_png", "draw"):
                if hasattr(gviz, meth):
                    try:
                        getattr(gviz, meth)(out_png)
                        return out_png
                    except Exception:
                        continue
        except Exception:
            pass
        # Mermaid fallback (always attempt)
        try:
            for meth in ("draw_mermaid", "to_mermaid"):
                if hasattr(gviz, meth):
                    try:
                        mer = getattr(gviz, meth)()
                        with open(out_mmd, "w", encoding="utf-8") as mf:
                            mf.write(mer if isinstance(mer, str) else str(mer))
                        return out_mmd
                    except Exception:
                        continue
        except Exception:
            pass
        return ""
    if any(a in ("--draw-graph", "--graph", "--diagram") for a in (sys.argv[1:] or [])):
        path = write_graph_diagram(app)
        if path:
            print(f"Graph exported to {path}")
        else:
            print("Graph exported (Mermaid or PNG)")
        sys.exit(0)
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your question about a team or a match\" [--graph]")
        sys.exit(2)
    prompt = sys.argv[1]
    thread_id = None
    # Optional thread id for retention across runs: take from argv[2] or env THREAD_ID
    if len(sys.argv) >= 3:
        thread_id = sys.argv[2]
    else:
        thread_id = os.getenv("THREAD_ID") or f"cli-{dt.datetime.now().strftime('%Y%m%d')}"
    init: GraphState = {
        "user_prompt": prompt,
        "queries": [],
        "tool_json": None,
        "final_text": None,
        "retained_text": None,
    }
    # Optionally export the graph diagram for documentation (always try once per run)
    write_graph_diagram(app)

    # Invoke with thread_id to enable MemorySaver retention
    try:
        out = app.invoke(init, config={"configurable": {"thread_id": thread_id}})
    except TypeError:
        out = app.invoke(init)
    # Output mode: human (default) or json
    out_mode = os.getenv("OUTPUT_MODE", "human").lower()
    if out_mode == "json":
        # Preserve previous JSON behavior
        intent_out = out.get("intent") or out.get("retained_intent")
        if intent_out == "xn_bet_request" and out.get("retained_selection"):
            payload = out.get("retained_selection")
        else:
            payload = out.get("retained_value") or []
        try:
            print(json.dumps(payload, ensure_ascii=False))
        except Exception:
            print("[]")
    else:
        # Human-readable text
        text = out.get("retained_text") or out.get("final_text") or ""
        print(text)
    # Save only resulting JSON to a file, plus derived features and meta
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_json = out.get("tool_json")
    json_fname = f"football-data-{ts}.json"
    feats_fname = f"football-features-{ts}.json"
    meta_fname = f"football-meta-{ts}.json"
    try:
        # Prefer saving the final packets array (like the text file previously showed)
        if raw_json:
            try:
                obj = json.loads(raw_json)
            except Exception:
                to_save = {"ok": False, "error": "Invalid tool_json payload.", "raw": raw_json}
            else:
                # If tool output has packets, save just that array; otherwise save the whole object
                if isinstance(obj, dict) and "packets" in obj:
                    to_save = obj["packets"]
                else:
                    to_save = obj
        else:
            to_save = {"ok": False, "error": "No tool output available."}
        with open(json_fname, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)

        # If packets array saved, attempt to export features + meta
        packets = to_save if isinstance(to_save, list) else (to_save.get("packets") if isinstance(to_save, dict) else None)
        if isinstance(packets, list):
            try:
                feats, metas = export_features_from_packets(packets)
            except Exception as ex:
                feats, metas = [], []
                print(f"[WARN] Failed to export features/meta: {ex}")
            else:
                with open(feats_fname, "w", encoding="utf-8") as ff:
                    json.dump(feats, ff, indent=2, ensure_ascii=False)
                with open(meta_fname, "w", encoding="utf-8") as mf:
                    json.dump(metas, mf, indent=2, ensure_ascii=False)
    except Exception as e:
        # Suppress any extra prints to adhere to output contract
        pass
