"""Final Project 2: Bayer Leverkusen's unbeaten Bundesliga season.

Run: python -m streamlit run final_project_part2.py
All football data comes from StatsBomb Open Data (competition 9, season 281).
Built on the supplied template's Streamlit / pandas / matplotlib / mplsoccer stack.
"""

from pathlib import Path
from io import BytesIO
import gzip
import json

import matplotlib
matplotlib.use("Agg")  # Render safely in Streamlit/server threads without a desktop GUI.
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import requests
import streamlit as st
from mplsoccer import Pitch


BG_COLOR = "#0d0d0d"
LINE_COLOR = "#FFFFFF"
MAIN_COLOR = "#7C3AED"
ACCENT_COLOR = "#A78BFA"
TEXT_COLOR = "#FFFFFF"
GRAY_COLOR = "#A0A0A0"
TEAM = "Bayer Leverkusen"
COMPETITION_ID, SEASON_ID = 9, 281
ROOT = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "leverkusen_violet", [BG_COLOR, "#352050", MAIN_COLOR, "#DDD0FF"]
)
plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR, "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR, "xtick.color": GRAY_COLOR,
    "ytick.color": GRAY_COLOR, "axes.edgecolor": "#39333F",
    "font.family": "DejaVu Sans", "font.size": 10,
})

CSS = """
<style>
.stApp { background: #0d0d0d; color: #fff; }
[data-testid="stSidebar"] { background: #111111; border-right: 1px solid #7C3AED; }
[data-testid="stHeader"] { background: #0d0d0d; }
h1 { color: #fff !important; font-weight: 800; letter-spacing: -1.5px; }
h2, h3 { color: #A78BFA !important; }
[data-testid="stMetric"] { background: #18151D; border: 1px solid #3D285B;
    border-radius: 12px; padding: 15px; }
[data-testid="stMetricLabel"] { color: #C9B7EA; }
[data-testid="stMetricValue"] { color: #fff; }
.stButton > button, .stDownloadButton > button { border: 1px solid #7C3AED; }
.stButton > button:hover, .stDownloadButton > button:hover { color: #A78BFA; }
.stDataFrame { border: 1px solid #3D285B; border-radius: 8px; }
.eyebrow { color: #A78BFA; font-size: 12px; font-weight: 700;
    letter-spacing: 3px; margin-top: 10px; }
.hero-copy { color: #B7B0C1; max-width: 780px; font-size: 17px; }
hr { border-color: #7C3AED; opacity: .3; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
"""


def read_source(relative_url, local_path):
    """Read an unchanged bundled StatsBomb snapshot, or fetch that exact source."""
    if local_path.exists():
        with gzip.open(local_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    response = requests.get(f"{DATA_URL}/{relative_url}", timeout=(10, 45))
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def load_matches():
    raw = read_source(f"matches/{COMPETITION_ID}/{SEASON_ID}.json",
                      ROOT / "data/matches.json.gz")
    rows = []
    for match in raw:
        home = match["home_team"]["home_team_name"]
        away = match["away_team"]["away_team_name"]
        if TEAM not in (home, away):
            continue
        is_home = home == TEAM
        gf = match["home_score"] if is_home else match["away_score"]
        ga = match["away_score"] if is_home else match["home_score"]
        rows.append({
            "match_id": match["match_id"], "match_week": match["match_week"],
            "date": match["match_date"], "home": home, "away": away,
            "home_score": match["home_score"], "away_score": match["away_score"],
            "opponent": away if is_home else home,
            "venue": "Home" if is_home else "Away", "goals_for": gf,
            "goals_against": ga, "result": "W" if gf > ga else "D" if gf == ga else "L",
            "points": 3 if gf > ga else 1 if gf == ga else 0,
        })
    result = pd.DataFrame(rows).sort_values(["match_week", "date"]).reset_index(drop=True)
    result["cumulative_points"] = result["points"].cumsum()
    result["label"] = result.apply(
        lambda r: f"MD {r.match_week:02d} · {r.home} {r.home_score}–{r.away_score} {r.away} · {r.date}", axis=1)
    return result


def normalize_events(raw):
    """Select named StatsBomb fields; keep missing outcomes missing (completed passes)."""
    rows = []
    for event in raw:
        loc = event.get("location", [np.nan, np.nan])
        passing = event.get("pass", {})
        shot = event.get("shot", {})
        end = passing.get("end_location", [np.nan, np.nan])
        rows.append({
            "id": event["id"], "index": event["index"],
            "period": event["period"], "minute": event["minute"],
            "second": event["second"], "timestamp": event["timestamp"],
            "type_name": event["type"]["name"],
            "team_name": event.get("team", {}).get("name"),
            "player_id": event.get("player", {}).get("id"),
            "player_name": event.get("player", {}).get("name"),
            "x": loc[0], "y": loc[1], "end_x": end[0], "end_y": end[1],
            "recipient_id": passing.get("recipient", {}).get("id"),
            "recipient_name": passing.get("recipient", {}).get("name"),
            "pass_outcome": passing.get("outcome", {}).get("name"),
            "pass_type": passing.get("type", {}).get("name"),
            "duel_type": event.get("duel", {}).get("type", {}).get("name"),
            "shot_outcome": shot.get("outcome", {}).get("name"),
            "shot_xg": shot.get("statsbomb_xg", 0.0),
            "shot_body_part": shot.get("body_part", {}).get("name"),
            "shot_type": shot.get("type", {}).get("name"),
        })
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=40)
def load_events(match_id):
    raw = read_source(f"events/{int(match_id)}.json",
                      ROOT / "data/events" / f"{int(match_id)}.json.gz")
    return normalize_events(raw)


def filter_events(events, team, period, minute_range):
    mask = events["team_name"].eq(team) & events["minute"].between(*minute_range)
    if period != "Full match":
        mask &= events["period"].eq(1 if period == "First half" else 2)
    return events.loc[mask].copy()


def team_metrics(events):
    passes = events[events.type_name.eq("Pass")]
    shots = events[events.type_name.eq("Shot")]
    return {
        "passes": len(passes), "completion": 100 * passes.pass_outcome.isna().mean() if len(passes) else None,
        "shots": len(shots), "xg": shots.shot_xg.sum(),
        "pressures": int(events.type_name.eq("Pressure").sum()),
        "carries": int(events.type_name.eq("Carry").sum()),
    }


def calculate_ppda(events, teams, period, minute_range):
    """StatsBomb's team_stats FAQ formula, calculated from open event data.

    Event coordinates are relative to the event team's attacking direction:
    opponent completed-pass starts x < 72; defending team's actions x >= 48.
    Each qualifying event is counted once, with no player/substitution filter.
    """
    rows = []
    for team in teams:
        own = filter_events(events, team, period, minute_range)
        opponent_names = [name for name in events.team_name.dropna().unique() if name != team]
        opponents = pd.concat(
            [filter_events(events, name, period, minute_range) for name in opponent_names],
            ignore_index=True,
        ) if opponent_names else events.iloc[:0]
        passes = opponents[
            opponents.type_name.eq("Pass") & opponents.pass_outcome.isna() & opponents.x.lt(72)
        ]
        zone = own[own.x.ge(48)]
        tackles = zone.type_name.eq("Duel") & zone.duel_type.eq("Tackle")
        interceptions = zone.type_name.eq("Interception") | (
            zone.type_name.eq("Pass") & zone.pass_type.eq("Interception"))
        fouls = zone.type_name.eq("Foul Committed")
        dribbled_past = zone.type_name.eq("Dribbled Past")
        actions = int((tackles | interceptions | fouls | dribbled_past).sum())
        rows.append({
            "Team": team, "Opponent completed passes": len(passes),
            "Tackles": int(tackles.sum()), "Interceptions": int(interceptions.sum()),
            "Fouls committed": int(fouls.sum()), "Dribbled past": int(dribbled_past.sum()),
            "Defensive actions": actions,
            "PPDA": len(passes) / actions if actions else np.nan,
        })
    return pd.DataFrame(rows)


def ppda_chart(summary):
    fig, ax = plt.subplots(figsize=(12, 4))
    values = summary["PPDA"].fillna(0)
    bars = ax.barh(summary["Team"], values, color=MAIN_COLOR, height=.45)
    limit = max(1, values.max())
    for bar, value in zip(bars, summary["PPDA"]):
        label = "N/A · no defensive actions" if pd.isna(value) else f"{value:.2f}"
        ax.text(bar.get_width() + limit * .025, bar.get_y() + bar.get_height() / 2,
                label, va="center", color=TEXT_COLOR)
    ax.set_xlim(0, limit * 1.55)
    ax.invert_yaxis()
    ax.set_xlabel("Opponent completed passes per defensive action · lower suggests more intense pressing")
    ax.set_title("PPDA · selected match window", loc="left", color=TEXT_COLOR, weight="bold", pad=20)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#39333F", alpha=.5)
    fig.tight_layout()
    return fig


def ppda_page(events, teams, period, minute_range, match_id):
    chosen = st.sidebar.multiselect("PPDA teams", teams, default=teams, key=f"ppda_teams_{match_id}")
    st.markdown("### PPDA · pressing intensity")
    st.caption(f"{period} · minutes {minute_range[0]}–{minute_range[1]} inclusive · team totals for this match only")
    st.markdown("**PPDA = opponent completed passes ÷ your team's defensive actions**, in the opponent’s defensive 60% of the pitch.")
    st.caption("Lower PPDA generally suggests more intense pressing; it does not measure pressing success. N/A means no qualifying defensive actions.")
    if not chosen:
        st.info("Select at least one team in the PPDA teams filter.")
        return
    summary = calculate_ppda(events, chosen, period, minute_range)
    for col, row in zip(st.columns(len(chosen)), summary.to_dict("records")):
        col.metric(f"{row['Team']} · PPDA", "N/A" if pd.isna(row["PPDA"]) else f"{row['PPDA']:.2f}")
        col.caption(f"{row['Opponent completed passes']} opponent passes / {row['Defensive actions']} defensive actions")
    show_figure(ppda_chart(summary), f"ppda_{match_id}.png")
    st.dataframe(summary, hide_index=True, use_container_width=True,
                 column_config={"PPDA": st.column_config.NumberColumn("PPDA", format="%.2f")})
    export = summary.assign(Period=period, **{"Minute from": minute_range[0], "Minute to": minute_range[1], "Match ID": match_id})
    st.download_button("Download PPDA · CSV", export.to_csv(index=False, na_rep="N/A").encode("utf-8-sig"),
                       f"ppda_{match_id}.csv", "text/csv")
    with st.expander("PPDA calculation details"):
        st.markdown("""
        Calculated from **StatsBomb Open Data**, following the explicit
        **team_stats FAQ formula** in [StatsBomb's PPDA glossary](https://hudl.my.site.com/support/s/article/passes-defensive-action?language=en_US&topic=Statsbomb_Global_Football_Data_Glossary).

        - Numerator: opponent **completed Pass** events starting at **x < 72**.
        - Denominator: your team's **Duel / Tackle**, **Interception** (including
          **Pass / Interception**), **Foul Committed**, and **Dribbled Past** events at **x ≥ 48**.
        - These coordinates use each event team's attacking direction on a 120 × 80 pitch.
        - Pressures, blocks, aerial duels and ball recoveries are excluded from this
          FAQ-based denominator. Each qualifying event is counted once.
        - The same period and inclusive minute filters apply to both teams' events.
          All players are included; the passing-network substitution cutoff does not apply.

        These are app-calculated values, not downloaded IQ statistics. The glossary's
        prose also mentions blocks, whereas its explicit FAQ formula omits them;
        this app consistently follows the latter. Definitions can differ by provider.
        """)


def pitch_canvas(title, subtitle):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG_COLOR,
                  line_color=LINE_COLOR, linewidth=1, line_zorder=3)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.subplots_adjust(top=.84, bottom=.12)
    fig.text(.06, .95, title, fontsize=19, weight="bold")
    fig.text(.06, .90, subtitle, fontsize=10, color=GRAY_COLOR)
    fig.text(.06, .035, "ATTACKING DIRECTION  →     |     StatsBomb Open Data", fontsize=9, color=ACCENT_COLOR)
    return pitch, fig, ax


def empty_pitch(ax, message):
    ax.text(60, 40, message, color=TEXT_COLOR, ha="center", va="center",
            bbox={"facecolor": BG_COLOR, "edgecolor": MAIN_COLOR, "pad": 12}, zorder=10)


def shot_map(events, title):
    shots = events[events.type_name.eq("Shot")].dropna(subset=["x", "y"])
    pitch, fig, ax = pitch_canvas(title, "Marker area reflects StatsBomb xG · stars mark goals")
    for goal, marker, color in [(False, "o", MAIN_COLOR), (True, "*", ACCENT_COLOR)]:
        subset = shots[shots.shot_outcome.eq("Goal").eq(goal)]
        pitch.scatter(subset.x, subset.y, s=70 + subset.shot_xg * 1200,
                      marker=marker, c=color, edgecolors=LINE_COLOR, linewidths=1,
                      alpha=.9, ax=ax, zorder=5)
    if shots.empty:
        empty_pitch(ax, "No shots in this selection")
    handles = [Line2D([], [], marker="o", linestyle="", color=MAIN_COLOR, label="Other shots"),
               Line2D([], [], marker="*", linestyle="", color=ACCENT_COLOR, markersize=12, label="Goals")]
    ax.legend(handles=handles, loc="upper left", facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
    return fig


def heatmap(events, title, subtitle, bins):
    locations = events.dropna(subset=["x", "y"])
    pitch, fig, ax = pitch_canvas(title, subtitle)
    if locations.empty:
        empty_pitch(ax, "No located events in this selection")
        return fig
    stats = pitch.bin_statistic(locations.x, locations.y, statistic="count", bins=bins)
    mesh = pitch.heatmap(stats, ax=ax, cmap=HEAT_CMAP, edgecolors="#28212F", linewidth=.3, zorder=1)
    bar = fig.colorbar(mesh, ax=ax, fraction=.025, pad=.02)
    bar.set_label("Event count")
    return fig


def passing_network_data(events, minimum):
    passes = events[events.type_name.eq("Pass") & events.pass_outcome.isna()].dropna(
        subset=["player_id", "recipient_id", "x", "y", "end_x", "end_y"]).copy()
    if passes.empty:
        return pd.DataFrame(), pd.DataFrame()
    starts = passes[["player_id", "player_name", "x", "y"]]
    ends = passes[["recipient_id", "recipient_name", "end_x", "end_y"]].rename(columns={
        "recipient_id": "player_id", "recipient_name": "player_name", "end_x": "x", "end_y": "y"})
    nodes = pd.concat([starts, ends]).groupby("player_id").agg(
        player_name=("player_name", "first"), x=("x", "mean"), y=("y", "mean"), involvement=("x", "size"))
    # Combine both directions explicitly: an edge is an undirected passing partnership.
    passes["a"] = passes[["player_id", "recipient_id"]].min(axis=1)
    passes["b"] = passes[["player_id", "recipient_id"]].max(axis=1)
    edges = passes.groupby(["a", "b"]).size().reset_index(name="passes")
    edges = edges[(edges.passes >= minimum) & edges.a.ne(edges.b)]
    return nodes, edges


def passing_network(events, minimum):
    pitch, fig, ax = pitch_canvas("The passing connections", "Completed passes · links combine both directions · average pass/receipt locations")
    nodes, edges = passing_network_data(events, minimum)
    if nodes.empty:
        empty_pitch(ax, "No completed passes with recipients in this selection")
        return fig
    for edge in edges.itertuples():
        a, b = nodes.loc[edge.a], nodes.loc[edge.b]
        pitch.lines(a.x, a.y, b.x, b.y, lw=1 + 9 * edge.passes / edges.passes.max(),
                    color=MAIN_COLOR, alpha=.65, ax=ax, zorder=2)
    pitch.scatter(nodes.x, nodes.y, s=130 + 950 * nodes.involvement / nodes.involvement.max(),
                  c=MAIN_COLOR, edgecolors=LINE_COLOR, ax=ax, zorder=4)
    for node in nodes.itertuples():
        pitch.annotate(node.player_name.split()[-1], (node.x, node.y), xytext=(0, 13),
                       textcoords="offset points", ha="center", color=TEXT_COLOR,
                       fontsize=9, weight="bold", ax=ax, zorder=5)
    if edges.empty:
        fig.text(.5, .075, "No partnerships meet the selected minimum.", ha="center", color=ACCENT_COLOR)
    return fig


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#39333F", alpha=.5)
    ax.set_axisbelow(True)


def points_chart(matches):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(matches.match_week, matches.cumulative_points, color=MAIN_COLOR, lw=3)
    ax.fill_between(matches.match_week, matches.cumulative_points, color=MAIN_COLOR, alpha=.13)
    for result, color in [("W", MAIN_COLOR), ("D", TEXT_COLOR), ("L", GRAY_COLOR)]:
        subset = matches[matches.result.eq(result)]
        if len(subset):
            ax.scatter(subset.match_week, subset.cumulative_points, c=color, s=35, label={"W":"Win", "D":"Draw", "L":"Loss"}[result], zorder=3)
    ax.set(xlabel="Matchweek", ylabel="Cumulative points", xlim=(.5, 34.5), ylim=(0, 105))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 34])
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    return fig


def xg_timeline(events, teams):
    """Use period timestamp + actual first-half duration to avoid stoppage-time overlap."""
    frame = events[events.period.isin([1, 2])].copy()
    seconds = pd.to_timedelta(frame.timestamp).dt.total_seconds()
    first_half = max(45 * 60, seconds[frame.period.eq(1)].max())
    frame["elapsed"] = (seconds + np.where(frame.period.eq(2), first_half, 0)) / 60
    end = frame.elapsed.max()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for team, color in zip(teams, [MAIN_COLOR, TEXT_COLOR]):
        shots = frame[frame.type_name.eq("Shot") & frame.team_name.eq(team)].sort_values("index")
        totals = shots.shot_xg.cumsum()
        ax.step([0, *shots.elapsed, end], [0, *totals, totals.iloc[-1] if len(totals) else 0],
                where="post", color=color, lw=2.5, label=f"{team} · {shots.shot_xg.sum():.2f} xG")
        goals = shots.shot_outcome.eq("Goal")
        ax.scatter(shots.loc[goals, "elapsed"], totals[goals], marker="*", s=160,
                   color=color, edgecolors=ACCENT_COLOR, zorder=4)
    ax.axvline(first_half / 60, color=GRAY_COLOR, ls=":", lw=1)
    ax.set(xlabel="Playing minutes including stoppage time (half-time break excluded)", ylabel="Cumulative StatsBomb xG")
    ax.set_title("The chance-creation story", loc="left", color=TEXT_COLOR, weight="bold", pad=20)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    return fig


def show_figure(fig, filename):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    st.pyplot(fig, use_container_width=True)
    st.download_button("Download chart · PNG", buffer.getvalue(), filename, "image/png", key=filename)
    plt.close(fig)


def season_page(matches):
    st.subheader("A season without defeat")
    counts = matches.result.value_counts()
    columns = st.columns(6)
    for col, label, value in zip(columns,
        ["Matches", "Wins", "Draws", "Defeats", "Points", "Goals for / against"],
        [len(matches), counts.get("W", 0), counts.get("D", 0), counts.get("L", 0),
         int(matches.points.sum()), f"{matches.goals_for.sum()} / {matches.goals_against.sum()}"]):
        col.metric(label, value)
    st.caption("Season totals use final match scores. Match filters below do not change these totals.")
    st.markdown("### The road to 90 points")
    show_figure(points_chart(matches), "leverkusen_points.png")
    st.markdown("### Every league match")
    left, right = st.columns(2)
    venue = left.selectbox("Venue", ["All venues", "Home", "Away"])
    result = right.selectbox("Result", ["All results", "W", "D", "L"])
    selected = matches.copy()
    if venue != "All venues":
        selected = selected[selected.venue.eq(venue)]
    if result != "All results":
        selected = selected[selected.result.eq(result)]
    display = selected[["match_week", "date", "opponent", "venue", "goals_for", "goals_against", "result", "points"]].rename(columns={
        "match_week": "Matchweek", "date": "Date", "opponent": "Opponent", "venue": "Venue",
        "goals_for": "Goals for", "goals_against": "Goals against", "result": "Result", "points": "Points"})
    if display.empty:
        st.info("No matches match these filters — Leverkusen had no league defeats.")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.download_button("Download selected results · CSV", display.to_csv(index=False).encode("utf-8-sig"),
                       "leverkusen_results.csv", "text/csv")


def match_page(matches):
    with st.sidebar:
        st.divider()
        match_id = st.selectbox("Match", matches.match_id.tolist(),
            format_func=dict(zip(matches.match_id, matches.label)).get,
            index=len(matches) - 1)
        match = matches[matches.match_id.eq(match_id)].iloc[0]
        view = st.radio("Visualization", ["Shot Map", "Passing Network", "Pressure Heatmap", "Player Heatmap", "xG Timeline", "PPDA"])
        team = st.selectbox("Team", [TEAM, match.opponent]) if view not in ["PPDA", "xG Timeline"] else TEAM
    with st.spinner("Loading StatsBomb match events…"):
        try:
            events = load_events(match_id)
        except (requests.RequestException, ValueError, OSError) as exc:
            st.error("The StatsBomb match could not be loaded. Restore the bundled data folder or check your internet connection.")
            with st.expander("Error details"):
                st.code(str(exc))
            return
    st.subheader(f"{match.home}  {match.home_score}–{match.away_score}  {match.away}")
    st.caption(f"Matchweek {match.match_week} · {match.date} · Bundesliga 2023/24 · StatsBomb match {match_id}")
    if view == "xG Timeline":
        st.info("Full-match comparison for both teams. Stars mark shot goals; own goals do not carry shot xG.")
        show_figure(xg_timeline(events, [TEAM, match.opponent]), f"xg_{match_id}.png")
        return
    with st.sidebar:
        period = st.selectbox("Period", ["Full match", "First half", "Second half"])
        maximum = int(events.minute.max())
        minute_range = st.slider("Match-minute range (inclusive)", 0, maximum, (0, maximum), key=f"minutes_{match_id}")
        st.caption("Use Period to separate first-half stoppage time from the start of the second half.")
    if view == "PPDA":
        ppda_page(events, [TEAM, match.opponent], period, minute_range, match_id)
        return
    selected = filter_events(events, team, period, minute_range)
    if view == "Passing Network":
        before_sub = st.sidebar.checkbox("Before team's first substitution", value=True)
        minimum = st.sidebar.slider("Minimum passes per partnership", 1, 20, 3)
        if before_sub:
            substitutions = events[events.team_name.eq(team) & events.type_name.eq("Substitution")]
            if not substitutions.empty:
                cutoff = substitutions.iloc[0]
                selected = selected[selected["index"] < cutoff["index"]]
                st.caption(f"Network and metrics stop before the first {team} substitution ({cutoff.minute}′), within your selected period and minutes.")
        st.caption("Node size = completed pass/receipt involvements. Link width = completed passes between the pair. Locations are event averages, not a formation or tracking data.")
    player = None
    if view in ["Shot Map", "Player Heatmap"]:
        players = sorted(events.loc[events.team_name.eq(team), "player_name"].dropna().unique())
        options = (["All players"] if view == "Shot Map" else []) + players
        if not options:
            st.info("No players are available in this match.")
            return
        player = st.sidebar.selectbox("Player", options)
        if player != "All players":
            selected = selected[selected.player_name.eq(player)]
    stats = team_metrics(selected)
    columns = st.columns(6)
    for col, label, value in zip(columns,
        ["Passes", "Pass completion", "Shots", "StatsBomb xG", "Pressures", "Carries"],
        [stats["passes"], f"{stats['completion']:.1f}%" if stats["completion"] is not None else "—",
         stats["shots"], f"{stats['xg']:.2f}", stats["pressures"], stats["carries"]]):
        col.metric(label, value)
    st.caption(f"Metrics: {team} · {player or 'all players'} · {period.lower()} · minutes {minute_range[0]}–{minute_range[1]}.")
    chart_events = selected
    if view == "Shot Map":
        figure = shot_map(selected, f"Where the chances came from · {player if player and player != 'All players' else team}")
        chart_events = selected[selected.type_name.eq("Shot")]
    elif view == "Passing Network":
        figure = passing_network(selected, minimum)
        chart_events = selected[selected.type_name.eq("Pass") & selected.pass_outcome.isna()]
    else:
        resolution = st.sidebar.select_slider("Heatmap detail", ["Coarse", "Medium", "Fine"], value="Medium")
        bins = {"Coarse": (6, 4), "Medium": (12, 8), "Fine": (18, 12)}[resolution]
        if view == "Pressure Heatmap":
            chart_events = selected[selected.type_name.eq("Pressure")]
            figure = heatmap(chart_events, f"Where {team} applied pressure", "Recorded pressure events per pitch bin · raw counts", bins)
        else:
            kinds = st.sidebar.multiselect("Player event types", ["Pass", "Carry", "Ball Receipt*", "Shot", "Pressure", "Ball Recovery", "Dribble"], default=["Pass", "Carry", "Ball Receipt*"])
            chart_events = selected[selected.type_name.isin(kinds)]
            figure = heatmap(chart_events, f"{player} · event locations", "Selected event starts per bin · not tracking, distance covered, or time spent", bins)
    show_figure(figure, f"{view.lower().replace(' ', '_')}_{match_id}.png")
    with st.expander("Explore the plotted events / download CSV"):
        columns = ["period", "minute", "second", "type_name", "player_name", "x", "y"]
        if view == "Shot Map":
            columns += ["shot_xg", "shot_outcome", "shot_body_part", "shot_type"]
        elif view == "Passing Network":
            columns += ["recipient_name", "end_x", "end_y"]
        table = chart_events[columns]
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download events · CSV", table.to_csv(index=False).encode("utf-8-sig"),
                           f"statsbomb_{match_id}_events.csv", "text/csv")


def main():
    # The first Streamlit UI command, as in the supplied template.
    st.set_page_config(page_title="UNBEATEN · Leverkusen 2023/24", page_icon="⚽", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## UNBEATEN / 04")
        st.caption("BAYER LEVERKUSEN · 2023/24")
        page = st.radio("Explore", ["Season Overview", "Match Explorer"])
        st.divider()
        st.caption("Bundesliga · Competition 9 · Season 281")
        st.caption("Coverage: Leverkusen's 34 league matches, including their opponents in those matches.")
    st.markdown('<div class="eyebrow">BUNDESLIGA 2023/24 · FINAL PROJECT 2</div>', unsafe_allow_html=True)
    st.title("The art of staying unbeaten.")
    st.markdown('<p class="hero-copy">Explore Bayer Leverkusen’s extraordinary league season — from the results that built the run to the passes, pressure and chances behind each match.</p>', unsafe_allow_html=True)
    st.divider()
    try:
        matches = load_matches()
    except (requests.RequestException, ValueError, OSError) as exc:
        st.error("Could not load StatsBomb match data. Restore the bundled data folder or check your internet connection.")
        st.code(str(exc))
        st.stop()
    if len(matches) != 34:
        st.warning(f"Only {len(matches)} Leverkusen matches are available; season totals reflect this coverage.")
    if page == "Season Overview":
        season_page(matches)
    else:
        match_page(matches)
    st.divider()
    with st.expander("Data & methodology"):
        st.markdown("""
        **Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data),
        Bundesliga 2023/2024 (competition **9**, season **281**). All football data,
        including results, players, event locations and xG, comes from this source.
        The bundled snapshot was retrieved on **21 November 2026**.

        **Coverage:** 34 Leverkusen Bundesliga matches, not the full league schedule.
        The app does not infer a league table or include cup/European matches.

        **Pass completion:** a Pass event with no pass outcome is complete.
        **xG:** sum of the supplied `shot.statsbomb_xg`, including penalties; not a model trained here.
        Match scores include own goals; own goals are not shot goals and do not add xG.

        **Pitches:** StatsBomb's 120 × 80 coordinate system. Each team's attacking
        actions are shown left to right in both halves. Heatmaps show counts of
        recorded events with locations, not tracking-based possession or movement.
        Player metrics follow the player, period and minute filters; heatmap event-type
        filters only affect the heatmap. Passing networks can include substitutes when
        the first-substitution cutoff is disabled.

        **Timeline:** periods are joined using the recorded first-half duration so
        first-half stoppage time never overlaps the second half. Half-time is excluded.
        This view always uses the full match for both teams.

        **PPDA:** opponent completed passes in their own 60% divided by the selected
        team's tackles, interceptions (including interception passes), fouls committed
        and dribbled-past events in that zone. Both teams use the same time window;
        zero defensive actions produces N/A. See the PPDA view for exact thresholds
        and the StatsBomb FAQ definition used.

        Charts and event tables are downloadable. Data is cached in memory;
        bundled gzip files are read locally, with the same StatsBomb repository used
        as a network fallback if a file is missing. Data terms accompany the app.
        """)
    st.caption("Football Data Analysis Mentorship — Cohort 2 · Sara Bentelli · 2026 · Data: StatsBomb Open Data (Bundesliga 2023/24)")


if __name__ == "__main__":
    main()
