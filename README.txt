UNBEATEN / 04 — Bayer Leverkusen 2023/24
Final Project 2: Streamlit Application

QUICK START
1. Unzip the complete project and open a terminal in leverkusen_app.
2. Use Python 3.11 or newer, then run:

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m streamlit run final_project_part2.py

   Windows: use python instead of python3, and activate with
   .venv\Scripts\activate

3. Open http://localhost:8501 if the browser does not open automatically.

Keep .streamlit/config.toml and data/ beside the Python file. Launch from
this project folder so Streamlit discovers the dark theme configuration.
Internet is needed to install packages. The bundled StatsBomb match and
event data then supports offline exploration. If a data file is absent,
the app downloads that exact file from the StatsBomb Open Data repository.
No API key or paid account is required.

WHAT IS INCLUDED
- Season overview: 34 matches, 28 wins, 6 draws, 0 defeats, 90 points,
  89 goals for and 24 against, calculated from StatsBomb match results.
- Cumulative points chart and filterable results table.
- Match Explorer: every Leverkusen league fixture, either team, periods,
  inclusive minute filters, and player filters where applicable.
- Shot maps with supplied StatsBomb xG and goal markers.
- Passing networks with a first-team-substitution cutoff and minimum
  partnership count. Links combine both directions. Nodes use completed
  pass starts and receipt endpoints, not tracking positions.
- Pressure count heatmaps and player event-location heatmaps.
- Full-match cumulative xG comparison with stoppage time handled in order.
- PPDA comparison: select PPDA in Match Explorer, choose either or both
  teams, and filter by period/minutes. The table shows opponent completed
  passes, tackles, interceptions, fouls, dribbled-past events and PPDA.
  Uses the explicit StatsBomb team_stats FAQ formula: opponent completed
  passes at x < 72 / own defensive actions at x >= 48. Interception passes
  are included; pressures and blocks are excluded. Zero denominator is N/A.
  PPDA is match-specific, not an opponent's complete-season average.
- PNG charts and CSV results/events/PPDA downloads.
- Required dark background, white pitch lines and violet #7C3AED accent.

DATA SCOPE AND ATTRIBUTION
Source: StatsBomb Open Data, https://github.com/statsbomb/open-data
Competition ID 9; season ID 281 (2023/2024).
The dataset covers Leverkusen's 34 matches, not all 306 Bundesliga games.
There is no complete league table, other-team season total, fabricated
event data, other data provider, or inferred tracking/possession metric.
The JSON snapshots are unchanged except for lossless gzip compression.
Retrieved 2026-11-21. Exact source paths are in data/SOURCE.txt.
StatsBomb's logo and original data license are included in data/.
Follow those data terms when sharing or publishing the project.

The original template's Streamlit/pandas/matplotlib/mplsoccer structure
and dark/violet style are retained and extended. Requests reads the raw
StatsBomb JSON with a timeout and local fallback instead of Sbopen so
the complete project can work without an active internet connection.

The app's Data & methodology expander explains filters, completed-pass
handling, xG, own goals, coordinates and heatmap limitations.

STREAMLIT COMMUNITY CLOUD
Upload the entire folder (including .streamlit and data) to your own
repository, select final_project_part2.py as the entrypoint, and use
requirements.txt for dependencies. This project is not deployed for you.

VALIDATION
Verified all 34 event files against match-score goal totals, season
results, passing-network accounting, empty charts, chronological xG,
PPDA counts against raw JSON for all 34 matches and two time windows,
zero-denominator handling, team selection, and interactive Streamlit controls across all six chart types.
Five automated tests passed; the browser preview was checked visually.
Tested with Python 3.14.7, Streamlit 1.67.0, pandas 3.1.1,
matplotlib 3.11.2, mplsoccer 1.8.1, numpy 2.5.5 and requests 2.34.2.

To run the included tests:
   python -m pip install -r requirements-dev.txt
   python -m pytest tests -q
