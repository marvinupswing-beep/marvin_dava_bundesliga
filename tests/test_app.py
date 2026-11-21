import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'final_project_part2.py'
sys.path.insert(0, str(APP.parent))
import final_project_part2 as app


def test_season_and_all_event_files():
    matches = app.load_matches()
    assert len(matches) == matches.match_id.nunique() == 34
    assert matches.result.value_counts().to_dict() == {'W': 28, 'D': 6}
    assert (matches.points.sum(), matches.goals_for.sum(), matches.goals_against.sum()) == (90, 89, 24)
    for match in matches.itertuples():
        events = app.load_events(match.match_id)
        assert set(events.team_name.unique()) == {match.home, match.away}
        assert events.id.is_unique and events['index'].is_monotonic_increasing
        goals = events.shot_outcome.eq('Goal').sum() + events.type_name.eq('Own Goal Against').sum()
        assert goals == match.home_score + match.away_score
        assert events.loc[events.type_name.eq('Shot'), 'shot_xg'].between(0, 1).all()
        first = app.filter_events(events, app.TEAM, 'First half', (0, 110))
        assert first.period.eq(1).all()
        assert first.team_name.eq(app.TEAM).all()


def test_network_and_empty_plots():
    events = app.load_events(int(app.load_matches().iloc[-1].match_id))
    selected = events[events.team_name.eq(app.TEAM)]
    nodes, edges = app.passing_network_data(selected, 1)
    eligible = selected[selected.type_name.eq('Pass') & selected.pass_outcome.isna()].dropna(subset=['player_id','recipient_id','x','y','end_x','end_y'])
    assert nodes.involvement.sum() == 2 * len(eligible)
    assert edges.passes.sum() == len(eligible[eligible.player_id.ne(eligible.recipient_id)])
    empty = selected.iloc[:0]
    for fig in [app.passing_network(empty, 3), app.shot_map(empty, 'Empty'), app.heatmap(empty, 'Empty', 'No data', (12,8)), app.xg_timeline(events, [app.TEAM, 'Augsburg'])]:
        fig.canvas.draw()
        plt.close(fig)
    timeline = app.xg_timeline(events, [app.TEAM, 'Augsburg'])
    for line in timeline.axes[0].lines[:2]:
        assert np.all(np.diff(line.get_xdata()) >= 0)
    plt.close(timeline)


def widget(at, kind, label):
    return next(x for x in getattr(at, kind) if x.label == label)


def test_interactive_app():
    at = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ['34','28','6','0','90','89 / 24']
    widget(at,'selectbox','Result').set_value('L').run()
    assert not at.exception and len(at.info) == 1
    widget(at,'radio','Explore').set_value('Match Explorer').run()
    assert not at.exception
    for view in ['Passing Network','Pressure Heatmap','Player Heatmap','xG Timeline','Shot Map']:
        widget(at,'radio','Visualization').set_value(view).run()
        assert not at.exception, (view, at.exception)
    widget(at,'selectbox','Team').set_value('Augsburg').run()
    widget(at,'selectbox','Period').set_value('Second half').run()
    widget(at,'slider','Match-minute range (inclusive)').set_value((0,0)).run()
    assert not at.exception
    assert at.metric[0].value == '0'
    for view in ['Passing Network','Pressure Heatmap','Player Heatmap']:
        widget(at,'radio','Visualization').set_value(view).run()
        assert not at.exception
    widget(at,'multiselect','Player event types').set_value([]).run()
    assert not at.exception


def test_ppda_matches_raw_statsbomb_events():
    import gzip
    import json
    for match in app.load_matches().itertuples():
        raw = json.load(gzip.open(APP.parent / 'data/events' / f'{match.match_id}.json.gz', 'rt'))
        events = app.load_events(match.match_id)
        for period, bounds in [('Full match', (0, 120)), ('Second half', (60, 75))]:
            summary = app.calculate_ppda(events, [match.home, match.away], period, bounds)
            for row in summary.to_dict('records'):
                window = [e for e in raw if bounds[0] <= e['minute'] <= bounds[1] and (period == 'Full match' or e['period'] == 2)]
                passes = sum(e['team']['name'] != row['Team'] and e['type']['name'] == 'Pass'
                             and 'outcome' not in e.get('pass', {}) and e.get('location', [999])[0] < 72 for e in window)
                actions = sum(e['team']['name'] == row['Team'] and e.get('location', [-1])[0] >= 48 and (
                    e['type']['name'] in ['Interception', 'Foul Committed', 'Dribbled Past'] or
                    (e['type']['name'] == 'Duel' and e.get('duel', {}).get('type', {}).get('name') == 'Tackle') or
                    (e['type']['name'] == 'Pass' and e.get('pass', {}).get('type', {}).get('name') == 'Interception')
                ) for e in window)
                assert row['Opponent completed passes'] == passes
                assert row['Defensive actions'] == actions
                assert actions == sum(row[k] for k in ['Tackles', 'Interceptions', 'Fouls committed', 'Dribbled past'])
                assert np.isclose(row['PPDA'], passes / actions) if actions else np.isnan(row['PPDA'])
    empty = app.calculate_ppda(events, [match.home, match.away], 'Second half', (0, 0))
    assert empty.PPDA.isna().all()
    assert empty['Defensive actions'].eq(0).all()
    fig = app.ppda_chart(empty)
    fig.canvas.draw()
    plt.close(fig)


def test_ppda_controls():
    at = AppTest.from_file(str(APP), default_timeout=45).run()
    widget(at, 'radio', 'Explore').set_value('Match Explorer').run()
    widget(at, 'radio', 'Visualization').set_value('PPDA').run()
    assert not at.exception
    expected = app.calculate_ppda(app.load_events(3895348), [app.TEAM, 'Augsburg'], 'Full match', (0, 93))
    assert [m.value for m in at.metric] == [f'{v:.2f}' for v in expected.PPDA]
    widget(at, 'multiselect', 'PPDA teams').set_value(['Augsburg']).run()
    assert len(at.metric) == 1 and 'Augsburg' in at.metric[0].label
    widget(at, 'selectbox', 'Period').set_value('Second half').run()
    widget(at, 'slider', 'Match-minute range (inclusive)').set_value((0, 0)).run()
    assert at.metric[0].value == 'N/A' and not at.exception
    widget(at, 'multiselect', 'PPDA teams').set_value([]).run()
    assert not at.exception and len(at.info) == 1
