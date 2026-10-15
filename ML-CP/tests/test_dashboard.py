"""Real Streamlit interaction smoke tests against completed experiment outputs."""
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT/'models/upgraded/run_metadata.json').exists(),reason='Run python run_upgrade.py first')


def app():
    return AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=30).run()


def test_overview_filters():
    at=app()
    assert not at.exception
    assert len(at.metric)==4
    at.selectbox(key='position').select('Goalkeeper').run()
    assert not at.exception
    assert at.metric[0].value!='111,706'
    at.selectbox(key='tier').select('Tier 1 - Big 5').run()
    assert not at.exception
    assert len(at.dataframe)>=1
    assert len(at.dataframe[0].value)==5
    assert any('Ordinary least squares' in str(x) for x in at.dataframe[0].value['Model'])
    assert any('Linear SVR' in str(x) for x in at.dataframe[0].value['Model'])


def test_player_search_and_history():
    at=app()
    at.radio[0].set_value('Player explorer').run()
    assert not at.exception
    at.text_input[0].set_value('Haaland').run()
    assert not at.exception
    assert 'Haaland' in at.subheader[0].value
    dates=at.selectbox[1].options
    at.selectbox[1].select(dates[0]).run()
    assert not at.exception
    assert len(at.metric)==3
    at.text_input[0].set_value('no_such_player_12345').run()
    assert not at.exception
    assert any('No matching player' in item.value for item in at.info)


def test_method_and_roadmap():
    at=app()
    for view in ('Method & caveats','48-hour roadmap'):
        at.radio[0].set_value(view).run()
        assert not at.exception
        assert len(at.dataframe)>=1


def test_learning_curves_model_selection():
    at=app()
    at.radio[0].set_value('Learning curves').run()
    assert not at.exception
    assert len(at.metric)==3
    at.selectbox(key='curve_model').set_value('recent_svr').run()
    assert not at.exception
    saved=pd.read_csv(ROOT/'models/upgraded/learning_curves.csv')
    expected=saved.loc[(saved.model=='recent_svr') & (saved.fraction==1.0),'validation_log_rmse'].iloc[0]
    assert at.metric[1].value==f'{expected:.4f}'
    assert len(at.dataframe[0].value)==4
