import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.recent_features import build_recent_features
from src.preprocessing import FootballFeatures,model_pipeline,BASE_NUMERIC,CATEGORICAL
from src.temporal import calendar_folds
from src.upgraded_training import learning_curve_splits, make_svr, fit_checked
from sklearn.linear_model import LinearRegression


def raw_fixture():
    df = pd.DataFrame({name:[1.,2.,3.,4.] for name in BASE_NUMERIC})
    for col in CATEGORICAL:
        df[col] = ['Attack','Attack','Defender','Defender'] if col=='position' else ['A']*4
    df['height_in_cm'] = [180.,np.nan,190.,np.nan]
    df['minutes_last_365d'] = [90.,180.,90.,0.]
    for metric in ('goals','assists','goal_contributions'):
        df[f'{metric}_per_90_365d'] = [1.,0.5,0.,0.]
    for d in (90,180,365):
        for metric in ('minutes','appearances','goals','assists'):
            df[f'observed_{metric}_{d}d'] = [90.,180.,90.,0.] if metric=='minutes' else [1.,2.,1.,0.]
    df['days_since_last_appearance'] = [10.,20.,5.,np.nan]
    df['observation_history_days'] = [100.,200.,300.,np.nan]
    df['has_observed_history'] = [1,1,1,0]
    return df


def test_calendar_folds_keep_entire_dates_together():
    dates = pd.Series(pd.to_datetime(['2019-12-31']*3+['2020-01-01']*4+['2020-12-31']*2+['2021-01-01']*5))
    for train,valid in calendar_folds(dates,(2020,2021)):
        assert dates.iloc[train].max()<dates.iloc[valid].min()
        assert not set(dates.iloc[train]) & set(dates.iloc[valid])
    with pytest.raises(ValueError,match='No training'):
        list(calendar_folds(dates,(2018,)))


def test_rolling_boundaries_and_future_invariance(tmp_path):
    anchor = pd.Timestamp('2024-01-01')
    base = pd.DataFrame({'player_id':[1,2], 'valuation_date':[anchor,anchor]})
    base.to_parquet(tmp_path/'base.parquet')
    offsets = [-366,-365,-180,-90,-1,0,1]
    apps = pd.DataFrame({'appearance_id':[str(i) for i in range(7)],'game_id':range(7),
        'player_id':[1]*7,'date':[anchor+pd.Timedelta(days=d) for d in offsets],
        'minutes_played':[90]*7,'goals':[1]*7,'assists':[0]*7})
    apps.to_csv(tmp_path/'appearances.csv',index=False)
    build_recent_features(tmp_path/'base.parquet',tmp_path/'appearances.csv',tmp_path/'output.parquet')
    result = pd.read_parquet(tmp_path/'output.parquet').set_index('player_id')
    assert result.loc[1,'observed_minutes_90d']==180
    assert result.loc[1,'observed_minutes_180d']==270
    assert result.loc[1,'observed_minutes_365d']==360
    assert result.loc[1,'days_since_last_appearance']==1
    assert result.loc[2,'has_observed_history']==0
    assert pd.isna(result.loc[2,'days_since_last_appearance'])
    # Remove on/after-anchor events: historical features must remain identical.
    apps.loc[apps.date<anchor].to_csv(tmp_path/'appearances.csv',index=False)
    build_recent_features(tmp_path/'base.parquet',tmp_path/'appearances.csv',tmp_path/'past.parquet')
    pd.testing.assert_frame_equal(result,pd.read_parquet(tmp_path/'past.parquet').set_index('player_id'))
    apps = pd.concat([apps.iloc[:2],apps.iloc[:1]],ignore_index=True)
    apps.to_csv(tmp_path/'appearances.csv',index=False)
    with pytest.raises(ValueError,match='duplicate appearance'):
        build_recent_features(tmp_path/'base.parquet',tmp_path/'appearances.csv',tmp_path/'bad.parquet')


def test_fold_local_statistics_unknown_categories_and_reload(tmp_path):
    train = raw_fixture()
    model = model_pipeline(Ridge(alpha=1),recent=True,scale=True)
    model.fit(train,np.log([100,200,300,400]))
    assert model.named_steps['features'].height_medians_['Attack']==180
    test = train.iloc[:1].copy()
    test['height_in_cm'] = np.nan
    test['foot'] = 'UNSEEN_FOOT'
    before = model.named_steps['features'].height_medians_.copy()
    prediction = model.predict(test)
    assert np.isfinite(prediction).all()
    assert 'UNSEEN_FOOT' not in set(model.named_steps['columns'].named_transformers_['category'].categories_[-1])
    pd.testing.assert_series_equal(before,model.named_steps['features'].height_medians_)
    joblib.dump(model,tmp_path/'model.joblib')
    np.testing.assert_allclose(prediction,joblib.load(tmp_path/'model.joblib').predict(test))


def test_shrinkage_and_rate_trends_use_training_priors():
    frame = raw_fixture()
    transformer = FootballFeatures().fit(frame)
    out = transformer.transform(frame)
    prior = transformer.rate_priors_['goals']['Attack']
    expected = (90*frame.loc[0,'observed_goals_90d']+450*prior)/(90+450)
    assert out.loc[0,'goals_per90_shrunk_90d']==pytest.approx(expected)
    assert out.loc[0,'minutes_trend_90d']==90
    assert out.loc[3,'has_recent_minutes']==0
    assert np.isfinite(out.loc[3,'goals_per90_shrunk_90d'])
    assert 'contract_days_remaining' not in BASE_NUMERIC


def test_learning_curves_nested_samples_and_no_future_rows():
    dates=pd.Series(pd.to_datetime(['2019-01-01','2020-01-01','2021-01-01','2022-12-31',
        '2023-01-01','2023-12-31','2024-01-01']))
    splits=list(learning_curve_splits(dates,fractions=(0.25,0.5,1.0)))
    previous=set()
    for fraction,tr,va in splits:
        assert previous<=set(tr)
        assert dates.iloc[tr].max()<dates.iloc[va].min()
        assert list(va)==[4,5]
        assert 6 not in tr and 6 not in va
        previous=set(tr)
    assert len(splits[-1][1])==4
    for first,second in zip(splits,learning_curve_splits(dates,fractions=(0.25,0.5,1.0))):
        np.testing.assert_array_equal(first[1],second[1])
    with pytest.raises(ValueError,match='Fractions'):
        list(learning_curve_splits(dates,fractions=(1.0,0.5)))


@pytest.mark.parametrize('estimator',[LinearRegression(),make_svr(0.1)])
def test_syllabus_models_predict_and_serialize(estimator,tmp_path):
    train=raw_fixture()
    pipeline=model_pipeline(estimator,recent=True,scale=True)
    fit_checked(pipeline,train,np.log([100000,200000,300000,400000]))
    test=train.iloc[:1].copy()
    test['foot']='unseen'
    expected=pipeline.predict(test)
    assert np.isfinite(expected).all()
    joblib.dump(pipeline,tmp_path/'syllabus.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'syllabus.joblib').predict(test),expected)
