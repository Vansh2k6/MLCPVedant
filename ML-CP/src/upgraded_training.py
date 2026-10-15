"""Versioned experiments for upgrades 1 and 2. Original artifacts stay intact."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor

from src.preprocessing import model_pipeline, BASE_NUMERIC, RECENT_NUMERIC, CATEGORICAL
from src.temporal import calendar_folds

SPLIT_DATE = '2024-01-01'
FLOOR, CEILING = 8.5, np.log(250_000_000)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_estimator(depth, trees=250):
    # The exact same constructor is used for validation and final fitting.
    return XGBRegressor(n_estimators=trees,max_depth=depth,learning_rate=0.06,
        subsample=0.85,colsample_bytree=0.85,reg_alpha=0.1,reg_lambda=1.0,
        min_child_weight=5,tree_method='hist',random_state=42,n_jobs=min(6,os.cpu_count() or 1))


def recorded_params(estimator):
    """Keep parameter manifests valid JSON; XGBoost uses NaN for missing data."""
    return {key: ('NaN' if isinstance(value,float) and np.isnan(value) else value)
            for key,value in estimator.get_params().items()}


def score(actual, log_pred):
    actual = np.asarray(actual, dtype=float)
    log_pred = np.asarray(log_pred, dtype=float)
    errors = np.log(actual)-log_pred
    euro_errors = actual-np.exp(log_pred)
    return dict(log_rmse=float(np.sqrt(np.mean(errors**2))),log_mae=float(np.mean(np.abs(errors))),
        euro_rmse=float(np.sqrt(np.mean(euro_errors**2))),euro_mae=float(np.mean(np.abs(euro_errors))),
        within_factor_two=float(np.mean(np.abs(errors)<=np.log(2))),rows=int(len(actual)))


def make_svr(c=1.0):
    """Full-data linear support-vector regression, not an RBF kernel approximation."""
    return LinearSVR(C=c,epsilon=0.1,loss='squared_epsilon_insensitive',dual=False,
        tol=1e-4,max_iter=20000,intercept_scaling=10,random_state=42)


def fit_checked(pipeline, X, y):
    # Never silently publish an unconverged support-vector model.
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        return pipeline.fit(X,y)


def learning_curve_splits(dates, fractions=(0.05,0.2,0.5,1.0), validation_year=2023, seed=42):
    """Nested seeded samples of earlier years; a fixed later year for all points.

    Sampling within the earlier pool is legitimate: no validation/future rows
    enter training, and it avoids conflating sample size with a newer cutoff.
    """
    dates=pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if dates.isna().any() or not fractions or any(f<=0 or f>1 for f in fractions):
        raise ValueError('Valid dates and fractions in (0,1] are required')
    if list(fractions)!=sorted(set(fractions)):
        raise ValueError('Fractions must be unique and increasing')
    start=pd.Timestamp(year=validation_year,month=1,day=1)
    end=pd.Timestamp(year=validation_year+1,month=1,day=1)
    pool=np.flatnonzero((dates<start).to_numpy())
    valid=np.flatnonzero(((dates>=start)&(dates<end)).to_numpy())
    if not len(pool) or not len(valid):
        raise ValueError('Need earlier training rows and a later validation year')
    order=np.random.default_rng(seed).permutation(pool)
    for fraction in fractions:
        n=max(1,int(round(len(pool)*fraction)))
        yield fraction,np.sort(order[:n]),valid


def save_learning_curve_plot(curves, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'recent_ols':'Ordinary least squares','recent_ridge':'Ridge',
            'recent_svr':'SVR (linear)','recent_xgb':'XGBoost'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True,layout='constrained')
    for ax,(key,label) in zip(axes.flat,labels.items()):
        frame=curves.loc[curves.model==key].sort_values('train_rows')
        ax.plot(frame.train_rows/1000,frame.train_log_rmse,'o-',color='#0f766e',label='Training')
        ax.plot(frame.train_rows/1000,frame.validation_log_rmse,'o--',color='#64748b',label='2023 validation')
        ax.set(title=label,xlabel='Training observations (thousands)',ylabel='Log RMSE')
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle('Learning curves: nested pre-2023 samples; fixed 2023 validation\nSingle seeded experiment; no test rows used',fontsize=12)
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def build_learning_curves(train, output_dir, xgb_depth, svr_c, trees=250):
    factories={
        'recent_ols':lambda: LinearRegression(),
        'recent_ridge':lambda: Ridge(alpha=100,solver='lsqr'),
        'recent_svr':lambda: make_svr(svr_c),
        'recent_xgb':lambda: make_estimator(xgb_depth,trees),
    }
    rows=[]
    for fraction,tr,va in learning_curve_splits(train.valuation_date):
        for name,factory in factories.items():
            started=time.time()
            model=model_pipeline(factory(),recent=True,scale=name!='recent_xgb')
            fit_checked(model,train.iloc[tr],np.log(train.market_value_in_eur.iloc[tr]))
            train_pred=np.clip(model.predict(train.iloc[tr]),FLOOR,CEILING)
            val_pred=np.clip(model.predict(train.iloc[va]),FLOOR,CEILING)
            tr_score=score(train.market_value_in_eur.iloc[tr],train_pred)
            va_score=score(train.market_value_in_eur.iloc[va],val_pred)
            rows.append(dict(model=name,fraction=fraction,train_rows=len(tr),validation_rows=len(va),
                train_log_rmse=tr_score['log_rmse'],validation_log_rmse=va_score['log_rmse'],
                generalization_gap=va_score['log_rmse']-tr_score['log_rmse'],
                train_start=str(train.valuation_date.iloc[tr].min().date()),
                train_end=str(train.valuation_date.iloc[tr].max().date()),
                validation_start=str(train.valuation_date.iloc[va].min().date()),
                validation_end=str(train.valuation_date.iloc[va].max().date()),
                elapsed_seconds=round(time.time()-started,2),seed=42))
            print(f'Learning curve {name} n={len(tr):,}: train {tr_score["log_rmse"]:.4f}, validation {va_score["log_rmse"]:.4f}',flush=True)
    curve_frame=pd.DataFrame(rows)
    curve_frame.to_csv(Path(output_dir)/'learning_curves.csv',index=False)
    save_learning_curve_plot(curve_frame,Path(output_dir)/'learning_curves.png')
    return dict(validation_year=2023,fractions=[0.05,0.2,0.5,1.0],seed=42,
        models=list(factories),scoring='Clipped log RMSE, same clipping as test evaluation',
        sampling='Nested random row samples of pre-2023 history; fixed 2023 validation; preprocessing refit per sample.',
        interpretation='Single seeded experiment, not confidence bands. Repeated players and temporal shift affect the gap; it is not a formal bias-variance decomposition.')


def train_upgraded(data_path, output_dir, depths=(5,7), trees=250):
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    data = pd.read_parquet(data_path)
    data['valuation_date'] = pd.to_datetime(data['valuation_date'])
    data['last_observed_appearance_date'] = pd.to_datetime(data['last_observed_appearance_date'])
    data = data.sort_values(['valuation_date','player_id']).reset_index(drop=True)
    train = data.loc[data.valuation_date < SPLIT_DATE].reset_index(drop=True)
    test = data.loc[data.valuation_date >= SPLIT_DATE].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError('Both pre-2024 training data and later test data are required')
    y = np.log(train.market_value_in_eur.to_numpy())
    folds = list(calendar_folds(train.valuation_date))
    fold_details = [dict(validation_year=int(train.valuation_date.iloc[va].dt.year.iloc[0]),
        train_rows=len(tr),validation_rows=len(va),train_end=str(train.valuation_date.iloc[tr].max().date()),
        validation_start=str(train.valuation_date.iloc[va].min().date()),
        validation_end=str(train.valuation_date.iloc[va].max().date())) for tr,va in folds]
    print(f'Training {len(train):,} rows; testing {len(test):,} rows. Calendar folds: 2020, 2021, 2022.',flush=True)
    predictions = test[['player_id','player_name','valuation_date','age','position','league_tier','club_name',
        'market_value_in_eur','minutes_last_365d','days_since_last_appearance','has_observed_history',
        'last_observed_appearance_date'] + [f'observed_{m}_{d}d' for d in (90,180,365) for m in ('minutes','appearances','goals','assists')]].copy()
    predictions['seen_in_training'] = predictions.player_id.isin(train.player_id)
    results, cv_results, fitted_params = [], [], {}
    specifications = [('annual_xgb','Corrected XGBoost · annual features',False),
                      ('recent_xgb','Upgraded XGBoost · recent form + coverage',True)]
    for key,label,recent in specifications:
        mean_scores = []
        for depth in depths:
            errors = []
            for i,(tr,va) in enumerate(folds):
                pipeline = model_pipeline(make_estimator(depth,trees),recent=recent)
                pipeline.fit(train.iloc[tr],y[tr])
                pred = np.clip(pipeline.predict(train.iloc[va]),FLOOR,CEILING)
                metric = score(train.market_value_in_eur.iloc[va],pred)
                cv_results.append(dict(model=key,depth=depth,fold=i+1,**fold_details[i],**metric))
                errors.append(metric['log_rmse'])
                print(f'{key}: depth {depth}, fold {i+1}: log RMSE {metric["log_rmse"]:.4f}',flush=True)
            mean_scores.append((float(np.mean(errors)),depth))
        cv_mean,depth = min(mean_scores)
        pipeline = model_pipeline(make_estimator(depth,trees),recent=recent)
        pipeline.fit(train,y)
        raw_pred = pipeline.predict(test)
        pred = np.clip(raw_pred,FLOOR,CEILING)
        predictions[f'{key}_log'] = pred
        predictions[f'{key}_eur'] = np.exp(pred)
        artifact = output_dir/f'{key}.joblib'
        joblib.dump(pipeline,artifact,compress=3)
        fitted_params[key]=recorded_params(pipeline.named_steps['regressor'])
        replay = joblib.load(artifact).predict(test.iloc[:128])
        np.testing.assert_allclose(replay,raw_pred[:128],rtol=1e-6,atol=1e-6)
        results.append(dict(id=key,label=label,depth=depth,cv_log_rmse=cv_mean,
            clipped_predictions=int(np.sum(raw_pred!=pred)),**score(test.market_value_in_eur,pred)))
        print(f'{key} final: {results[-1]}',flush=True)
        if recent:
            columns = pipeline.named_steps['columns'].get_feature_names_out()
            weights = pipeline.named_steps['regressor'].feature_importances_
            pd.DataFrame({'feature':columns,'gain':weights}).sort_values('gain',ascending=False).to_csv(output_dir/'feature_importance.csv',index=False)
    # Ridge sees exactly the same engineered input feature set as the recent-form XGBoost.
    ridge = model_pipeline(Ridge(alpha=100.0,solver='lsqr'),recent=True,scale=True)
    ridge.fit(train,y)
    pred = np.clip(ridge.predict(test),FLOOR,CEILING)
    predictions['recent_ridge_log'], predictions['recent_ridge_eur'] = pred,np.exp(pred)
    joblib.dump(ridge,output_dir/'recent_ridge.joblib',compress=3)
    fitted_params['recent_ridge']=recorded_params(ridge.named_steps['regressor'])
    results.append(dict(id='recent_ridge',label='Ridge · same upgraded features',alpha=100,
        **score(test.market_value_in_eur,pred)))

    # Tune only on the original calendar development folds, never the test period
    # or the fixed 2023 learning-curve validation period.
    svr_scores=[]
    for c in (0.01,0.1,1.0):
        fold_errors=[]
        for i,(tr,va) in enumerate(folds):
            candidate=model_pipeline(make_svr(c),recent=True,scale=True)
            fit_checked(candidate,train.iloc[tr],y[tr])
            pred=np.clip(candidate.predict(train.iloc[va]),FLOOR,CEILING)
            metric=score(train.market_value_in_eur.iloc[va],pred)
            cv_results.append(dict(model='recent_svr',C=c,fold=i+1,**fold_details[i],**metric))
            fold_errors.append(metric['log_rmse'])
            print(f'Linear SVR: C={c}, fold {i+1}: log RMSE {metric["log_rmse"]:.4f}',flush=True)
        svr_scores.append((float(np.mean(fold_errors)),c))
    svr_cv,svr_c=min(svr_scores)
    for key,label,estimator in [
        ('recent_ols','Ordinary least squares · same upgraded features',LinearRegression()),
        ('recent_svr','Linear SVR · same upgraded features',make_svr(svr_c)),
    ]:
        pipeline=model_pipeline(estimator,recent=True,scale=True)
        fit_checked(pipeline,train,y)
        raw_pred=pipeline.predict(test)
        pred=np.clip(raw_pred,FLOOR,CEILING)
        predictions[f'{key}_log'],predictions[f'{key}_eur']=pred,np.exp(pred)
        artifact=output_dir/f'{key}.joblib'
        joblib.dump(pipeline,artifact,compress=3)
        np.testing.assert_allclose(joblib.load(artifact).predict(test.iloc[:128]),raw_pred[:128],rtol=1e-6,atol=1e-6)
        fitted_params[key]=recorded_params(estimator)
        result=dict(id=key,label=label,clipped_predictions=int(np.sum(raw_pred!=pred)),**score(test.market_value_in_eur,pred))
        if key=='recent_svr':
            result.update(C=svr_c,cv_log_rmse=svr_cv,iterations=int(estimator.n_iter_))
        results.append(result)
        print(f'{key} final: {result}',flush=True)
    xgb_depth=next(r['depth'] for r in results if r['id']=='recent_xgb')
    curve_metadata=build_learning_curves(train,output_dir,xgb_depth,svr_c,trees)
    predictions.to_parquet(output_dir/'test_predictions.parquet',index=False)
    pd.DataFrame(cv_results).to_csv(output_dir/'validation_scores.csv',index=False)
    pd.DataFrame(results).to_csv(output_dir/'metrics.csv',index=False)
    segments = []
    segment_frame = predictions.assign(year=predictions.valuation_date.dt.year.astype(str),
        coverage=np.where(predictions.observed_minutes_365d>0,'Recorded minutes','No recorded minutes'),
        player_cohort=np.where(predictions.seen_in_training,'Seen player','Unseen player'),
        age_band=pd.cut(predictions.age,[0,21,25,29,33,100],right=False,labels=['Under 21','21–24','25–28','29–32','33+']).astype(str))
    for dimension in ('year','coverage','player_cohort','age_band','position','league_tier'):
        for group, frame in segment_frame.groupby(dimension):
            segments.append(dict(dimension=dimension,group=str(group),**score(frame.market_value_in_eur,frame.recent_xgb_log)))
    pd.DataFrame(segments).to_csv(output_dir/'segments.csv',index=False)
    versions = {name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','xgboost','duckdb','joblib','pyarrow')}
    metadata = dict(version='college-syllabus-models-v2',headline_model='recent_xgb',
        train_rows=len(train),test_rows=len(test),players=int(data.player_id.nunique()),
        train_start=str(train.valuation_date.min().date()),train_end=str(train.valuation_date.max().date()),
        test_start=str(test.valuation_date.min().date()),test_end=str(test.valuation_date.max().date()),
        split_date=SPLIT_DATE,folds=fold_details,search_depths=list(depths),
        estimator_params=fitted_params,learning_curves=curve_metadata,svr_search_C=[0.01,0.1,1.0],
        metrics=results,versions=versions,seed=42,elapsed_seconds=round(time.time()-started,2),
        feature_sets={'annual':BASE_NUMERIC+CATEGORICAL,'upgraded':BASE_NUMERIC+RECENT_NUMERIC+CATEGORICAL},
        data_sha256=file_hash(data_path),preprocessing='All learned transforms fitted only on each training fold; complete pipelines saved.',
        protocol='Valuation-at-date retrospective evaluation. No valuation-history inputs. Test period was examined in the original project; not a new blind holdout.',
        limitations=['Player position, foot and height are snapshot attributes; club/league historical attribution is assumed.',
            'Missing appearances do not imply inactivity or injury. Recorded-history length is not career length.',
            'Prediction caps EUR 4,915 to EUR 250m retained for comparison. No prediction intervals or smearing in this run.',
            'Recent form is not guaranteed to improve error; compare corrected annual and upgraded models on common rows.'])
    (output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str,allow_nan=False),encoding='utf-8')
    print(f'Completed in {metadata["elapsed_seconds"]:.1f}s. Outputs: {output_dir}',flush=True)
    return metadata
