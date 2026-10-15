"""Serializable, fold-local feature preparation used in both training and inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC = [
    'age','age_squared','height_in_cm','height_missing',
    'minutes_last_365d','appearances_last_365d','goals_last_365d','assists_last_365d',
    'yellow_cards_last_365d','red_cards_last_365d','goals_per_90_clipped',
    'assists_per_90_clipped','goal_contributions_per_90_clipped','minutes_per_appearance_365d',
    'career_appearances_prior','career_minutes_prior','career_goals_prior','career_assists_prior',
    'attack_goals_per_90','midfield_assists_per_90','gk_minutes_last_365d','def_minutes_last_365d',
]
CATEGORICAL = ['position','sub_position','league_tier','competition_id','foot']
RECENT_NUMERIC = [f'observed_{metric}_{days}d' for days in (90,180) for metric in ('minutes','appearances','goals','assists')] + [
    'days_since_last_appearance','observation_history_days','has_observed_history',
    'has_recent_observations','has_recent_minutes','minutes_trend_90d',
    'goals_rate_trend_90d','assists_rate_trend_90d',
] + [f'{metric}_per90_shrunk_{days}d' for days in (90,180,365) for metric in ('goals','assists')]


class FootballFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, recent=True, pseudo_minutes=450.0):
        self.recent = recent
        self.pseudo_minutes = pseudo_minutes

    def fit(self, X, y=None):
        if self.pseudo_minutes <= 0:
            raise ValueError('pseudo_minutes must be positive')
        self.height_medians_ = X.groupby('position')['height_in_cm'].median()
        self.height_global_ = float(X['height_in_cm'].median()) if X['height_in_cm'].notna().any() else 182.0
        self.rate_priors_ = {}
        self.global_priors_ = {}
        if self.recent:
            # Exposure-weighted training priors, never estimated on validation/test rows.
            for metric in ('goals','assists'):
                totals = X.groupby('position')[[f'{metric}_last_365d','minutes_last_365d']].sum()
                self.global_priors_[metric] = 90 * X[f'{metric}_last_365d'].sum() / max(X['minutes_last_365d'].sum(), 1)
                self.rate_priors_[metric] = (90 * totals[f'{metric}_last_365d'] / totals['minutes_last_365d'].replace(0,np.nan)).fillna(self.global_priors_[metric])
        return self

    def transform(self, X):
        df = X.copy()
        df['height_missing'] = df['height_in_cm'].isna().astype(int)
        df['height_in_cm'] = df['height_in_cm'].fillna(df['position'].map(self.height_medians_)).fillna(self.height_global_)
        for metric in ('goals','assists','goal_contributions'):
            df[f'{metric}_per_90_clipped'] = df[f'{metric}_per_90_365d'].clip(0,3)
        df['attack_goals_per_90'] = df['goals_per_90_clipped'] * df['position'].eq('Attack')
        df['midfield_assists_per_90'] = df['assists_per_90_clipped'] * df['position'].eq('Midfield')
        df['gk_minutes_last_365d'] = df['minutes_last_365d'] * df['position'].eq('Goalkeeper')
        df['def_minutes_last_365d'] = df['minutes_last_365d'] * df['position'].eq('Defender')
        if self.recent:
            df['has_recent_observations'] = df['observed_appearances_365d'].gt(0).astype(int)
            df['has_recent_minutes'] = df['observed_minutes_365d'].gt(0).astype(int)
            df['minutes_trend_90d'] = 2 * df['observed_minutes_90d'] - df['observed_minutes_180d']
            for metric in ('goals','assists'):
                prior = df['position'].map(self.rate_priors_[metric]).fillna(self.global_priors_[metric])
                for days in (90,180,365):
                    df[f'{metric}_per90_shrunk_{days}d'] = (
                        90 * df[f'observed_{metric}_{days}d'] + self.pseudo_minutes * prior
                    ) / (df[f'observed_minutes_{days}d'] + self.pseudo_minutes)
                previous_minutes = df['observed_minutes_180d']-df['observed_minutes_90d']
                previous_events = df[f'observed_{metric}_180d']-df[f'observed_{metric}_90d']
                previous_rate = (90 * previous_events + self.pseudo_minutes * prior)/(previous_minutes+self.pseudo_minutes)
                df[f'{metric}_rate_trend_90d'] = df[f'{metric}_per90_shrunk_90d']-previous_rate
        for col in CATEGORICAL:
            df[col] = df[col].fillna('Unknown').astype(str)
        return df


def model_pipeline(estimator, recent=True, scale=False):
    numeric = BASE_NUMERIC + (RECENT_NUMERIC if recent else [])
    steps = [('imputer', SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    columns = ColumnTransformer([
        ('numeric', Pipeline(steps), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore',sparse_output=False), CATEGORICAL),
    ], sparse_threshold=0, verbose_feature_names_out=True)
    return Pipeline([('features',FootballFeatures(recent=recent)),('columns',columns),('regressor',estimator)])
