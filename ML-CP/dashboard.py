"""Local presentation dashboard. Run: python -m streamlit run dashboard.py"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT/'models/upgraded'
st.set_page_config(page_title='Touchline | Football valuation lab',page_icon='⚽',layout='wide')
st.markdown('''<style>
 .block-container {max-width:1200px;padding-top:4.5rem;padding-bottom:3rem;}
 h1,h2,h3 {text-wrap:balance;} p {text-wrap:pretty;}
 [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
 [data-testid="stSidebar"] {border-right:1px solid #e2e8f0;}
 .eyebrow {color:#0f766e;font-size:.8rem;font-weight:700;}
 @media print { [data-testid="stSidebar"],header {display:none;} }
 </style>''',unsafe_allow_html=True)


@st.cache_data
def load_outputs(signature):
    metadata = json.loads((ARTIFACTS/'run_metadata.json').read_text(encoding='utf-8'))
    predictions = pd.read_parquet(ARTIFACTS/'test_predictions.parquet')
    return metadata,predictions


def euros(value):
    return f'€{value/1e6:,.2f}m' if abs(value)>=1e6 else f'€{value:,.0f}'


def style_chart(fig, title=None):
    fig.update_layout(template='plotly_white',font=dict(family='Arial',color='#334155'),
        margin=dict(t=45 if title else 20,l=10,r=10,b=15),title=title,
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h',y=-0.2),hoverlabel=dict(bgcolor='white'))
    return fig


def metrics_table(frame):
    records=[]
    for key,name in [('annual_xgb','Annual features · XGBoost'),('recent_xgb','Recent form + coverage · XGBoost'),
        ('recent_ridge','Same recent features · Ridge'),('recent_ols','Same recent features · Ordinary least squares'),
        ('recent_svr','Same recent features · Linear SVR')]:
        if f'{key}_log' not in frame:
            continue
        log_error=np.log(frame.market_value_in_eur)-frame[f'{key}_log']
        records.append({'Model':name,'Average error (€)':float((frame.market_value_in_eur-frame[f'{key}_eur']).abs().mean()),
            'Log RMSE':float(np.sqrt(np.mean(log_error**2))),
            'Within 2× (%)':float(100*(log_error.abs()<=np.log(2)).mean())})
    return pd.DataFrame(records)


st.sidebar.markdown('## ⚽ Touchline')
st.sidebar.caption('FOOTBALL VALUATION LAB')
view=st.sidebar.radio('Explore the project',['Overview','Player explorer','Learning curves','Method & caveats','48-hour roadmap'])
st.sidebar.divider()
st.sidebar.caption('College ML project · retrospective analysis\n\nTransfermarkt market estimates, not transfer fees.')

if not (ARTIFACTS/'run_metadata.json').exists():
    st.title('Your valuation lab is almost ready')
    st.info('Run the upgraded experiment once, then refresh this page.')
    st.code('python run_upgrade.py',language='bash')
    st.stop()

metadata,data=load_outputs((ARTIFACTS/'run_metadata.json').stat().st_mtime_ns)
st.markdown('<div class="eyebrow">MACHINE LEARNING / FOOTBALL ANALYTICS</div>',unsafe_allow_html=True)

if view=='Overview':
    st.title('What is a player worth?')
    st.write('Explore how age, league and recorded match performance relate to a player’s market valuation.')
    st.caption(f'Evaluation: {metadata["test_start"]} to {metadata["test_end"]} · Trained on valuations before 2024 · Saved historical predictions')
    c1,c2=st.columns(2)
    position=c1.selectbox('Position',['All positions']+sorted(data.position.dropna().unique()),key='position')
    tier=c2.selectbox('League group',['All leagues']+sorted(data.league_tier.dropna().unique()),key='tier')
    frame=data.copy()
    if position!='All positions': frame=frame.loc[frame.position==position]
    if tier!='All leagues': frame=frame.loc[frame.league_tier==tier]
    if frame.empty:
        st.info('No observations match these filters. Choose All positions or All leagues.')
        st.stop()
    metrics=metrics_table(frame)
    enhanced=metrics.iloc[1]
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Valuations evaluated',f'{len(frame):,}',help='Repeated valuations of the same player are separate rows.')
    k2.metric('Average error',euros(enhanced['Average error (€)']),help='Mean absolute difference between predicted and reference value. Lower is better.')
    k3.metric('Within a factor of 2',f'{enhanced["Within 2× (%)"]:.1f}%',help='Prediction lies between half and twice the reference value. This is not classification accuracy.')
    no_minutes=100*frame.observed_minutes_365d.eq(0).mean()
    k4.metric('No recorded minutes',f'{no_minutes:.1f}%',help='No minutes captured in the preceding year; missing coverage can explain this.')
    st.divider()
    left,right=st.columns([1.25,1])
    with left:
        st.subheader('Did recent form help?')
        st.caption('Same test rows and calendar validation. Lower error is better.')
        chart_metrics=metrics.assign(Variant=['Annual XGBoost','Recent-form XGBoost','Ridge','Ordinary least squares','SVR (linear)'][:len(metrics)])
        fig=px.bar(chart_metrics,x='Average error (€)',y='Variant',orientation='h',
            color='Variant',color_discrete_sequence=['#94a3b8','#0f766e','#cbd5e1','#64748b','#afc8c5'],text_auto='.3s')
        fig.update_layout(showlegend=False,height=350,yaxis_title=None)
        fig.update_xaxes(tickprefix='€',tickformat='.2s')
        st.plotly_chart(style_chart(fig),width='stretch')
        delta=100*(metrics.iloc[0]['Average error (€)']-enhanced['Average error (€)'])/metrics.iloc[0]['Average error (€)']
        direction='lower' if delta>=0 else 'higher'
        st.write(f'With recent form and coverage, average euro error is **{abs(delta):.1f}% {direction}** than the corrected annual-feature model for this selection.')
    with right:
        st.subheader('Where are the blind spots?')
        st.caption('Relative error by observed playing-time coverage.')
        segment=frame.assign(Coverage=np.where(frame.observed_minutes_365d>0,'Recorded minutes','No recorded minutes'),
            log_abs_error=(np.log(frame.market_value_in_eur)-frame.recent_xgb_log).abs())
        grouped=segment.groupby('Coverage',as_index=False).agg(**{'Log MAE':('log_abs_error','mean'),'Rows':('player_id','size')})
        fig=px.bar(grouped,x='Coverage',y='Log MAE',text_auto='.3f',color_discrete_sequence=['#0f766e'],hover_data=['Rows'])
        fig.update_layout(height=280)
        st.plotly_chart(style_chart(fig),width='stretch')
        st.caption('Zero recorded minutes is a coverage signal. It does not establish inactivity, injury or low ability.')
    st.subheader('The comparison, in numbers')
    st.dataframe(metrics,hide_index=True,width='stretch',column_config={
        'Average error (€)':st.column_config.NumberColumn(format='€%.0f'),
        'Log RMSE':st.column_config.NumberColumn(format='%.4f'),
        'Within 2× (%)':st.column_config.NumberColumn(format='%.1f%%')})
    st.caption('Log RMSE measures error on a relative scale. OLS, Ridge, linear SVR and recent-form XGBoost use the same inputs and full training rows. The headline remains recent-form XGBoost; models are not selected using these test results.')
    st.download_button('Download comparison CSV',metrics.to_csv(index=False),'model_comparison.csv','text/csv')

elif view=='Player explorer':
    st.title('A player, through the model’s eyes')
    st.write('Choose a recorded player and valuation date to compare the reference value with the model’s estimate.')
    query=st.text_input('Search by player name',placeholder='Try Lamine Yamal, Haaland or a player you know')
    latest=data.sort_values('valuation_date').drop_duplicates('player_id',keep='last')
    choices=latest.loc[latest.player_name.str.contains(query,case=False,regex=False,na=False)] if query.strip() else latest
    choices=choices.sort_values(['market_value_in_eur','player_id'],ascending=[False,True]).head(100)
    if choices.empty:
        st.info('No matching player in the saved test predictions. Try a shorter name.')
        st.stop()
    labels={int(row.player_id):f'{row.player_name} · {row.position}' for row in choices.itertuples()}
    player=st.selectbox('Player (up to 100 matches, sorted by latest reference value)',list(labels),format_func=labels.get)
    history=data.loc[data.player_id==player].sort_values('valuation_date')
    dates=history.valuation_date.dt.strftime('%Y-%m-%d').tolist()
    date=st.selectbox('Historical valuation date',dates,index=len(dates)-1)
    row=history.loc[history.valuation_date.dt.strftime('%Y-%m-%d')==date].iloc[0]
    st.subheader(row.player_name)
    st.caption(f'{row.club_name} · {row.position} · Age {row.age:.1f} · {row.league_tier} · {date}')
    a,b,c=st.columns(3)
    a.metric('Reference market value',euros(row.market_value_in_eur))
    b.metric('Model estimate',euros(row.recent_xgb_eur))
    gap=100*(row.recent_xgb_eur/row.market_value_in_eur-1)
    c.metric('Model minus reference',f'{gap:+.1f}%',help='A disagreement, not evidence that the market is wrong.')
    if row.observed_minutes_365d==0:
        st.warning('No minutes were recorded in the preceding year. Treat this estimate cautiously: match coverage may be incomplete.')
    else:
        st.info(f'{int(row.observed_minutes_365d):,} recorded minutes in the previous year. Last recorded appearance: {pd.Timestamp(row.last_observed_appearance_date):%d %b %Y}.')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=history.valuation_date,y=history.market_value_in_eur,name='Reference market value',mode='lines+markers',line=dict(color='#334155')))
    fig.add_trace(go.Scatter(x=history.valuation_date,y=history.recent_xgb_eur,name='Model estimate',mode='lines+markers',line=dict(color='#0f766e',dash='dash')))
    fig.update_layout(yaxis_title='Market value (€)',xaxis_title='Valuation date',height=350)
    fig.update_yaxes(tickprefix='€',tickformat='.2s')
    st.plotly_chart(style_chart(fig,'Reference and estimate across the test period'),width='stretch')
    st.subheader('The recent-form inputs')
    st.caption('These are recorded inputs, not a feature-attribution explanation. Windows overlap and stop before the selected valuation date.')
    windows=pd.DataFrame([{'Window':f'Previous {d} days','Minutes':int(row[f'observed_minutes_{d}d']),
        'Appearances':int(row[f'observed_appearances_{d}d']),'Goals':int(row[f'observed_goals_{d}d']),
        'Assists':int(row[f'observed_assists_{d}d'])} for d in (90,180,365)])
    st.dataframe(windows,hide_index=True,width='stretch')
    st.caption(f'Player present in pre-2024 training data: {"yes" if row.seen_in_training else "no"}. Reference values appear here for retrospective comparison and were not supplied to the predictor.')
    st.download_button('Download this player’s results',history.to_csv(index=False),f'player_{player}_results.csv','text/csv')

elif view=='Learning curves':
    st.title('Does more training data help?')
    st.write('Compare training error with error on a fixed later year as the number of training observations increases.')
    curve_file=ARTIFACTS/'learning_curves.csv'
    if not curve_file.exists():
        st.info('Learning curves have not been generated yet. Run python run_upgrade.py, then refresh.')
        st.stop()
    curves=pd.read_csv(curve_file)
    names={'recent_ols':'Ordinary least squares','recent_ridge':'Ridge regression',
           'recent_svr':'SVR (linear)','recent_xgb':'XGBoost · recent form'}
    selected=st.selectbox('Model',list(names),format_func=names.get,key='curve_model')
    rows=curves.loc[curves.model==selected].sort_values('train_rows')
    st.caption('Training samples: pre-2023 history. Fixed validation: 2023. All preprocessing is refitted on each training sample. No 2024–2026 test rows enter these curves.')
    fig=go.Figure()
    for column,label,color in [('train_log_rmse','Training error','#0f766e'),('validation_log_rmse','Validation error','#64748b')]:
        fig.add_trace(go.Scatter(x=rows.train_rows,y=rows[column],mode='lines+markers',name=label,line=dict(color=color),
            hovertemplate='%{x:,} training rows<br>Log RMSE %{y:.4f}<extra>'+label+'</extra>'))
    fig.update_layout(xaxis_title='Training observations',yaxis_title='Log RMSE · lower is better',height=390)
    st.plotly_chart(style_chart(fig),width='stretch')
    last=rows.iloc[-1]
    a,b,c=st.columns(3)
    a.metric('Full-pool training error',f'{last.train_log_rmse:.4f}')
    b.metric('2023 validation error',f'{last.validation_log_rmse:.4f}')
    c.metric('Validation − training',f'{last.generalization_gap:+.4f}',help='A gap can reflect overfitting, time drift or cohort differences. It is not a direct variance estimate.')
    st.subheader('How to explain this to your examiner')
    st.write('• Low training error with much higher validation error can suggest overfitting.\n\n'
             '• High training and validation errors that plateau can suggest underfitting or insufficient features.\n\n'
             '• Falling validation error as the sample grows suggests more observations help at this model complexity.')
    st.info('These are diagnostics, not proof of a bias–variance decomposition. The validation year is later, so market drift also affects the gap. Curves use one seeded, nested sample sequence and do not provide confidence intervals.')
    st.subheader('All models at the largest training size')
    full=curves.loc[curves.fraction==1.0].copy()
    full['Model']=full.model.map(names)
    st.dataframe(full[['Model','train_rows','train_log_rmse','validation_log_rmse','generalization_gap']],hide_index=True,width='stretch')
    st.download_button('Download all learning-curve results',curves.to_csv(index=False),'learning_curves.csv','text/csv')

elif view=='Method & caveats':
    st.title('From match records to an estimate')
    st.write('The model estimates a valuation on a historical date from information recorded before that date. It does not forecast six months ahead.')
    st.subheader('1 · Build historical features')
    st.write('Match records → 90/180/365-day windows → recent-form changes and coverage flags. Goals and assists per 90 minutes are shrunk toward a training-only positional prior when exposure is small.')
    st.subheader('2 · Validate on later calendar years')
    st.dataframe(pd.DataFrame(metadata['folds']),hide_index=True,width='stretch')
    st.write('Each fold fits its own imputer, rate priors, category encoder and model. All observations on a date stay together. Tuning and final training use the same estimator configuration.')
    st.subheader('Syllabus models')
    st.write('Ordinary least squares minimizes squared log-value error without a coefficient penalty. Ridge adds L2 regularization. Linear SVR uses a squared epsilon-insensitive loss: small errors inside its epsilon band carry no loss, with regularization controlled by C. This is a genuine linear support-vector regressor, not an RBF-kernel model.')
    st.caption('SVR C is chosen on the same 2020–2022 calendar folds. OLS has no regularization hyperparameter; Ridge retains alpha 100. All three use standardized numeric features and the same categorical inputs as recent-form XGBoost.')
    st.subheader('3 · Evaluate saved models')
    st.write(f'Final training: {metadata["train_start"]}–{metadata["train_end"]}. Evaluation: {metadata["test_start"]}–{metadata["test_end"]}. The feature transformer, encoder and model are saved together and prediction replay is checked.')
    st.subheader('What this experiment cannot establish')
    for limitation in metadata['limitations']:
        st.write('• '+limitation)
    st.warning('The 2024–2026 period was already examined in the original project. This is a historical benchmark, not a newly untouched holdout. There are no calibrated prediction intervals in upgrades 1–2.')
    st.caption('The dashboard uses the new run in models/upgraded. Original README tables and models/ files describe the earlier benchmark.')
    with st.expander('Could this work for other sports?'):
        st.write('Yes, as a shared evaluation and presentation framework with sport-specific features and models. Football market value, a baseball salary and an IPL auction bid are different targets. They must be labeled and evaluated separately; converting currencies does not make them equivalent. This implementation currently supports football only.')
    with st.expander('Reproducibility details'):
        st.json({k:metadata[k] for k in ['version','seed','versions','search_depths','elapsed_seconds','data_sha256']})
    st.download_button('Download run manifest',json.dumps(metadata,indent=2),'run_metadata.json','application/json')

else:
    st.title('A focused 48-hour upgrade')
    st.write('The validation repair, recent-form features and presentation interface are implemented. Use the remaining time to strengthen evidence and rehearse the explanation.')
    st.dataframe(pd.DataFrame([
        ('0–8 hours','Reproduce this run; inspect coverage and largest errors','Reliable baseline and data audit'),
        ('8–18 hours','Add persistence and a separate valuation-history experiment','A stronger research question'),
        ('18–30 hours','Add grouped ablations OR a small dated-text pilot','Evidence that each input helps'),
        ('30–40 hours','Add local explanations and polish the presentation','Explain 4–6 contrasting cases'),
        ('40–48 hours','Freeze results, test on presentation laptop, rehearse','A reproducible demonstration'),
    ],columns=['Time budget','Work','Outcome']),hide_index=True,width='stretch')
    st.subheader('Can it become multimodal?')
    st.write('Yes: combine numeric match statistics with dated scouting or news text. Start with TF–IDF + Ridge for a simple baseline, then optionally try frozen sentence embeddings. Compare statistics-only, text-only and fused models on the same covered rows and chronological splits.')
    st.warning('No historical scouting-text dataset is included here. Text collection, player matching and publication-date verification are the limiting steps. The current model is tabular, not multimodal.')
    st.write('Remove explicit market values and transfer fees from text, and exclude articles published on or after the prediction date. Do not create historical reports from future knowledge. Player photos, a chatbot, or a summary generated from the same numeric inputs do not establish a useful multimodal valuation model.')
    st.caption('See UPGRADE_GUIDE.md for the complete scope, commands and multimodal experiment design.')

st.divider()
st.caption('Touchline · College ML project · Estimates describe model behavior, not validated transfer opportunities.')
