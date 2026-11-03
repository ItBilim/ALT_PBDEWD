"""Regenerate frozen daily features from raw files and audited row mapping."""
from .common import *
from .features import build_features
from .splits import reserve
def rebuild(ds):
 plan=cfg();meta=read_json(ROOT/f'data/prepared/{ds}_features.json');names=meta['predictor_names']
 raw=pd.read_parquet(ROOT/f'data/raw/{ds}.parquet',columns=['user_id','timestamp','label',*names]);mapping=pd.read_parquet(ROOT/f'data/mapping/{ds}.parquet')
 assert len(raw)==len(mapping) and np.array_equal(mapping.raw_row,np.arange(len(raw)))
 for c in ['user_id','timestamp','label']:assert np.array_equal(raw[c].to_numpy(),mapping[c].to_numpy()),c
 frame=raw.loc[mapping.retained.to_numpy()].copy()
 frame['row_id']=[f'{ds}:{r:07d}' for r in mapping.loc[mapping.retained,'raw_row']]
 frame=frame.sort_values(['user_id','timestamp'],kind='stable').reset_index(drop=True)
 assert not frame.duplicated(['user_id','timestamp']).any()
 frame['available_at']=frame.timestamp+pd.Timedelta(hours=1);frame['window_start']=frame.timestamp-pd.Timedelta(hours=1)
 frame['decision_at']=frame.available_at.dt.normalize()+pd.Timedelta(days=1);frame['reserved_unseen']=frame.user_id.map(reserve)
 past_labels=np.full(len(frame),np.nan)
 for indices in frame.groupby('user_id',sort=False).indices.values():
  t=frame.loc[indices,'timestamp'].to_numpy(dtype='datetime64[ns]').astype(np.int64);y=frame.loc[indices,'label'].to_numpy()
  previous=np.searchsorted(t,t-2*3600*10**9,side='left')-1;valid=previous>=0
  past_labels[np.asarray(indices)[valid]]=y[previous[valid]]
 features,_=build_features(frame,names,plan['features'])
 base=frame[['row_id','user_id','timestamp','label','available_at','window_start','decision_at','reserved_unseen']].set_index('row_id')
 base['diagnostic_past_label']=past_labels
 eligible=features['meta__common_cohort_eligible'].eq(1)
 daily=base.sort_values(['available_at'],kind='stable').reset_index().drop_duplicates(['user_id','decision_at'],keep='last').set_index('row_id')
 daily=daily.loc[daily.index.isin(features.index[eligible]) & daily.label.isin([0,1])].join(features.loc[lambda x:x.index.isin(daily.index)])
 daily['batch_window_age_hours']=((daily.decision_at-daily.available_at).dt.total_seconds()/3600).astype('float32')
 daily=daily.sort_values(['decision_at','user_id']).reset_index()
 saved=load_daily(ds);pd.testing.assert_frame_equal(daily,saved,check_exact=True)
 receipt=dict(status='PASS_EXACT_REBUILD',dataset=ds,raw_rows=len(raw),retained_rows=len(frame),daily_rows=len(daily),prepared_sha256=sha(ROOT/f'data/prepared/{ds}.parquet'),labels_used_in_predictor_construction=False,diagnostic_past_label_excluded_from_predictors=True)
 write_json(ROOT/f'results/preparation/{ds}.json',receipt)
 print(f'RAW FEATURE REBUILD PASS {ds}: {len(daily)} exact daily rows',flush=True)

