"""Recompute four research-question tables from fresh saved predictions."""
import pickle,json
from .common import *
from .splits import select_split
from .run import collect
from .evaluation import evaluate_predictions
from sklearn.metrics import average_precision_score,roc_auc_score

def metrics(pred):
 rows=[]
 for key,g in pred.groupby(['dataset','split','cohort','model','representation'],sort=True):
  for col in ['score_raw','score']:
   if g.label.nunique()!=2:raise RuntimeError('Single class comparison')
   rows.append(dict(zip(['dataset','split','cohort','model','representation'],key),score_type=col,rows=len(g),users=g.user_id.nunique(),positive_rows=int(g.label.sum()),positive_users=g.loc[g.label.eq(1),'user_id'].nunique(),prevalence=float(g.label.mean()),AP=float(average_precision_score(g.label,g[col])),ROC_AUC=float(roc_auc_score(g.label,g[col]))))
 return pd.DataFrame(rows)

def reference_check(pred,branch):
 old=pd.read_parquet(ROOT/f'protocol/reference/{branch}_predictions.parquet')
 keys=['dataset','split','cohort','model','representation','row_id']
 joined=pred.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
 if len(joined)!=len(pred) or len(joined)!=len(old):raise RuntimeError(f'{branch} reference row mismatch')
 for col in ['user_id','timestamp','label']:
  assert np.array_equal(joined[col+'_new'],joined[col+'_old']),col
 for col in ['score','score_raw']:
  if not np.allclose(joined[col+'_new'],joined[col+'_old'],rtol=1e-10,atol=1e-12):raise RuntimeError(f'{branch} fresh refit changed reference {col}, maxdiff {np.max(np.abs(joined[col+"_new"]-joined[col+"_old"]))}')
 return dict(branch=branch,status='PASS_FRESH_FITS_MATCH_PREVIOUS_PREDICTIONS',rows=len(joined),tolerance_rtol=1e-10,tolerance_atol=1e-12)

def transfer(plan):
 rows=[];predictions=[];split=next(s for s in plan['splits'] if s['name']=='FINAL')
 for source,target in plan['transfer_pairs']:
  target_data=load_daily(target);_,ev,_=select_split(target_data,split);ev=ev.loc[ev.cohort.eq('UNSEEN')].copy()
  for model in plan['models']:
   for rep in ['R1_HISTORY','R4_DRIFT']:
    folder=ROOT/'models/main'/source/'FINAL'/model/rep;fit=read_json(folder/'fit.json')
    with (folder/'model.pkl').open('rb') as stream:pipe=pickle.load(stream)
    partition=pd.read_parquet(folder/'partition_rows.parquet');assert not set(partition.user_id)&set(ev.user_id)
    raw=pipe.raw_scores(ev[fit['columns']]);cal=pipe.predict_scores(ev[fit['columns']])
    rows.append(dict(source=source,target=target,model=model,representation=rep,rows=len(ev),users=ev.user_id.nunique(),positive_users=ev.loc[ev.label.eq(1),'user_id'].nunique(),AP_native=float(average_precision_score(ev.label,raw)),AP_calibrated=float(average_precision_score(ev.label,cal)),ROC_AUC=float(roc_auc_score(ev.label,raw)),target_labels_used_for_fit=False,source_target_user_overlap=0))
    p=ev[['row_id','user_id','timestamp','label']].copy();p['source']=source;p['target']=target;p['model']=model;p['representation']=rep;p['score_raw']=raw;p['score']=cal;predictions.append(p)
 result=pd.DataFrame(rows);result.to_csv(ROOT/'results/device_transfer_metrics.csv',index=False);pd.concat(predictions,ignore_index=True).to_parquet(ROOT/'results/device_transfer_predictions.parquet',index=False)
 contrasts=[]
 for key,g in result.groupby(['source','target','model'],sort=True):
  lo=g.loc[g.representation.eq('R1_HISTORY')].iloc[0];hi=g.loc[g.representation.eq('R4_DRIFT')].iloc[0]
  contrasts.append(dict(zip(['source','target','model'],key),R1_AP=lo.AP_native,R4_AP=hi.AP_native,delta_AP=hi.AP_native-lo.AP_native,scope='DESCRIPTIVE_DEVICE_TRANSFER_NO_CI'))
 pd.DataFrame(contrasts).to_csv(ROOT/'results/RQ4_device_transfer.csv',index=False)
 return result

def aggregate():
 plan=cfg();pred,registry=collect(replay=True);out=ROOT/'results';out.mkdir(exist_ok=True)
 registry.to_csv(out/'fit_registry.csv',index=False)
 main=pred.loc[pred.branch.eq('main')].copy();factor=pred.loc[pred.branch.eq('factorial')].copy()
 for name,frame in [('main',main),('factorial',factor)]:frame.to_parquet(out/f'{name}_predictions.parquet',index=False)
 migration=[reference_check(main,'main'),reference_check(factor,'factorial')]
 write_json(out/'fresh_refit_reference_checks.json',migration)
 m=metrics(main);a=metrics(factor);m.to_csv(out/'main_metrics.csv',index=False);a.to_csv(out/'factorial_metrics.csv',index=False)
 _,pairs,metadata=evaluate_predictions(main,dict(datasets=plan['datasets'],primary_models=plan['models'],primary_contrasts=plan['statistics']['new_contrasts'],secondary_contrasts=[],bootstrap_repeats=10000,bootstrap_chunk_size=64,seed=plan['seed'],primary_split='FINAL',primary_cohort='UNSEEN',primary_population='USER_DAY_PRIMARY',time_column='decision_at',capacities=[.01,.05],uncalibrated_models=plan['models']))
 pairs=pairs.rename(columns={'primary':'exploratory_family_member'});pairs['registration']=plan['registration'];pairs.to_csv(out/'exploratory_40_and_descriptive_contrasts.csv',index=False)
 metadata['registration']=plan['registration'];metadata['expected_draws_per_bonferroni_tail']=6.25;metadata['warning']='Retrospective; conditional on saved predictions; not fitting uncertainty; original40 family retained'
 write_json(out/'uncertainty_metadata.json',metadata)
 primary=pairs.loc[pairs.exploratory_family_member].copy();assert len(primary)==40
 for rq,contrast in [('RQ1','R1_HISTORY - R0_CURRENT'),('RQ2','R4_DRIFT - R1_HISTORY')]:primary.loc[primary.contrast.eq(contrast)].to_csv(out/f'{rq}_contrasts.csv',index=False)
 marg=[]
 for key,g in a.groupby(['dataset','split','cohort','model','score_type'],sort=True):
  scores={int(r.representation.split('_')[1]):r.AP for r in g.itertuples()}
  for bit,factor_name in [(1,'STATIC'),(2,'ADAPT'),(4,'DRIFT')]:
   diffs=[scores[n|bit]-scores[n] for n in range(8) if not n&bit]
   marg.append(dict(zip(['dataset','split','cohort','model','score_type'],key),factor=factor_name,mean_AP_difference=float(np.mean(diffs)),minimum_context_difference=min(diffs),maximum_context_difference=max(diffs),interaction='Movement enters only when STATIC and ADAPT present; not pure causal effect',scope='DESCRIPTIVE_NO_CI'))
 pd.DataFrame(marg).to_csv(out/'RQ3_factorial_effects.csv',index=False)
 tr=transfer(plan);assert len(tr)==16
 old=read_json(ROOT/'protocol/historical/v2_statistics.json');assert old['primary_family_size']==metadata['primary_family_size']==40
 oldpairs=pd.read_csv(ROOT/'protocol/historical/v2_paired_AP.csv');oldpairs=oldpairs.loc[oldpairs.exploratory_family_member]
 joined=primary.merge(oldpairs,on=['dataset','model','contrast'],suffixes=('_new','_old'),validate='one_to_one');assert len(joined)==40
 for col in ['effect','ci95_lower','ci95_upper','bonferroni_lower','bonferroni_upper']:
  if not np.allclose(joined[col+'_new'],joined[col+'_old'],rtol=1e-10,atol=1e-12):raise RuntimeError(f'Bootstrap reference differs {col}')
 write_json(out/'result_contract.json',dict(status='PASS',fresh_fitting=True,selected_models=520,hpo_trials=2400,calibrators=520,main_models=200,factorial_models=320,device_transfer_evaluations=16,device_transfer_paired_contrasts=8,exploratory_family_size=40,original32_in_historical_reference=True,reference_predictions_match=True,reference40_intervals_match=True,scope='FOUR_RQ_ONLY',onset_claim=False,external_validation=False))
 print('FOUR RQ AGGREGATION PASS: fresh520 models,2400 trials; predictions and all40 intervals match previous design',flush=True)
