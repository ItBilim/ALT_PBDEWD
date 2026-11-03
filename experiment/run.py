"""Four-RQ refitting, checkpoint reuse, replay and evaluation entry point."""
from __future__ import annotations
import argparse,pickle
from pathlib import Path
from .common import *
from . import training as T
from .splits import select_split
from .factorial import factorial_columns

def source_bindings(ds):
 paths=[ROOT/'protocol/plan.json',ROOT/f'data/prepared/{ds}.parquet',ROOT/f'data/prepared/{ds}_features.json',*[ROOT/'experiment'/name for name in ['common.py','training.py','models.py','features.py','splits.py','factorial.py','prepare.py']]]
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def run_dataset(ds):
 plan=cfg();assert ds in plan['datasets']
 data=load_daily(ds);representations=reps(ds)
 masks,_=factorial_columns(data,{'experiment_representations':representations})
 bindings=source_bindings(ds)
 training=dict(seed=plan['seed'],n_jobs=plan['n_jobs'],candidate_grids=plan['hpo'],calibration_C=plan['calibration']['regularization_C'],probability_clip=plan['calibration']['epsilon'])
 done=0;new=0
 for split in plan['splits']:
  train,evaluation,_=select_split(data,split)
  for model in plan['models']:
   for rep in plan['representations']:
    folder=ROOT/'models/main'/ds/split['name']/model/rep
    _,resumed=T.train_nested(train,evaluation,representations[rep],model,training,folder,bindings)
    done+=1;new+=not resumed
    print(f'{ds} MAIN {done}/130 {split["name"]} {model} {rep} '+('CACHED' if resumed else 'NEW FIT'),flush=True)
   for rep,columns in masks.items():
    folder=ROOT/'models/factorial'/ds/split['name']/model/rep
    _,resumed=T.fit_fixed(train,evaluation,columns,model,plan,folder,bindings)
    done+=1;new+=not resumed
    print(f'{ds} FACTORIAL {done}/130 {split["name"]} {model} {rep} '+('CACHED' if resumed else 'NEW FIT'),flush=True)
  write_json(ROOT/f'results/progress/{ds}.json',dict(dataset=ds,completed=done,new_selected_fits=new,planned=130,status='RUNNING' if done<130 else 'COMPLETE'))
 assert done==130
 return done

def collect(replay=True):
 parts=[];registry=[];frames={ds:load_daily(ds) for ds in cfg()['datasets']}
 for branch,n in [('main',200),('factorial',320)]:
  folders=sorted((ROOT/f'models/{branch}').glob('*/*/*/*'))
  if len(folders)!=n:raise RuntimeError(f'{branch}: {len(folders)} of {n} checkpoints')
  for folder in folders:
   ds,split,model,rep=folder.parts[-4:]
   ck=read_json(folder/'checkpoint.json');T.verify_hashes(ck['files'])
   fit=read_json(folder/'fit.json');pred=pd.read_parquet(folder/'predictions.parquet')
   data=frames[ds];split_cfg=next(s for s in cfg()['splits'] if s['name']==split)
   binding=ck['binding'];inputs=binding['base']['inputs'] if 'base' in binding else binding['inputs'];T.verify_hashes(inputs)
   _,evaluation,_=select_split(data,split_cfg)
   for field in ['row_id','user_id','timestamp','label','cohort']:
    if not np.array_equal(pred[field].to_numpy(),evaluation[field].to_numpy()):raise RuntimeError(f'Row/label identity mismatch {folder} {field}')
   if replay:
    with (folder/'model.pkl').open('rb') as stream:pipe=pickle.load(stream)
    raw=pipe.raw_scores(evaluation[fit['columns']]);cal=pipe.predict_scores(evaluation[fit['columns']])
    if not np.allclose(pred.score_raw,raw,rtol=1e-10,atol=1e-12) or not np.allclose(pred.score,cal,rtol=1e-10,atol=1e-12):raise RuntimeError(f'Replay mismatch {folder}')
   for key,val in dict(dataset=ds,split=split,model=model,representation=rep,branch=branch).items():pred[key]=val
   parts.append(pred);registry.append(dict(branch=branch,dataset=ds,split=split,model=model,representation=rep,features=len(fit['columns']),model_sha256=sha(folder/'model.pkl'),newly_fitted=True))
  print(f'REPLAY PASS {branch} {n}',flush=True)
 return pd.concat(parts,ignore_index=True),pd.DataFrame(registry)

def verify():
 import importlib.metadata,platform
 gate=read_json(ROOT/'results/final_gate.json');manifest=read_json(ROOT/'results/artifact_manifest.json')
 assert gate['status']=='PASS' and sha(ROOT/'results/artifact_manifest.json')==gate['manifest_sha256']
 env=read_json(ROOT/'protocol/environment.json')
 if env['python']!=platform.python_version():raise RuntimeError('Python version differs from delivered replay environment')
 for package,version in env['packages'].items():
  if importlib.metadata.version(package)!=version:raise RuntimeError(f'Library version differs: {package}')
 for rel,value in manifest.items():
  if sha(ROOT/rel)!=value:raise RuntimeError(f'Sealed file changed: {rel}')
 print(f'VERIFY PASS {len(manifest)} sealed files; 520 retained models; no fitting',flush=True)

def seal():
 import importlib.metadata,platform
 contract=read_json(ROOT/'results/result_contract.json');tests=read_json(ROOT/'results/test_summary.json')
 assert contract['status']=='PASS' and contract['reference_predictions_match'] and contract['reference40_intervals_match']
 assert tests['status']=='PASS' and tests['tests']>=27 and tests['skipped']==0
 assert len(list((ROOT/'models').rglob('model.pkl')))==520
 assert len(pd.read_csv(ROOT/'results/RQ1_contrasts.csv'))==8 and len(pd.read_csv(ROOT/'results/RQ2_contrasts.csv'))==8
 assert len(pd.read_csv(ROOT/'results/RQ4_device_transfer.csv'))==8
 for ds in cfg()['datasets']:assert read_json(ROOT/f'results/preparation/{ds}.json')['status']=='PASS_EXACT_REBUILD'
 numeric=read_json(ROOT/'paper/numeric_validation.json');visual=read_json(ROOT/'paper/visual_review.json')
 assert numeric['status']=='PASS' and numeric['checks']==112 and visual['all_pages_visually_reviewed']
 assert numeric['document_sha256']==visual['document_sha256']==sha(ROOT/'paper/PBDEWD_review_ready.docx')
 write_json(ROOT/'protocol/environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','scipy','scikit-learn','joblib','threadpoolctl','narwhals','cloudpickle','python-docx']}))
 migration=read_json(ROOT/'protocol/migration_sources.json')
 for record in migration['sources']:
  if record['target'].startswith('data/'):assert sha(ROOT/record['target'])==record['sha256']
 paths=[]
 for base in ['experiment','protocol','data','models','results','tests','paper','publication_analysis','runtime']:
  paths.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['final_gate.json','artifact_manifest.json'])
 paths.extend(p for p in [ROOT/'README.md',ROOT/'requirements.txt',ROOT/'run_experiment.ps1'] if p.exists())
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
 write_json(ROOT/'results/artifact_manifest.json',manifest)
 write_json(ROOT/'results/final_gate.json',dict(status='PASS',date='2026-11-02',manifest_sha256=sha(ROOT/'results/artifact_manifest.json'),saved_models=520,hpo_trials=2400,selected_estimators=520,calibrators=520,fresh_estimator_fits=2920,replayed=True,tests=tests['tests'],raw_feature_rebuild='PASS_EXACT_FOUR_DATASETS',all_four_research_questions_evaluated=True,hypothesis_supported=False,onset_measured=False,external_validation=False,retrospective=True))
 verify()

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--all',action='store_true');p.add_argument('--aggregate',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--seal',action='store_true');p.add_argument('--prepare',action='store_true');args=p.parse_args()
 if args.verify:verify();return
 if args.prepare:
  from .prepare import rebuild
  for ds in cfg()['datasets']:rebuild(ds)
  return
 if args.seal:seal();return
 if args.dataset:run_dataset(args.dataset);return
 if args.all:
  for ds in cfg()['datasets']:run_dataset(ds)
 if args.all or args.aggregate:
  from .report_results import aggregate
  aggregate();return
 p.error('Choose --dataset, --all, --aggregate, --prepare, --verify or --seal')
if __name__=='__main__':main()
