from __future__ import annotations
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/python_packages'))
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
def cfg():return read_json(ROOT/'protocol/plan.json')
def inner_blocks(train):
 from .splits import inner_blocks as inner
 return inner(train)
def load_daily(ds):return pd.read_parquet(ROOT/f'data/prepared/{ds}.parquet')
def reps(ds):return read_json(ROOT/f'data/prepared/{ds}_features.json')['experiment_representations']
