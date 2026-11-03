"""Check the actual Word table values against the rerun CSV files."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.common import *
from docx import Document
def main():
 docpath=ROOT/'paper/PBDEWD_review_ready.docx';tables=Document(docpath).tables;checks=[]
 def num(s):return float(s.replace('−','-').replace('+','').replace(' ',''))
 def check(table,row,col,written,expected):
  passed=abs(num(written)-float(expected))<=0.00005000001
  checks.append(dict(table=table,row=row,column=col,written=written,expected_full_precision=float(expected),passed=passed))
 def name(v):return v.replace(' desktop','_desktop').replace(' smartphone','_smartphone')
 for index,rq in [(2,'RQ1'),(3,'RQ2')]:
  ref=pd.read_csv(ROOT/f'results/{rq}_contrasts.csv')
  for n,row in enumerate(tables[index].rows[1:],1):
   v=[c.text for c in row.cells];r=ref.loc[ref.dataset.eq(name(v[0]))&ref.model.eq(v[1])].iloc[0]
   for j,key in [(2,'lower_value'),(3,'higher_value'),(4,'effect')]:check(index+1,n,j,v[j],r[key])
   if rq=='RQ2':
    lo,hi=v[5].strip('[]').split(';');check(4,n,5,lo,r.ci95_lower);check(4,n,6,hi,r.ci95_upper)
 ref=pd.read_csv(ROOT/'results/RQ3_factorial_effects.csv');ref=ref.loc[ref.split.eq('FINAL')&ref.cohort.eq('UNSEEN')&ref.score_type.eq('score')]
 for n,row in enumerate(tables[4].rows[1:],1):
  v=[c.text for c in row.cells]
  for j,factor in enumerate(['STATIC','ADAPT','DRIFT'],2):
   r=ref.loc[ref.dataset.eq(name(v[0]))&ref.model.eq(v[1])&ref.factor.eq(factor)].iloc[0];check(5,n,j,v[j],r.mean_AP_difference)
 ref=pd.read_csv(ROOT/'results/RQ4_device_transfer.csv')
 for n,row in enumerate(tables[5].rows[1:],1):
  v=[c.text for c in row.cells];left,right=[s.strip() for s in v[0].split('→')]
  if right in ['desktop','smartphone']:right=left.split()[0]+' '+right
  r=ref.loc[ref.source.eq(name(left))&ref.target.eq(name(right))&ref.model.eq(v[1])].iloc[0]
  for j,key in [(2,'R1_AP'),(3,'R4_AP'),(4,'delta_AP')]:check(6,n,j,v[j],r[key])
 result=dict(status='PASS' if all(r['passed'] for r in checks) else 'FAIL',checks=len(checks),failed=[r for r in checks if not r['passed']],document_sha256=sha(docpath),values=checks)
 assert len(checks)==112
 write_json(ROOT/'paper/numeric_validation.json',result)
 if result['status']!='PASS':raise RuntimeError(result['failed'])
 print('MANUSCRIPT NUMERIC PASS:112/112 table values match fresh results')
if __name__=='__main__':main()
