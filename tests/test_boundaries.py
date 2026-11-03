import unittest
from experiment.common import *
from experiment.splits import select_split,inner_blocks
from experiment.factorial import factorial_columns
from experiment.models import fit_predict
class Boundaries(unittest.TestCase):
 def test_unseen_users_excluded_and_windows_purged(self):
  for ds in cfg()['datasets']:
   data=load_daily(ds)
   for split in cfg()['splits']:
    train,ev,_=select_split(data,split)
    self.assertFalse(set(train.user_id)&set(ev.loc[ev.cohort.eq('UNSEEN'),'user_id']))
    self.assertFalse(set(train.row_id)&set(ev.row_id))
    self.assertLess(train.available_at.max(),ev.window_start.min())
 def test_inner_boundaries_do_not_use_labels(self):
  train,_,_=select_split(load_daily('NonEnc_desktop'),cfg()['splits'][1]);changed=train.copy();changed['label']=1-changed.label
  a=inner_blocks(train);b=inner_blocks(changed)
  for i in range(4):self.assertEqual(a[i].row_id.tolist(),b[i].row_id.tolist())
 def test_movement_requires_static_and_adaptive(self):
  ds='NonEnc_desktop';designs,_=factorial_columns(load_daily(ds),{'experiment_representations':reps(ds)})
  for n in range(8):
   movement=[c for c in designs[f'FACTORIAL_{n}'] if c.startswith('adaptive_movement_abs_z__')]
   self.assertEqual(bool(movement),bool(n&1 and n&2))
 def test_preprocessing_does_not_use_evaluation_values(self):
  x=pd.DataFrame({'x':[0.,1.,np.nan,3.,4.,5.],'z':[2.,1.,4.,np.nan,6.,7.]});y=[0,1,0,1,0,1]
  a=pd.DataFrame({'x':[10.,20.],'z':[5.,7.]});b=a*10000
  _,_,p=fit_predict(x,y,a,'LR',{'seed':7});_,_,q=fit_predict(x,y,b,'LR',{'seed':7})
  np.testing.assert_array_equal(p.medians,q.medians);np.testing.assert_allclose(p.estimator.coef_,q.estimator.coef_,rtol=0,atol=0)
 def test_identifiers_and_diagnostic_labels_not_predictors(self):
  for ds in cfg()['datasets']:
   for rep in cfg()['representations']:
    self.assertFalse(set(reps(ds)[rep])&{'label','user_id','row_id','timestamp','diagnostic_past_label','reserved_unseen'})
if __name__=='__main__':unittest.main()
