import hashlib
from .common import np,pd
def reserve(user):
    return int.from_bytes(hashlib.sha256(("MEREI_GLOBAL_UNSEEN_V1|" + user).encode()).digest()[:8], "big") % 10000 < 2500

def select_split(data, split):
    lower, upper = pd.Timestamp(split["lower"]), pd.Timestamp(split["upper"])
    train = data.loc[~data.reserved_unseen & data.available_at.lt(lower) & data.decision_at.le(lower)].copy()
    evaluation = data.loc[data.window_start.ge(lower) & data.available_at.lt(upper) & data.decision_at.gt(lower) & data.decision_at.le(upper)].copy()
    known_users = set(train.user_id)
    included = evaluation.reserved_unseen | evaluation.user_id.isin(known_users)
    excluded = int((~included).sum())
    evaluation = evaluation.loc[included].copy()
    evaluation["cohort"] = np.where(evaluation.reserved_unseen, "UNSEEN", "KNOWN")
    if set(train.row_id) & set(evaluation.row_id) or set(train.user_id) & set(evaluation.loc[evaluation.cohort.eq("UNSEEN"), "user_id"]):
        raise RuntimeError("Train/evaluation or unseen identity contamination")
    if train.label.nunique() != 2:
        raise RuntimeError(f"Training needs both classes for fixed split {split['name']}")
    return train, evaluation, excluded

def split_by_time(data,lower,upper):
 lower,upper=pd.Timestamp(lower),pd.Timestamp(upper)
 return data.loc[data.window_start.ge(lower)&data.available_at.lt(upper)&data.decision_at.gt(lower)&data.decision_at.le(upper)].copy()

def inner_blocks(train):
 """Label-free cumulative row mass60/80 cutoffs on observed training days."""
 days=sorted(pd.to_datetime(train.decision_at.unique()))
 if len(days)<5: raise RuntimeError('At least5 observed training days required')
 counts=train.groupby('decision_at').size().reindex(days).cumsum().to_numpy()/len(train)
 candidates=list(range(1,len(days)-2))
 i=min(candidates,key=lambda k:(abs(counts[k]-.6),k))
 j=min(range(i+1,len(days)-1),key=lambda k:(abs(counts[k]-.8),k))
 a,b=days[i],days[j]
 fitting=train.loc[train.available_at.lt(a)&train.decision_at.le(a)].copy()
 selection=split_by_time(train,a,b)
 calibration=train.loc[train.window_start.ge(b)&train.decision_at.gt(b)].copy()
 refit=train.loc[train.available_at.lt(b)&train.decision_at.le(b)].copy()
 ids=[set(fitting.row_id),set(selection.row_id),set(calibration.row_id)]
 if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise RuntimeError('Inner temporal row overlap')
 for part in [fitting,selection,calibration,refit]:
  if part.label.nunique()!=2:raise RuntimeError(f'Fixed inner block requires both classes: cutoffs{a},{b};size{len(part)}')
 return fitting,selection,calibration,refit,dict(selection_lower=str(a),calibration_lower=str(b),
  rule='Label-free cumulative row mass60/80percent on observed training days with boundary purge; no outcome-based cutoff search',
  fitting_rows=len(fitting),selection_rows=len(selection),calibration_rows=len(calibration),refit_rows=len(refit),
  calibration_positive_users=int(calibration.loc[calibration.label.eq(1),'user_id'].nunique()),
  calibration_negative_users=int(calibration.loc[calibration.label.eq(0),'user_id'].nunique()))
