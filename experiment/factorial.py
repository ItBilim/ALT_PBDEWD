from itertools import product
from .common import np,pd
FORBIDDEN={"row_id","user_id","timestamp","available_at","decision_at","window_start","label","reserved_unseen","cohort"}
def _row_identity(frame,name):
    if 'row_id' not in frame or frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError(f'{name} needs unique nonmissing row_id')

def factorial_columns(daily,metadata):
    """Eight S/A/D designs with movement as an explicit S×A interaction.

    Static movement is abs(EWMA-static_mean)/static_scale; removing the direct
    static block while retaining movement would leave a hidden static anchor.
    Movement therefore enters only when both STATIC and ADAPT are present.
    Full111 exactly matches the frozen R4 representation;000 matches R1.
    """
    _row_identity(daily,'daily')
    reps=metadata.get('experiment_representations',metadata.get('representations',{}))
    base=list(reps.get('R1_HISTORY',reps.get('R1',[])))
    full=list(reps.get('R4_DRIFT',reps.get('R4',[])))
    if not base or not full or not set(base).issubset(full):
        raise ValueError('Frozen R1/R4 representation contract is required')
    blocks={
        'STATIC':[c for c in full if c.startswith('static_abs_z__')],
        'ADAPT':[c for c in full if c.startswith('adaptive_abs_z__')],
        'MOVEMENT_INTERACTION':[c for c in full if c.startswith('adaptive_movement_abs_z__')],
        'DRIFT':[c for c in full if c.startswith(('drift_abs_z__','drift_change_abs__'))],
    }
    if any(not v for v in blocks.values()):
        raise ValueError('All four declared frozen component blocks are required')
    identified=set(base).union(*map(set,blocks.values()))
    if identified!=set(full) or any(c not in daily for c in identified):
        raise ValueError('Frozen full representation has unknown or unavailable columns')
    if identified&FORBIDDEN or any(not pd.api.types.is_numeric_dtype(daily[c]) for c in full):
        raise ValueError('Frozen predictor lists must contain only numeric feature columns')
    if np.isinf(daily[full].to_numpy(dtype=float,na_value=np.nan)).any():
        raise ValueError('Infinite frozen feature values are forbidden')
    designs={}
    for static,adapt,drift in product((0,1),repeat=3):
        keep=set(base)
        if static:keep.update(blocks['STATIC'])
        if adapt:keep.update(blocks['ADAPT'])
        if static and adapt:keep.update(blocks['MOVEMENT_INTERACTION'])
        if drift:keep.update(blocks['DRIFT'])
        # Preserve the exact frozen order; no outcome-driven feature selection.
        mask=static+2*adapt+4*drift
        designs[f'FACTORIAL_{mask}']=[c for c in full if c in keep]
    designs={f'FACTORIAL_{mask}':designs[f'FACTORIAL_{mask}'] for mask in range(8)}
    if designs['FACTORIAL_7']!=full or designs['FACTORIAL_0']!=base:
        raise ValueError('Frozen base/full column order must be preserved')
    return designs,{
        'blocks':blocks,'movement_requires':['STATIC','ADAPT'],'mask_bits':{'STATIC':1,'ADAPT':2,'DRIFT':4},
        'interaction_interpretation':'Static/adapt contrasts include their explicitly modeled movement interaction when both components are present',
        'normal_history_verified':False,'label_used_in_design':False,
    }
