from __future__ import annotations
import hashlib,json,math,os,pickle,time,warnings
from pathlib import Path
from . import common as C, models
np,pd=C.np,C.pd
ROOT=C.ROOT
SCOPE="RETROSPECTIVE_FOUR_RQ_REPRODUCTION"
MODELS=("LR","HGB")
def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=str, allow_nan=False).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    C.write_json(temporary, value)
    temporary.replace(path)

def verify_hashes(records):
    for name, expected in records.items():
        if not (ROOT/name).is_file() or C.sha(ROOT/name) != expected:
            raise RuntimeError(f"Nested-training frozen artifact changed: {name}")

def row_hash(frame):
    columns = [c for c in ("row_id", "user_id", "timestamp", "available_at", "decision_at", "label") if c in frame]
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()

def temporal_parts(train, fractions=None, cutpoints=None):
    """Delegate to the frozen shared label-free row-mass temporal partition."""
    if fractions is not None or cutpoints is not None:
        raise ValueError("Inner boundaries are exclusively defined by completion_v2.common.inner_blocks")
    fit, selection, calibration, refit, metadata = C.inner_blocks(train)
    frames = dict(inner_fit=fit, inner_select=selection, calibration=calibration, refit=refit)
    if set(refit.row_id) & set(calibration.row_id):
        raise RuntimeError("Calibration leaked into final estimator refit")
    if any(not frame.label.isin([0,1]).all() for frame in frames.values()):
        raise ValueError("Only observed binary training labels are allowed")
    report = dict(metadata, envelope_purging=True, calibration_not_in_estimator_refit=True,
        outer_evaluation_labels_used=False, parts={name:dict(rows=len(frame),users=int(frame.user_id.nunique()),
        positives=int(frame.label.sum()),positive_users=int(frame.loc[frame.label.eq(1),'user_id'].nunique()),
        negative_users=int(frame.loc[frame.label.eq(0),'user_id'].nunique()),row_sha256=row_hash(frame),
        earliest_window_start=str(frame.window_start.min()),latest_available_at=str(frame.available_at.max()),
        earliest_decision=str(frame.decision_at.min()),latest_decision=str(frame.decision_at.max()))
        for name,frame in frames.items()})
    return frames, report

class PlattCalibratedPipeline:
    """Picklable raw ranking pipeline plus an independently fitted sigmoid."""
    def __init__(self, estimator_pipeline, calibrator=None, probability_clip=1e-6, thread_limit=2):
        self.estimator_pipeline = estimator_pipeline
        self.calibrator = calibrator
        self.probability_clip = float(probability_clip)
        self.thread_limit = int(thread_limit)

    def raw_scores(self, X):
        return models.predict_scores(self.estimator_pipeline, X)

    def predict_scores(self, X):
        scores = self.raw_scores(X)
        if self.calibrator is None:
            return scores
        if len(scores) == 0:
            return np.empty(0, dtype=float)
        clipped = np.clip(scores, self.probability_clip, 1-self.probability_clip)
        logit = np.log(clipped/(1-clipped)).reshape(-1,1)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=self.thread_limit):
            result = self.calibrator.predict_proba(logit)[:,list(self.calibrator.classes_).index(1)]
        if not np.isfinite(result).all():
            raise RuntimeError("Calibrated prediction is not finite")
        return np.asarray(result, dtype=float)

def partition_rows(parts):
    chunks=[]
    for role,frame in parts.items():
        part=frame[["row_id","user_id","timestamp","label","window_start","available_at","decision_at"]].copy()
        part["role"]=role
        chunks.append(part)
    return pd.concat(chunks,ignore_index=True)

def fit_platt(estimator_pipeline, calibration_X, calibration_y, *, C_value=1.0, seed=20261022, n_jobs=2, probability_clip=1e-6):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    labels = np.asarray(calibration_y)
    if labels.ndim != 1 or len(labels) != len(calibration_X) or not np.isin(labels,[0,1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Platt calibration requires disjoint held-out rows with both binary classes")
    raw = models.predict_scores(estimator_pipeline, calibration_X)
    if not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
        raise ValueError("Platt input must be supervised probability scores")
    clipped = np.clip(raw,float(probability_clip),1-float(probability_clip))
    features = np.log(clipped/(1-clipped)).reshape(-1,1)
    calibrator = LogisticRegression(C=float(C_value), penalty="l2", solver="lbfgs", max_iter=1000,
                                    class_weight=None, random_state=int(seed))
    with threadpool_limits(limits=int(n_jobs)):
        calibrator.fit(features,labels)
    pipeline = PlattCalibratedPipeline(estimator_pipeline, calibrator, probability_clip, n_jobs)
    metadata = dict(method="UNWEIGHTED_HELD_OUT_PLATT_ON_LOGIT_RAW_PROBABILITY", C=float(C_value),
                    rows=len(labels), positive_rows=int(labels.sum()), negative_rows=int(len(labels)-labels.sum()),
                    class_weight=None, probability_clip=float(probability_clip), seed=int(seed),
                    slope=float(calibrator.coef_[0,0]), intercept=float(calibrator.intercept_[0]),
                    outer_eval_labels_used=False, guarantee="Calibration estimated on a finite held-out temporal block; not a guarantee of future calibration")
    return pipeline, metadata

def validate_candidates(model, candidates):
    if model not in MODELS or not isinstance(candidates,list) or len(candidates) != 12:
        raise ValueError("Exactly12 frozen configurations are required for each supervised algorithm")
    hashes = [json_hash(c) for c in candidates]
    if len(set(hashes)) != 12:
        raise ValueError("HPO candidate configurations must be distinct")
    if any(not isinstance(c,dict) for c in candidates):
        raise ValueError("Each frozen HPO candidate must be a parameter mapping")
    return candidates

def _binding(base, model, columns, candidate=None):
    return dict(base=base, model=model, columns=list(columns), candidate=candidate)

def train_nested(train, evaluation, columns, model_name, config, folder, bindings):
    """Run exactly12 inner trials, one selected refit and one held-out calibrator.

    Every checkpoint binds trial parameters, temporal row hashes and source
    code/plan hashes. Restoring after an interrupted run never reruns valid fits.
    """
    from sklearn.metrics import average_precision_score
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    candidates = validate_candidates(model_name,config["candidate_grids"][model_name])
    parts, partition = temporal_parts(train)
    if evaluation.row_id.isna().any() or evaluation.row_id.duplicated().any() or set(train.row_id)&set(evaluation.row_id):
        raise ValueError("Outer evaluation rows must be unique and disjoint from training")
    if evaluation.window_start.min() <= train.available_at.max():
        raise ValueError("Outer training/evaluation source envelopes overlap")
    if not columns or set(columns)&{"label","user_id","row_id","timestamp","available_at","decision_at","reserved_unseen","cohort"}:
        raise ValueError("Predictor list must be nonempty and exclude targets and identifiers")
    # Ground-truth-free evaluation signature: labels do not guide any fit.
    eval_signature = hashlib.sha256(pd.util.hash_pandas_object(evaluation[["row_id","user_id","timestamp","available_at","decision_at"]],index=False).values.tobytes()).hexdigest()
    base = dict(inputs=bindings, train_row_sha256=row_hash(train), evaluation_identity_sha256=eval_signature,
                partition=partition, model_config=config, columns=list(columns))
    receipt_path = folder/"checkpoint.json"
    if receipt_path.exists():
        receipt = C.read_json(receipt_path)
        if receipt["binding"] != _binding(base,model_name,columns):
            raise RuntimeError("Existing nested-fit checkpoint differs from frozen configuration/rows")
        verify_hashes(receipt["files"])
        return C.read_json(folder/"fit.json"), True
    trials = []
    for index,candidate in enumerate(candidates):
        path = folder/"trials"/f"trial_{index:02d}.json"
        prediction_path = path.with_suffix(".parquet")
        binding = _binding(base,model_name,columns,candidate)
        if path.exists():
            record = C.read_json(path)
            if record["binding"] != binding:
                raise RuntimeError("Existing HPO trial differs from frozen candidate/inputs")
            seal = C.read_json(path.with_suffix(".sha256.json"))
            if seal["sha256"] != C.sha(path):
                raise RuntimeError("Existing HPO trial checkpoint content changed")
            if seal["predictions_sha256"] != C.sha(prediction_path):
                raise RuntimeError("Existing HPO trial predictions changed")
        else:
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                score,details,_ = models.fit_predict(parts["inner_fit"][columns].copy(),parts["inner_fit"].label,
                    parts["inner_select"][columns].copy(),model_name,
                    dict(seed=config["seed"],n_jobs=config.get("n_jobs",2),model_params=candidate))
            objective = float(average_precision_score(parts["inner_select"].label,score))
            if not math.isfinite(objective):
                raise RuntimeError("Inner candidate AP is not finite")
            record = dict(binding=binding,trial_index=index,parameters=candidate,selection_AP=objective,
                          fitting_seconds=time.perf_counter()-started,fit_details=details,
                          warnings=[str(w.message) for w in caught],objective="INNER_SELECTION_AP",outer_labels_used=False,
                          trial_estimator_retained=False,trial_estimator_disposition="DISCARDED_AFTER_SELECTION_SCORE; selected configuration refitted independently")
            prediction_path.parent.mkdir(parents=True,exist_ok=True)
            trial_predictions=parts["inner_select"][["row_id","user_id","timestamp","label","available_at","decision_at"]].copy()
            trial_predictions["score"]=score
            trial_predictions.to_parquet(prediction_path,index=False)
            atomic_json(path,record)
            atomic_json(path.with_suffix(".sha256.json"),dict(sha256=C.sha(path),predictions_sha256=C.sha(prediction_path),binding_sha256=json_hash(binding)))
        trials.append(record)
    selected = min(trials,key=lambda r:(-r["selection_AP"],r["trial_index"]))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal_raw,fit_metadata,raw_pipeline = models.fit_predict(parts["refit"][columns].copy(),parts["refit"].label,
            parts["calibration"][columns].copy(),model_name,
            dict(seed=config["seed"],n_jobs=config.get("n_jobs",2),model_params=selected["parameters"]))
        pipeline,calibration = fit_platt(raw_pipeline,parts["calibration"][columns],parts["calibration"].label,
            C_value=config.get("calibration_C",1.0),seed=config["seed"],n_jobs=config.get("n_jobs",2),
            probability_clip=config.get("probability_clip",1e-6))
    raw_scores = pipeline.raw_scores(evaluation[columns])
    scores = pipeline.predict_scores(evaluation[columns])
    with (folder/"model.pkl").open("wb") as stream:
        pickle.dump(pipeline,stream,pickle.HIGHEST_PROTOCOL)
    with (folder/"model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    if not np.allclose(scores,restored.predict_scores(evaluation[columns]),rtol=1e-10,atol=1e-12):
        raise RuntimeError("Saved nested calibrated model replay failed")
    predicted = evaluation[["row_id","user_id","timestamp","label","available_at","decision_at","cohort"]].copy()
    predicted["score_raw"],predicted["score"] = raw_scores,scores
    predicted.to_parquet(folder/"predictions.parquet",index=False)
    calibration_rows = parts["calibration"][["row_id","user_id","timestamp","label","available_at","decision_at"]].copy()
    calibration_rows["score_raw"] = cal_raw
    calibration_rows["score"] = pipeline.predict_scores(parts["calibration"][columns])
    calibration_rows.to_parquet(folder/"calibration_predictions.parquet",index=False)
    partition_rows(dict(outer_train=train,**parts)).to_parquet(folder/"partition_rows.parquet",index=False)
    record = dict(model=model_name,scope=SCOPE,trial_count=12,selected_trial_index=selected["trial_index"],
        selected_parameters=selected["parameters"],selection_AP=selected["selection_AP"],
        tie_break="FIRST_FROZEN_CANDIDATE_INDEX",partition=partition,fit_details=fit_metadata,
        calibration=calibration,columns=list(columns),outer_eval_rows=len(evaluation),
        estimator_refit_rows=len(parts["refit"]),calibration_rows=len(parts["calibration"]),
        outer_labels_used_for_selection_or_fitting=False,calibration_labels_in_estimator_refit=False,
        immediate_pickle_replay_verified=True,fit_seconds=time.perf_counter()-started,
        model_fits=dict(hpo=12,selected_estimator=1,platt=1),warnings=[str(w.message) for w in caught])
    C.write_json(folder/"fit.json",record)
    files = {str(path.relative_to(ROOT)):C.sha(path) for path in [folder/"model.pkl",folder/"predictions.parquet",
             folder/"calibration_predictions.parquet",folder/"partition_rows.parquet",folder/"fit.json",*sorted((folder/"trials").glob("trial_*"))]}
    atomic_json(receipt_path,dict(binding=_binding(base,model_name,columns),files=files,status="COMPLETE"))
    return record,False

def fit_fixed(train, evaluation, columns, modelname, config, folder, bindings):
    """Fit a fixed-parameter diagnostic estimator and held-out calibrator.

    API matches train_nested's (record,resumed) receipt. A full root plan or
    a flat config is accepted; model_params is an explicitly frozen mapping.
    The training part and calibration definitions are identical to main HPO.
    """
    if modelname not in MODELS:
        raise ValueError("Fixed calibrated fits support LR/HGB/RF/XGB only")
    if "calibration" in config:
        effective = dict(seed=config["seed"],n_jobs=config["n_jobs"],
            calibration_C=config["calibration"]["regularization_C"],
            probability_clip=config["calibration"]["epsilon"],model_params=config.get("model_params",{}))
    else:
        effective = dict(config)
    if not columns or set(columns)&{"label","row_id","user_id","timestamp","available_at","decision_at","cohort"}:
        raise ValueError("Fixed predictors cannot include targets or observation identity")
    if set(train.row_id)&set(evaluation.row_id) or evaluation.row_id.duplicated().any():
        raise ValueError("Fixed fit requires disjoint unique outer evaluation rows")
    if evaluation.window_start.min() <= train.available_at.max():
        raise ValueError("Fixed fit outer source envelopes overlap")
    parts,partition = temporal_parts(train)
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    checkpoint = folder/"checkpoint.json"
    binding = dict(inputs=bindings,model=modelname,columns=list(columns),config=effective,
        train_row_sha256=row_hash(train),evaluation_row_sha256=row_hash(evaluation.drop(columns="label")),partition=partition)
    if checkpoint.exists():
        saved = C.read_json(checkpoint)
        if saved["binding"] != binding:
            raise RuntimeError("Fixed fit checkpoint configuration changed")
        verify_hashes(saved["files"])
        return C.read_json(folder/"fit.json"),True
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw,details,estimator = models.fit_predict(parts["refit"][columns].copy(),parts["refit"].label,
            parts["calibration"][columns].copy(),modelname,
            dict(seed=effective["seed"],n_jobs=effective.get("n_jobs",2),model_params=effective.get("model_params",{})))
        pipeline,calibration = fit_platt(estimator,parts["calibration"][columns],parts["calibration"].label,
            C_value=effective.get("calibration_C",1.),seed=effective["seed"],n_jobs=effective.get("n_jobs",2),
            probability_clip=effective.get("probability_clip",1e-6))
    predicted = evaluation[["row_id","user_id","timestamp","label","available_at","decision_at","cohort"]].copy()
    predicted["score_raw"] = pipeline.raw_scores(evaluation[columns])
    predicted["score"] = pipeline.predict_scores(evaluation[columns])
    predicted.to_parquet(folder/"predictions.parquet",index=False)
    partition_rows(dict(outer_train=train,**parts)).to_parquet(folder/"partition_rows.parquet",index=False)
    with (folder/"model.pkl").open("wb") as stream:
        pickle.dump(pipeline,stream,pickle.HIGHEST_PROTOCOL)
    with (folder/"model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    if not np.allclose(predicted.score,restored.predict_scores(evaluation[columns]),rtol=1e-10,atol=1e-12):
        raise RuntimeError("Fixed calibrated saved-model replay mismatch")
    record = dict(model=modelname,columns=list(columns),scope=SCOPE,fixed_parameters=True,hyperparameter_selection=False,
        partition=partition,fit_details=details,calibration=calibration,model_fits=dict(estimator=1,platt=1),
        outer_eval_rows=len(evaluation),estimator_refit_rows=len(parts["refit"]),calibration_rows=len(parts["calibration"]),
        calibration_labels_in_estimator_refit=False,outer_labels_used_for_selection_or_fitting=False,
        immediate_pickle_replay_verified=True,fit_seconds=time.perf_counter()-started,
        warnings=[str(w.message) for w in caught])
    C.write_json(folder/"fit.json",record)
    atomic_json(checkpoint,dict(binding=binding,files={str((folder/name).relative_to(ROOT)):C.sha(folder/name)
        for name in ("model.pkl","predictions.parquet","partition_rows.parquet","fit.json")},status="COMPLETE"))
    return record,False
