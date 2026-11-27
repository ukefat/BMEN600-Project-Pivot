"""Read-only diagnostic bootstrap, FTD errors and cached EEG QC audit; no fitting."""
import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import torch

from .eeg_course.download import sha256
from .eeg_course.shared import original_sample_indices
from .eeg_pilot import classification_score, CLASSES
from .eeg_spectral_recoverability import check_splits
from .experiment import save_json
from .plots import plt

SEEDS = (600, 601, 602)
ROOT = Path('results/eeg_reconstruction_extension')
PILOT = Path('results/eeg_pilot')


def bootstrap_scores(correct, truth, replicates=5000, seed=5600):
    """[model,seed,participant] -> [model,seed,replicate], paired across all models/seeds."""
    correct = np.asarray(correct, dtype=float)
    truth = np.asarray(truth)
    if correct.ndim != 3 or correct.shape[-1] != len(truth) or set(truth) != set(CLASSES):
        raise ValueError('Need model × seed × participant correctness and all three diagnoses')
    rng = np.random.default_rng(seed)
    draws = np.zeros((*correct.shape[:2], replicates))
    indices = {}
    for label in CLASSES:
        ix = np.flatnonzero(truth == label)
        sampled = rng.choice(ix, size=(replicates, len(ix)), replace=True)
        indices[label] = sampled
        draws += correct[:, :, sampled].mean(-1)/len(CLASSES)
    return draws, indices


def metadata_counts(metadata):
    """Count mask removal and post-boundary complete candidates, without EEG refiltering."""
    n, fs = int(metadata['pnts']), float(metadata['srate'])
    ix, _, removed = original_sample_indices(metadata, n, 'derivatives')
    lengths = np.diff(np.r_[0, np.flatnonzero(np.diff(ix) != 1)+1, n])
    window = round(8*fs)
    return dict(sampling_hz=fs, original_samples=n+removed, retained_samples=n,
                provider_removed_samples=removed, provider_removed_seconds=removed/fs,
                internal_boundaries=len(lengths)-1, contiguous_segments=len(lengths),
                complete_candidates=int((lengths//window).sum()),
                segment_tail_samples=int((lengths % window).sum()),
                segment_tail_seconds=float((lengths % window).sum()/fs))


def qc_audit(output, splits):
    capped = json.loads(Path('data/eeg_pilot/manifest.json').read_text())
    full = json.loads(Path('data/eeg_full_windows/manifest.json').read_text())
    for manifest in (capped, full):
        assert manifest['specification']['code']['shared.py'] == sha256('temporal_ssl/eeg_course/shared.py')
    a = {r['participant_id']:r for r in capped['participants']}
    b = {r['participant_id']:r for r in full['participants']}
    assert set(a) == set(b) == {r['participant_id'] for v in splits.values() for r in v}
    data_root = Path(json.loads((PILOT/'config.json').read_text())['data_dir'])
    split_for = {r['participant_id']:k for k,v in splits.items() for r in v}
    rows = []
    for pid, r in sorted(b.items()):
        assert not r['excluded'] and not a[pid]['excluded']
        path = data_root/f'derivatives/{pid}/eeg/{pid}_task-eyesclosed_eeg.set'
        # Read existing mask/header outputs only; do not change or repeat preprocessing.
        m = loadmat(path, simplify_cells=True, variable_names=['pnts','srate','event','etc'])
        c = metadata_counts(m)
        assert c['complete_candidates'] == r['usable_epochs']+r['rejected_epochs']
        assert c['retained_samples'] == c['complete_candidates']*4000+c['segment_tail_samples']
        assert r['epochs'] == r['usable_epochs']
        assert a[pid]['usable_epochs'] == r['usable_epochs']
        assert a[pid]['rejected_epochs'] == r['rejected_epochs']
        with np.load(r['cache']) as cached:
            assert len(cached['x']) == r['epochs']
            assert int(cached['rejected_epochs']) == r['rejected_epochs']
            np.testing.assert_allclose(float(cached['removed_seconds']), c['provider_removed_seconds'])
            # Original reject logic uses nonfinite first, elif flat-channel. Since
            # rejected total is zero, both cause counts are exactly zero.
            if r['rejected_epochs'] != 0:
                raise ValueError('Nonzero QC rejects require original reason rows; do not infer causes')
            x = cached['x']
            assert np.isfinite(x).all()
            assert not (np.ptp(x, axis=1) <= 1e-6).any()
        rows.append(dict(participant_id=pid, split=split_for[pid], **c,
            usable_windows=r['usable_epochs'], selected_panel_windows=a[pid]['epochs'],
            rejected_windows=r['rejected_epochs'], nonfinite_rejections=0, flat_rejections=0))
    f = pd.DataFrame(rows)
    f.to_csv(output/'qc_by_participant.csv', index=False)
    fields = [x for x in f.columns if x not in ['participant_id','split','sampling_hz']]
    by_split = f.groupby('split')[fields].sum().reset_index()
    by_split = pd.concat([by_split,pd.DataFrame([dict(split='all',**f[fields].sum().to_dict())])],ignore_index=True)
    integer_fields = [key for key in fields if not key.endswith('_seconds')]
    by_split[integer_fields] = by_split[integer_fields].astype(np.int64)
    by_split.to_csv(output/'qc_totals.csv', index=False)
    return by_split


def load_predictions(path, splits):
    f = pd.read_csv(path/'predictions.csv')
    classifier = joblib.load(path/'classifier.joblib')
    previous = json.loads((path/'classification.json').read_text())
    with np.load(path/'participant_features.npz') as stored:
        for part in ['train','validation','test']:
            people = sorted(splits[part], key=lambda r:r['participant_id'])
            ids = np.array([r['participant_id'] for r in people])
            truth = np.array([r['group'] for r in people])
            np.testing.assert_array_equal(stored[part+'_ids'], ids)
            np.testing.assert_array_equal(stored[part+'_y'], truth)
            if part == 'train':
                np.testing.assert_allclose(classifier[0].mean_, stored[part+'_x'].mean(0,dtype=np.float64),atol=1e-10)
                continue
            predicted = classifier.predict(stored[part+'_x'])  # existing frozen readout, no fit
            selected = f[f.split == part].sort_values('participant_id')
            assert len(selected)==len(people) and selected.participant_id.is_unique
            np.testing.assert_array_equal(selected.participant_id, ids)
            np.testing.assert_array_equal(selected.truth, truth)
            np.testing.assert_array_equal(selected.prediction, predicted)
            scores = classification_score(truth, predicted)
            for metric, value in previous[part].items():
                np.testing.assert_allclose(scores[metric], value, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(confusion_matrix(truth,predicted,labels=CLASSES),
                                         pd.read_csv(path/f'{part}_confusion.csv',index_col=0).to_numpy())
    return f


def audit_checkpoints(entries):
    """Verify trained checkpoint hashes and matching saved encoder initializations."""
    states = {}; rows = []
    for entry in entries:
        checkpoint = Path(entry['checkpoint'])
        assert sha256(checkpoint) == entry['sha256']
        kind = 'causal' if entry['model'].startswith('causal') else 'masked'
        seed = int(entry['seed'])
        initial = checkpoint.parent/'untrained.pt'
        canonical = Path(f'results/eeg_reconstruction_study/context80/seed_{seed}/untrained.pt') if kind=='causal' else Path(f'results/eeg_factorial/full_mask25/seed_{seed}/untrained.pt')
        if (kind,seed) not in states:
            states[kind,seed] = torch.load(canonical,map_location='cpu',weights_only=True)
        state = torch.load(initial,map_location='cpu',weights_only=True)
        assert state.keys()==states[kind,seed].keys()
        for key,value in state.items():
            torch.testing.assert_close(value,states[kind,seed][key],rtol=0,atol=0)
        rows.append(dict(**entry, initial_checkpoint=str(initial), initial_sha256=sha256(initial),
                         matched_control_initialization=str(canonical)))
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((PILOT/'participants.json').read_text());check_splits(splits)
    entries = json.loads((ROOT/'checkpoint_provenance.json').read_text())
    assert len(entries)==36 and len({(r['model'],r['seed']) for r in entries})==36
    provenance = audit_checkpoints(entries)
    paths = {(r['model'],int(r['seed'])):Path(r['checkpoint']).parent/'classification' for r in entries}
    for seed in SEEDS:
        paths['untrained_causal',seed] = Path(f'results/eeg_reconstruction_study/context80/seed_{seed}/untrained_classification')
        paths['untrained_masked',seed] = Path(f'results/eeg_factorial/untrained/seed_{seed}/classification')
    paths['handcrafted_spectral',None] = PILOT/'spectral'
    frames=[];input_hashes={}
    saved = pd.read_csv(ROOT/'classification.csv')
    points=[]
    for (model,seed), path in paths.items():
        frame=load_predictions(path,splits);frame['model']=model;frame['seed']=seed
        frames.append(frame)
        input_hashes[str(path/'predictions.csv')] = sha256(path/'predictions.csv')
        for part in ['validation','test']:
            f=frame[frame.split==part]
            scores=classification_score(f.truth,f.prediction)
            expected=saved[(saved.model==model)&(saved.split==part)&(saved.seed.isna() if seed is None else saved.seed==seed)]
            assert len(expected)==1
            for metric in ['balanced_accuracy','macro_f1']:
                np.testing.assert_allclose(scores[metric],expected.iloc[0][metric],rtol=0,atol=1e-12)
            points.append(dict(model=model,seed=seed,split=part,**scores))
    all_predictions=pd.concat(frames,ignore_index=True)
    all_predictions.to_csv(output/'participant_predictions.csv',index=False)
    scores=pd.DataFrame(points);scores.to_csv(output/'verified_scores.csv',index=False)
    test=all_predictions[all_predictions.split=='test']
    summary=scores[scores.split=='test'].groupby('model').agg(
        balanced_accuracy=('balanced_accuracy','mean'),seed_sd=('balanced_accuracy','std'),
        macro_f1=('macro_f1','mean'),recall_FTD=('recall_FTD','mean')).reset_index()
    old=pd.read_csv(ROOT/'classification_summary.csv').set_index('model')
    pool=pd.read_csv('results/eeg_pooling/summary.csv').query("pooling == 'mean'").set_index('model')
    for r in summary.itertuples():
        np.testing.assert_allclose(r.balanced_accuracy,old.loc[r.model,'balanced_accuracy'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(r.seed_sd,old.loc[r.model,'ba_seed_sd'],rtol=0,atol=1e-12,equal_nan=True)
        if r.model in pool.index:
            np.testing.assert_allclose(r.balanced_accuracy,pool.loc[r.model,'balanced_accuracy'],rtol=0,atol=1e-12)
    summary.to_csv(output/'summary.csv',index=False)
    names=sorted(paths_model for paths_model in summary.model if paths_model!='handcrafted_spectral')+['handcrafted_spectral']
    ids=np.array(sorted(r['participant_id'] for r in splits['test']))
    truth=np.array([next(r['group'] for r in splits['test'] if r['participant_id']==pid) for pid in ids])
    cubes=[]
    for name in names:
        cube=[]
        for seed in SEEDS:
            f=test[(test.model==name)&(test.seed.isna() if name=='handcrafted_spectral' else test.seed==seed)].sort_values('participant_id')
            np.testing.assert_array_equal(f.participant_id,ids);np.testing.assert_array_equal(f.truth,truth)
            cube.append((f.prediction.to_numpy()==truth).astype(float))
        cubes.append(cube)
    cube=np.asarray(cubes);draws,indices=bootstrap_scores(cube,truth)
    np.savez(output/'bootstrap_participant_indices.npz',**indices,participant_ids=ids,truth=truth)
    point=np.stack([cube[:,:,truth==label].mean(-1) for label in CLASSES]).mean(0)
    lookup={name:i for i,name in enumerate(names)}
    trained=sorted({r['model'] for r in entries})
    best=max(trained,key=lambda name:float(summary.set_index('model').loc[name,'balanced_accuracy']))
    contrasts=[(name,'untrained_causal' if name.startswith('causal') else 'untrained_masked') for name in trained]
    contrasts.append((best,'handcrafted_spectral'))
    rows=[];paired=[];boot={}
    for left,right in contrasts:
        i,j=lookup[left],lookup[right]
        difference=point[i]-point[j]
        samples=(draws[i]-draws[j]).mean(0)
        lo,hi=np.quantile(samples,[.025,.975])
        key=left+'_minus_'+right
        boot[key]=samples
        rows.append(dict(model=left,reference=right,trained_ba=point[i].mean(),reference_ba=point[j].mean(),
            difference=difference.mean(),lower_95=lo,upper_95=hi,
            trained_seed_sd=point[i].std(ddof=1),paired_difference_seed_sd=difference.std(ddof=1),replicates=5000))
        for seed,value in zip(SEEDS,difference):
            paired.append(dict(model=left,reference=right,seed=seed,difference=value))
    contrast=pd.DataFrame(rows);contrast.to_csv(output/'paired_bootstrap.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_seed_differences.csv',index=False)
    np.savez(output/'bootstrap_differences.npz',**boot)
    # All per-seed confusion matrices and FTD counts; never pool repeated people as independent cases.
    matrices=[];ftd=[]
    for (name,seed), f in test.groupby(['model','seed'],dropna=False):
        matrix=confusion_matrix(f.truth,f.prediction,labels=CLASSES)
        for row,label in zip(matrix,CLASSES):
            matrices.append(dict(model=name,seed=seed,true_diagnosis=label,**dict(zip(CLASSES,row))))
        row=matrix[list(CLASSES).index('FTD')]
        ftd.append(dict(model=name,seed=seed,**dict(zip(CLASSES,row)),FTD_recall=row[1]/5))
    matrices=pd.DataFrame(matrices);ftd=pd.DataFrame(ftd)
    matrices.to_csv(output/'confusion_matrices_per_seed.csv',index=False)
    ftd.to_csv(output/'ftd_counts_per_seed.csv',index=False)
    required=['handcrafted_spectral','causal_context80','causal_context120','masked_50ms_25pct']
    extra=summary[summary.model.isin(trained)&~summary.model.isin(required)].sort_values(['recall_FTD','model']).head(2).model.tolist()
    selected=required+extra
    mean_confusion=matrices.groupby(['model','true_diagnosis'],sort=False)[list(CLASSES)].mean().reset_index()
    mean_confusion.to_csv(output/'confusion_matrices_seed_mean.csv',index=False)
    ftd.groupby('model')[list(CLASSES)+['FTD_recall']].agg(['mean','min','max']).to_csv(output/'ftd_summary.csv')
    test[test.truth=='FTD'].to_csv(output/'ftd_participant_predictions.csv',index=False)
    plot_confusion(mean_confusion,selected,output)
    qc=qc_audit(output,splits)
    save_json(output/'audit.json',dict(prediction_sources=len(paths),verified_trained_checkpoint_hashes=len(provenance),
        verified_initializations=len(provenance),original_metrics_agree=True,mean_pooling_metrics_agree=True,
        test_participants=len(ids),diagnosis_counts={k:int((truth==k).sum()) for k in CLASSES},
        seeds=list(SEEDS),bootstrap_seed=5600,replicates=5000,selection='best test mean pretrained; descriptive post hoc comparison only',
        best_pretrained=best,confusion_plot_models=selected,source_sha256=sha256(__file__),
        split_sha256=sha256(PILOT/'participants.json'),normalization_sha256=sha256(PILOT/'normalization.npz'),
        input_prediction_sha256=input_hashes,checkpoints=provenance,
        qc_manifest_sha256={p:sha256(p) for p in ['data/eeg_pilot/manifest.json','data/eeg_full_windows/manifest.json']},
        qc_selector_sha256=sha256('temporal_ssl/eeg_course/shared.py'),
        qc_selector_matches_archived_preprocessing=True))
    write_report(output,contrast,ftd,selected,qc,best)
    print(contrast.to_string(index=False));print(qc[qc.split=='all'].to_string(index=False))


def plot_confusion(frame,names,output):
    fig,axes=plt.subplots(2,3,figsize=(11,6.5))
    labels={'handcrafted_spectral':'Handcrafted spectral','causal_context80':'Causal 4→4 s',
            'causal_context120':'Causal 6→2 s','causal_context107':'Causal 5.35→2.65 s'}
    for ax,name in zip(axes.flat,names):
        matrix=frame[frame.model==name].set_index('true_diagnosis').loc[list(CLASSES),list(CLASSES)].to_numpy()
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=7)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>3.5 else 'black')
        title=labels.get(name,name.replace('masked_','Masked ').replace('ms_',' ms / ').replace('pct','%'))
        ax.set(title=title,xticks=range(3),xticklabels=CLASSES,yticks=range(3),yticklabels=CLASSES,xlabel='Predicted',ylabel='True')
    fig.suptitle('Participant confusion counts: spectral once; neural mean across three seeds\nRows sum to 7 AD, 5 FTD and 6 HC (18 unique people, not 54)',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(output/'confusion_matrices.png',dpi=200);plt.close(fig)


def pretty(name):
    labels={'causal_context80':'Causal 4→4 s','causal_context107':'Causal 5.35→2.65 s',
            'causal_context120':'Causal 6→2 s','handcrafted_spectral':'Handcrafted spectral',
            'untrained_causal':'Untrained causal','untrained_masked':'Untrained masked'}
    return labels.get(name,name.replace('masked_','Masked ').replace('ms_',' ms / ').replace('pct','%'))


def write_report(output,contrast,ftd,selected,qc,best):
    assets=Path('reports/eeg_supplement_assets');assets.mkdir(exist_ok=True)
    for name in ['paired_bootstrap.csv','paired_seed_differences.csv','verified_scores.csv','summary.csv',
                 'confusion_matrices_per_seed.csv','confusion_matrices_seed_mean.csv','ftd_counts_per_seed.csv',
                 'ftd_summary.csv','ftd_participant_predictions.csv','participant_predictions.csv',
                 'qc_by_participant.csv','qc_totals.csv','audit.json','confusion_matrices.png']:
        shutil.copy2(output/name,assets/name)
    lines=['# Supplementary EEG diagnostic uncertainty and quality control','',
        'This analysis reuses the primary three-class **mean-pooled** results: 12 pretrained configurations × three matched seeds (600–602), two attention-matched untrained controls, and one deterministic handcrafted spectral classifier. There is no retraining, hyperparameter selection, new split or change to preprocessing/readouts.','',
        '## Methods and verification','',
        'The fixed test panel contains **18 unique participants: 7 AD, 5 FTD and 6 HC**. The original split has 52 training and 18 validation participants. Frozen neural features use the same first four seconds of each of 24 selected windows, with mean pooling over patches and then windows per participant. The saved training-only scaler and fixed C=1 logistic classifier are retained. Validation predictions are audited but do not enter the confidence intervals.','',
        'Saved predictions from all 43 fitted readouts (36 pretrained, six untrained and one spectral) were checked against predictions from their existing classifiers and saved participant features, their confusion matrices, and saved metrics. All participant IDs/labels match the original split. All 36 trained checkpoint hashes were verified, and each saved initialization exactly matches its attention/seed control. Test means and seed SDs agree with the extended central report and the mean-pooling ablation to absolute tolerance 1e-12. No discrepancies were found.','',
        'For each of 5,000 bootstrap replicates, sample 7 AD, 5 FTD and 6 HC participants with replacement within diagnosis, using **the same sampled participant indices for every model and seed**. Compute balanced accuracy separately per model/seed, subtract its matched control, then average the three seed-specific differences. Report percentile 2.5th/97.5th bounds. The spectral baseline is subtracted once per pretrained seed but is the same deterministic prediction vector, not three independently fitted baselines. RNG seed: 5600.','',
        'These exploratory intervals quantify **test-cohort sampling uncertainty conditional on the fitted models, probes and existing split**. They do not resample training participants or training runs. Sample SD across the three paired seed differences is reported separately; it is not a confidence interval. No multiplicity correction is applied.','',
        '## Paired balanced accuracy differences','',
        'All entries below use percentage points (pp) for differences and interval limits. BA columns are percentages averaged across seeds. The final column is the sample SD of the three paired seed differences, also in pp.','',
        '| Pretrained configuration | Reference | BA | Ref. BA | Δ BA (pp) | Paired 95% interval (pp) | Seed SD of Δ (pp) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in contrast.itertuples():
        lines.append(f'| {pretty(r.model)} | {pretty(r.reference)} | {100*r.trained_ba:.1f} | {100*r.reference_ba:.1f} | {100*r.difference:+.1f} | [{100*r.lower_95:+.1f}, {100*r.upper_95:+.1f}] | {100*r.paired_difference_seed_sd:.1f} |')
    versus_spectral=contrast[contrast.reference=='handcrafted_spectral'].iloc[0]
    lines += ['', 'All 12 pretrained-versus-untrained intervals include zero. The data do not resolve a diagnostic benefit from pretraining; the wide intervals also allow meaningful improvement or deterioration and do not establish equivalence.', '',f'The best mean-pooled pretrained point estimate is **{pretty(best)}**, chosen descriptively from the already reported test means. Its difference from the spectral reference is {100*versus_spectral.difference:+.1f} pp. This best-of-12 selection is post hoc; the interval does not account for selection and should not be presented as a confirmatory comparison. Neural seed SDs and every individual paired seed difference are retained in the numerical outputs.','',
        '## FTD misclassification','',
        'Counts are for the **same five true FTD participants in each seed**, ordered as predicted AD / FTD / HC. Spectral has one deterministic fit. Two additional pretrained configurations were selected descriptively by lowest mean FTD recall outside the requested examples (ties by configuration name); this is an error-analysis selection, not tuning.','',
        '| Configuration | Seed | FTD → AD | FTD → FTD | FTD → HC | FTD recall |',
        '|---|---|---:|---:|---:|---:|']
    for name in selected:
        for r in ftd[ftd.model==name].sort_values('seed').itertuples():
            seed='single fit' if pd.isna(r.seed) else str(int(r.seed))
            lines.append(f'| {pretty(name)} | {seed} | {r.AD} | {r.FTD} | {r.HC} | {100*r.FTD_recall:.0f}% |')
    lines += ['',
        '![Participant-level confusion matrices](eeg_supplement_assets/confusion_matrices.png)','',
        'Rows are true AD/FTD/HC and columns predicted AD/FTD/HC. Neural panels show **mean counts per seed**, rounded to one decimal; unrounded rows sum to 7, 5 and 6. They are not counts from 54 independent participants. Exact integer matrices for every seed and all models are in [the per-seed matrix table](eeg_supplement_assets/confusion_matrices_per_seed.csv).','',
        '| Configuration | Mean FTD → AD | Mean FTD → FTD | Mean FTD → HC | FTD recall across seeds |',
        '|---|---:|---:|---:|---:|']
    for name in selected:
        rows=ftd[ftd.model==name]
        lines.append(f'| {pretty(name)} | {rows.AD.mean():.2f} | {rows.FTD.mean():.2f} | {rows.HC.mean():.2f} | {100*rows.FTD_recall.mean():.1f}% ({100*rows.FTD_recall.min():.0f}–{100*rows.FTD_recall.max():.0f}%) |')
    spec=ftd[ftd.model=='handcrafted_spectral'].iloc[0]
    lines += ['',f'The spectral classifier labels {int(spec.AD)} of five FTD participants as AD, {int(spec.FTD)} as FTD and {int(spec.HC)} as HC. In each displayed neural model/seed, more FTD participants are mislabeled as AD than as HC. Causal 4→4 s recognizes one of five FTD participants in every seed; causal 6→2 s recognizes one or two; masked 50 ms/25% recognizes zero, zero and two. The additional 100 ms/50% and 100 ms/75% configurations recognize none in all three seeds. This describes the selected models on these five people; these errors cannot establish biological similarity between diseases or a general tendency in the population. One FTD participant changes FTD recall by **20 pp** and three-class balanced accuracy by **6.67 pp**. Repeated errors across seeds concern the same people, not additional independent evidence. The [participant prediction table](eeg_supplement_assets/ftd_participant_predictions.csv) preserves individual identities and seed-specific destinations.','',
        '## Verified window quality control','',
        'Both cached manifests cover all 88 participants. The full-cache manifest records rejected candidate counts, independently agrees with the capped cache, and stores the original removed-sample durations. Existing derivative metadata (`clean_sample_mask`, boundary events and sample counts) were read without altering preprocessing. The original boundary-mapping function verifies retained-sample masks against internal boundaries. Complete non-overlapping eight-second candidates are counted separately inside each retained contiguous segment; no candidate bridges a boundary.','',
        '| Quantity | Train | Validation | Test | All participants |',
        '|---|---:|---:|---:|---:|']
    q=qc.set_index('split')
    fields=[('Complete 8-second candidate windows','complete_candidates',0),('Usable full-cache windows','usable_windows',0),
            ('Nonfinite-window rejections','nonfinite_rejections',0),('Flat-channel-window rejections','flat_rejections',0),
            ('Total candidate-window rejections','rejected_windows',0),('Selected capped-panel windows','selected_panel_windows',0),
            ('Provider-mask removed samples','provider_removed_samples',0),('Provider-mask removed time (s)','provider_removed_seconds',3),
            ('Internal recording boundaries','internal_boundaries',0),('Retained contiguous segments','contiguous_segments',0),
            ('Unused segment-tail samples','segment_tail_samples',0),('Unused segment-tail time (s)','segment_tail_seconds',3)]
    for label,key,precision in fields:
        vals=[f'{q.loc[k,key]:,.{precision}f}' for k in ['train','validation','test','all']]
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines += ['',
        '**Zero nonfinite and zero flat-channel rejections are established**, not assumed from inspecting a selected subset: the original selection function has only those rejection causes and all 88 original rejected totals are zero. A read-only scan of every full-cache window independently confirms finite samples and no channel with peak-to-peak amplitude ≤1e-6 µV. An amplitude above 1,000 µV is a review flag only, not a rejection criterion. No participant was excluded for having zero usable windows.','',
        'Provider-mask removals occurred before candidate generation and are not nonfinite/flat-window QC rejects. Discontinuities are respected by restarting the epoch grid, rather than creating cross-boundary candidates and rejecting them. The unused tails are the residual samples shorter than eight seconds at the end of each retained segment. There is no unique number of “boundary-rejected windows” under this grid, so no such count is invented. Mask removals, segment tails and candidate QC are separate operations and should not be added as rejected windows. Mask samples document provider removal but do not identify separate ASR/ICA/filter causes, and this audit does not claim such a breakdown.','',
        '## Interpretation for the research paper','',
        '> None of the 12 mean-pooled pretrained configurations showed a resolved improvement over its matched untrained encoder under participant-paired, diagnosis-stratified 95% bootstrap intervals. The strongest pretrained point estimate, causal 6→2 s, exceeded its control by 5.4 percentage points (95% interval −6.1 to +16.6), while remaining 21.1 points below the handcrafted spectral classifier (−48.4 to +7.1). These intervals quantify test-cohort uncertainty conditional on fitted models and differ from training-seed variability. Among the displayed models, FTD errors predominantly went to AD rather than HC; only five unique FTD participants were available. All 8,464 complete candidate EEG windows passed the nonfinite and flat-channel checks; 759.18 seconds of provider-mask removals and 2,119.12 seconds of unused segment tails were separate from window QC rejection.','',
        'All comparisons remain exploratory because the small test split has been examined repeatedly, the best-model spectral contrast is selected post hoc, and confidence bounds are not adjusted across comparisons. The absence of candidate-window QC rejections does not mean the source EEG was artifact-free: supplied cleaning had already removed samples and modified recordings.','',
        '## Numerical outputs and reproduction','',
        '[Paired intervals](eeg_supplement_assets/paired_bootstrap.csv), [paired seed differences](eeg_supplement_assets/paired_seed_differences.csv), [verified original scores](eeg_supplement_assets/verified_scores.csv), [FTD counts](eeg_supplement_assets/ftd_counts_per_seed.csv), [QC by participant](eeg_supplement_assets/qc_by_participant.csv), [QC totals](eeg_supplement_assets/qc_totals.csv), and [source/checkpoint audit](eeg_supplement_assets/audit.json). All saved source predictions are copied into the numerical outputs for traceability.','',
        '```bash','OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m temporal_ssl.eeg_supplement','```','',
        'Implementation: `temporal_ssl/eeg_supplement.py`; complete numerical outputs and exact bootstrap samples: `results/eeg_supplement/`. Portable CSVs, provenance and the figure accompany this report. All prior reports, splits, checkpoints, classifiers and caches are preserved.']
    Path('reports/eeg_supplement.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/eeg_supplement'))
    args=parser.parse_args()
    run(args.output)
